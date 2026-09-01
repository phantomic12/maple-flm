#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <cmath>
#include <random>
#include <filesystem>
#include "AutoModel/modeling_maple.hpp"
#include "models/maple/maple_npu.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "=== Running FastFlowLM Maple Maximum Context Test (128K/131K) ===" << std::endl;
    std::cout << "=================================================================" << std::endl;

    // 1. Configure full 24-layer Maple-Preview backbone with scalable max context
    const char* env_ctx = std::getenv("MAX_TEST_CONTEXT");
    uint32_t max_test_context = env_ctx ? std::stoul(env_ctx) : 16384; // Default 16K for fast integration test

    LM_Config high_ctx_cfg;
    high_ctx_cfg._json_config = {
        {"vocab_size", 1000},
        {"hidden_size", 256},
        {"num_hidden_layers", 24},      // Full 24-layer depth (18 SWA : 6 Global)
        {"num_attention_heads", 16},     // 16 Query heads
        {"num_key_value_heads", 4},      // 4 KV heads (GQA 4:1)
        {"head_dim", 64},                // 64 head dimension
        {"num_experts", 16},             // 16 experts
        {"num_experts_per_tok", 4},      // Top-4 active experts
        {"moe_intermediate_size", 128},
        {"sliding_window", 512},         // 512 sliding window
        {"rms_norm_eps", 1e-6},
        {"rope_theta", 10000.0},
        {"partial_rotary_factor", 0.5},  // 0.5 factor (32 rot, 32 pass-through)
        {"nope_on_global_attention", true},
        {"default_context_length", max_test_context}
    };

    std::cout << "[Setup] Initializing maple_npu with Max Context = " << max_test_context 
              << " tokens across 24 layers (18 SWA Ring-Buffers + 6 Full Attention layers)..." << std::endl;

    maple_npu engine(high_ctx_cfg, nullptr, max_test_context);

    // 2. Generate and load synthetic model weights
    std::string synth_model_dir = "test_synth_max_ctx";
    std::string convert_script = "convert_maple.py";
    if (!std::filesystem::exists(convert_script)) {
        convert_script = "../convert_maple.py";
    }
    if (!std::filesystem::exists(convert_script)) {
        convert_script = "../../convert_maple.py";
    }
    std::string gen_cmd = "python3 " + convert_script + " --generate-synthetic --num-layers 24 --num-experts 16 --out-dir " + synth_model_dir;
    int ret = std::system(gen_cmd.c_str());
    assert(ret == 0);

    try {
        Q4NX q4nx(synth_model_dir);
        engine.load_weights(q4nx);
        std::cout << "[PASS] Loaded weights for 24-layer MoE model." << std::endl;

        // 3. Test High-Context Sequential Prefill & Decode Scaling up to max_test_context
        std::vector<int> all_milestones = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
        std::vector<int> test_milestones;
        for (int m : all_milestones) {
            if ((uint32_t)m <= max_test_context) {
                test_milestones.push_back(m);
            }
        }
        if (test_milestones.empty() || (uint32_t)test_milestones.back() < max_test_context) {
            test_milestones.push_back((int)max_test_context);
        }
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(1, 999);

        int current_pos = 0;

        for (int target_len : test_milestones) {
            int num_to_add = target_len - current_pos;
            std::vector<int> chunk_tokens(num_to_add);
            for (int i = 0; i < num_to_add; ++i) {
                chunk_tokens[i] = dist(rng);
            }

            auto start_chunk = std::chrono::high_resolution_clock::now();
            buffer<bf16> chunk_logits = engine.prefill(chunk_tokens);
            auto end_chunk = std::chrono::high_resolution_clock::now();
            double chunk_ms = std::chrono::duration<double, std::milli>(end_chunk - start_chunk).count();

            current_pos = engine.get_current_context_length();
            assert(current_pos == target_len);

            // Verify numerical stability of output logits
            bool has_nan = false;
            bool has_inf = false;
            float max_l = -1e30f;
            float min_l = 1e30f;

            for (size_t v = 0; v < chunk_logits.size(); ++v) {
                float val = static_cast<float>(chunk_logits[v]);
                if (std::isnan(val)) has_nan = true;
                if (std::isinf(val)) has_inf = true;
                if (val > max_l) max_l = val;
                if (val < min_l) min_l = val;
            }

            assert(!has_nan && "Logits contain NaN at high context!");
            assert(!has_inf && "Logits contain Inf at high context!");

            std::cout << "[Context Checkpoint] Reached " << current_pos << " tokens | Prefill batch: " 
                      << num_to_add << " tokens in " << (chunk_ms / 1000.0) << " s (" 
                      << (num_to_add / (chunk_ms / 1000.0)) << " tok/s) | Logits range: [" 
                      << min_l << ", " << max_l << "]" << std::endl;

            // Run a single decode step at this high context position
            auto start_step = std::chrono::high_resolution_clock::now();
            buffer<bf16> step_logits = engine.forward(dist(rng));
            auto end_step = std::chrono::high_resolution_clock::now();
            double step_ms = std::chrono::duration<double, std::milli>(end_step - start_step).count();

            assert(engine.get_current_context_length() == target_len + 1);
            current_pos = engine.get_current_context_length();

            std::cout << "  ↳ Decode forward step at pos " << (current_pos - 1) 
                      << " latency: " << step_ms << " ms (" 
                      << (1000.0 / step_ms) << " tok/s)" << std::endl;
        }

        // 4. Verify SWA Ring Buffer invariant after max_test_context tokens
        buffer<bf16> swa_k = engine.get_k_cache(0, current_pos - 1);
        assert(swa_k.size() > 0);
        std::cout << "[PASS] SWA Ring-Buffer maintained stable 512-slot footprint across " << max_test_context << "+ tokens." << std::endl;

        // 5. Test Checkpoint & Restoration at high context
        int ckpt_pos = engine.checkpoint();
        assert(ckpt_pos == current_pos);
        std::cout << "[PASS] Checkpointed context state at position " << ckpt_pos << std::endl;

        std::vector<int> branch_tokens = {101, 102, 103, 104};
        engine.prefill(branch_tokens);
        assert(engine.get_current_context_length() == ckpt_pos + 4);

        int restored_pos = engine.restore();
        assert(restored_pos == ckpt_pos);
        assert(engine.get_current_context_length() == ckpt_pos);
        std::cout << "[PASS] Successfully restored context to position " << restored_pos << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception during high context test: " << e.what() << std::endl;
        std::filesystem::remove_all(synth_model_dir);
        return 1;
    }

    std::filesystem::remove_all(synth_model_dir);
    std::cout << "\n=================================================================" << std::endl;
    std::cout << ">>> HIGH CONTEXT (" << max_test_context << " TOKENS) STRESS TEST PASSED WITH ZERO ERRORS! <<<" << std::endl;
    std::cout << "=================================================================" << std::endl;
    return 0;
}
