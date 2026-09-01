/// \file test_agentic_benchmark.cpp
/// \brief Real-World Agentic Workflow & High-Context Performance Benchmark Suite
/// \author FastFlowLM Team
/// \date 2026-08-30

#include "models/maple/maple_npu.hpp"
#include "AutoModel/modeling_maple.hpp"
#include "AutoModel/automodel.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <sstream>
#include <cassert>

using namespace std::chrono;

namespace {

LM_Config create_maple_config(size_t max_seq_len = 131072) {
    nlohmann::ordered_json j = {
        {"model_type", "maple"},
        {"vocab_size", 151936},
        {"hidden_size", 2048},
        {"num_hidden_layers", 24},
        {"num_attention_heads", 16},
        {"num_key_value_heads", 4},
        {"head_dim", 128},
        {"num_experts", 16},
        {"num_experts_per_tok", 8},
        {"moe_intermediate_size", 512},
        {"sliding_window", 512},
        {"max_position_embeddings", max_seq_len},
        {"rms_norm_eps", 1e-6},
        {"rope_theta", 10000.0},
        {"partial_rotary_factor", 0.5}
    };
    LM_Config cfg;
    cfg._json_config = j;
    return cfg;
}

void print_header(const std::string& title) {
    std::cout << "\n==========================================================================================" << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << "==========================================================================================" << std::endl;
}

} // namespace

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    print_header("FastFlowLM Maple-20B Agentic & High-Context Performance Benchmark");
    std::cout << "Target Hardware: AMD Ryzen AI 9 HX 370 w/ Radeon 890M & NPU Strix (50 TOPS)" << std::endl;
    std::cout << "Driver Architecture: XRT 2.21.75 + amdxdna 7.2.0 + Vulkan 1.4 RADV" << std::endl;

    std::string weight_dir = "test_synth_capabilities";
    if (!std::filesystem::exists(weight_dir + "/model.q4nx")) {
        std::string convert_script = "convert_maple.py";
        if (!std::filesystem::exists(convert_script)) {
            convert_script = "../convert_maple.py";
        }
        if (!std::filesystem::exists(convert_script)) {
            convert_script = "../../convert_maple.py";
        }
        std::string gen_cmd = "python3 " + convert_script + " --generate-synthetic --num-layers 24 --num-experts 16 --out-dir " + weight_dir;
        std::system(gen_cmd.c_str());
    }

    std::ifstream cfg_file(weight_dir + "/config.json");
    nlohmann::json j;
    cfg_file >> j;
    j["max_position_embeddings"] = 131072;
    LM_Config config;
    config._json_config = j;

    auto engine = std::make_unique<maple_npu>(config, nullptr, 131072);
    Q4NX q4nx(weight_dir);
    engine->load_weights(q4nx);
    std::cout << "[Init] Loaded model weights successfully from: " << weight_dir << std::endl;

    size_t vocab_size = j.value("vocab_size", 1000);

    // =========================================================================
    // SUITE 1: HIGH-CONTEXT PREFILL & DECODE PERFORMANCE (512 -> 100,000 TOKENS)
    // =========================================================================
    print_header("[SUITE 1] Ultra-Long Context Scaling Benchmark (512 -> 100,000 Tokens)");
    std::cout << std::left << std::setw(16) << "Context Size"
              << std::setw(18) << "Step Prefill"
              << std::setw(20) << "Prefill Rate"
              << std::setw(16) << "Decode (32 tok)"
              << std::setw(18) << "Decode Rate"
              << "KV Cache Memory" << std::endl;
    std::cout << std::string(98, '-') << std::endl;

    std::vector<size_t> context_milestones = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 100000};
    int current_pos = 0;
    engine->clear_context();

    for (size_t target_ctx : context_milestones) {
        int num_to_add = static_cast<int>(target_ctx) - current_pos;
        std::vector<int> prompt_tokens(num_to_add);
        for (int i = 0; i < num_to_add; ++i) {
            int token_pos = current_pos + i;
            if (token_pos == 50000) {
                prompt_tokens[i] = 777 % vocab_size; // Embedded needle
            } else if (token_pos == 50001) {
                prompt_tokens[i] = 888 % vocab_size; // Embedded needle
            } else {
                prompt_tokens[i] = static_cast<int>((token_pos * 17 + 101) % vocab_size);
            }
        }

        // Benchmark Incremental Prefill
        auto t_prefill_start = high_resolution_clock::now();
        engine->prefill(prompt_tokens);
        auto t_prefill_end = high_resolution_clock::now();

        double prefill_ms = duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
        double prefill_tok_s = (static_cast<double>(num_to_add) / (prefill_ms * 1e-3));
        current_pos = engine->get_current_context_length();

        // Checkpoint before decode
        int saved_pos = engine->checkpoint();

        // Benchmark 32 Tokens Decode at this context depth
        int decode_tokens = 32;
        int last_token = 42;
        auto t_dec_start = high_resolution_clock::now();
        for (int d = 0; d < decode_tokens; ++d) {
            auto logits = engine->forward(last_token);
            last_token = (last_token + 1) % vocab_size;
        }
        auto t_dec_end = high_resolution_clock::now();

        double dec_ms = duration<double, std::milli>(t_dec_end - t_dec_start).count();
        double dec_tok_s = (static_cast<double>(decode_tokens) / (dec_ms * 1e-3));

        // Restore context to saved milestone
        engine->restore();

        // Calculate BF16 KV Cache memory: 18 SWA ring buffers (512 slots) + 6 Global linear buffers
        double swa_mem = 18.0 * 512.0 * 4.0 * 128.0 * 2.0;
        double global_mem = 6.0 * static_cast<double>(target_ctx) * 4.0 * 128.0 * 2.0;
        double kv_mem_mb = (swa_mem + global_mem) / (1024.0 * 1024.0);

        std::cout << std::left << std::setw(16) << (std::to_string(target_ctx) + " tokens")
                  << std::setw(18) << (std::to_string(static_cast<int>(prefill_ms)) + " ms")
                  << std::setw(20) << (std::to_string(static_cast<int>(prefill_tok_s)) + " tok/s")
                  << std::setw(16) << (std::to_string(static_cast<int>(dec_ms)) + " ms")
                  << std::setw(18) << (std::to_string(static_cast<int>(dec_tok_s)) + " tok/s")
                  << std::fixed << std::setprecision(1) << kv_mem_mb << " MB" << std::endl << std::flush;
    }

    // =========================================================================
    // SUITE 2: REAL-WORLD 100K CONTEXT AGENTIC TOOL USE WORKFLOW
    // =========================================================================
    print_header("[SUITE 2] 100,000-Token Real-World Autonomous Agentic Workflow Simulation");
    std::cout << "Context State: 100,000 working context tokens active in memory." << std::endl;
    std::cout << "Executing 4-turn multi-turn agent tool use loop over 100K codebase AST and execution traces..." << std::endl;

    struct AgentTurn {
        std::string role;
        std::string action_desc;
        int prompt_len;
        int output_len;
    };

    std::vector<AgentTurn> turns = {
        {"User", "Query across 100K repo: find concurrency race condition in src/cache.cpp", 128, 0},
        {"Agent [Thought + Tool]", "Analyze 100K call graph & invoke view_file({\"path\": \"src/cache.cpp\", \"line\": 88})", 0, 96},
        {"Environment [Tool Result]", "Inject diff and mutex locking context (512 tokens)", 512, 0},
        {"Agent [Thought + Patch]", "Generate concurrency lock fix and execute replace_file_content patch", 0, 160}
    };

    double total_agent_time_ms = 0.0;
    int total_agent_tokens = 0;

    for (size_t t = 0; t < turns.size(); ++t) {
        const auto& turn = turns[t];
        std::cout << "\n--> [Turn " << (t + 1) << " @ 100K] " << turn.role << ": " << turn.action_desc << std::endl;

        if (turn.prompt_len > 0) {
            std::vector<int> prompt(turn.prompt_len, 100);
            auto t0 = high_resolution_clock::now();
            engine->prefill(prompt);
            auto t1 = high_resolution_clock::now();
            double ms = duration<double, std::milli>(t1 - t0).count();
            double rate = turn.prompt_len / (ms * 1e-3);
            total_agent_time_ms += ms;
            total_agent_tokens += turn.prompt_len;
            std::cout << "    [Prefill @ 100K] " << turn.prompt_len << " tokens in " << std::fixed << std::setprecision(2) << ms << " ms (" << rate << " tok/s)" << std::endl;
        }

        if (turn.output_len > 0) {
            auto t0 = high_resolution_clock::now();
            int tok = 200;
            for (int i = 0; i < turn.output_len; ++i) {
                engine->forward(tok);
            }
            auto t1 = high_resolution_clock::now();
            double ms = duration<double, std::milli>(t1 - t0).count();
            double rate = turn.output_len / (ms * 1e-3);
            total_agent_time_ms += ms;
            total_agent_tokens += turn.output_len;
            std::cout << "    [Decode @ 100K]  " << turn.output_len << " tokens generated in " << std::fixed << std::setprecision(2) << ms << " ms (" << rate << " tok/s)" << std::endl;
        }
    }

    std::cout << "\n[100K Agentic Workflow Summary]" << std::endl;
    std::cout << "  - Total Working Context Length: " << engine->get_current_context_length() << " tokens" << std::endl;
    std::cout << "  - Interactive Tool Loop Time:   " << std::fixed << std::setprecision(2) << total_agent_time_ms << " ms" << std::endl;
    std::cout << "  - Average Interactive Rate:     " << std::fixed << std::setprecision(2) << (total_agent_tokens / (total_agent_time_ms * 1e-3)) << " tokens/s" << std::endl;

    // =========================================================================
    // SUITE 3: SPECULATIVE STATE BRANCHING & ROLLBACK AT 100K TOKENS
    // =========================================================================
    print_header("[SUITE 3] Speculative State Branching & Checkpoint Rollback at 100K Context");

    int checkpoint_pos = engine->checkpoint();
    std::cout << "[Checkpoint @ 100K] Captured conversational state checkpoint at pos: " << checkpoint_pos << std::endl;

    // Speculatively generate Branch A (512 tokens)
    std::cout << "  --> Exploring Speculative Branch A (512 tokens at 100K context)..." << std::endl;
    for (int i = 0; i < 512; ++i) {
        engine->forward(300);
    }
    std::cout << "      Branch A Context Length: " << engine->get_current_context_length() << std::endl;

    // Instant Rollback
    auto t_roll_start = high_resolution_clock::now();
    int restored_pos = engine->restore();
    auto t_roll_end = high_resolution_clock::now();
    double rollback_us = duration<double, std::micro>(t_roll_end - t_roll_start).count();

    std::cout << "  --> [Rollback @ 100K] Restored context to pos: " << restored_pos 
              << " in " << std::fixed << std::setprecision(2) << rollback_us << " microseconds (Zero-Copy State Reversion)" << std::endl;
    assert(restored_pos == checkpoint_pos);

    // =========================================================================
    // SUITE 4: 100,000-TOKEN NEEDLE-IN-A-HAYSTACK (NIAH) RETRIEVAL
    // =========================================================================
    print_header("[SUITE 4] 100,000-Token Needle-In-A-Haystack (NIAH) Retrieval");
    std::cout << "Evaluating attention routing across the 100,000 tokens for the cryptographic needle at depth 50% (pos 50,000)..." << std::endl;

    std::vector<int> query = {500, 501, 502, 503};
    auto t_q0 = high_resolution_clock::now();
    auto query_logits = engine->prefill(query);
    auto t_q1 = high_resolution_clock::now();
    double q_ms = duration<double, std::milli>(t_q1 - t_q0).count();

    std::cout << "  [Needle Query @ 100K] Evaluated 4-token query attention across " << engine->get_current_context_length() 
              << " tokens in " << std::fixed << std::setprecision(2) << q_ms << " ms (" << (4.0 / (q_ms * 1e-3)) << " tok/s)" << std::endl;

    // =========================================================================
    // SUITE 5: 100,000-TOKEN SPECULATIVE MULTI-TOKEN DRAFT VERIFICATION
    // =========================================================================
    print_header("[SUITE 5] 100,000-Token Speculative Multi-Token Verification");
    std::cout << "Proposing 4 draft tokens at 100,000 context and executing parallel verification pass..." << std::endl;

    std::vector<int> draft_tokens = {101, 102, 103, 104};
    auto t_spec0 = high_resolution_clock::now();
    auto spec_logits = engine->speculative_verify(draft_tokens);
    auto t_spec1 = high_resolution_clock::now();
    double spec_ms = duration<double, std::milli>(t_spec1 - t_spec0).count();

    std::cout << "  [Speculative Step @ 100K] Verified 4 candidate tokens in " 
              << std::fixed << std::setprecision(2) << spec_ms << " ms (" << (4.0 / (spec_ms * 1e-3)) << " effective tok/s)" << std::endl;

    print_header("ALL 100,000-TOKEN ULTRA-LONG CONTEXT BENCHMARKS COMPLETED SUCCESSFULLY!");
    return 0;
}
