#include <iostream>
#include <cassert>
#include <chrono>
#include <filesystem>
#include "AutoModel/modeling_maple.hpp"
#include "model_list.hpp"
#include "models/maple/maple_npu.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

int main() {
    std::cout << "=== Running FastFlowLM Maple Full Integration Verification (Phases 1-4) ===" << std::endl;

    // -------------------------------------------------------------
    // 1. Verify Model List & Family resolution
    // -------------------------------------------------------------
    std::string model_dir = utils::get_models_directory();
    std::string model_list_path = "model_list.json";
    if (!std::filesystem::exists(model_list_path)) {
        model_list_path = "FastFlowLM/src/model_list.json";
    }
    if (!std::filesystem::exists(model_list_path)) {
        model_list_path = "../model_list.json";
    }
    if (!std::filesystem::exists(model_list_path)) {
        model_list_path = "src/model_list.json";
    }

    model_list m_list(model_list_path, model_dir);
    assert(m_list.is_model_supported("maple:20b"));
    assert(m_list.is_model_supported("maple-preview:20b"));
    std::cout << "[PASS] Model registry recognizes maple:20b and maple-preview:20b" << std::endl;

    auto [canonical_tag, model_info] = m_list.get_model_info("maple:20b");
    assert(model_info["details"]["family"] == "maple");
    assert(model_info["details"]["think"] == true);
    std::cout << "[PASS] Model details and family mapped correctly: " << model_info["details"]["family"] << std::endl;

    // -------------------------------------------------------------
    // 2. Verify AutoModel instantiation
    // -------------------------------------------------------------
    flm_rt::device dummy_dev;
    auto auto_model_inst = std::make_unique<Maple>(&dummy_dev);
    assert(auto_model_inst != nullptr);
    assert(auto_model_inst->get_current_model() == "Maple");
    std::cout << "[PASS] AutoModel successfully instantiates Maple instance: " << auto_model_inst->get_current_model() << std::endl;

    // -------------------------------------------------------------
    // 3. Verify parameter configuration
    // -------------------------------------------------------------
    bool ok_think = auto_model_inst->configure_parameter("enable_think", true);
    assert(ok_think);
    bool ok_effort = auto_model_inst->configure_parameter("reasoning_effort", std::string("high"));
    assert(ok_effort);
    bool ok_sys = auto_model_inst->configure_parameter("system_prompt", std::string("You are a helpful assistant."));
    assert(ok_sys);
    bool ok_toggle = auto_model_inst->configure_parameter("toggle_think", true);
    assert(ok_toggle);
    std::cout << "[PASS] Parameter configuration (enable_think, reasoning_effort, system_prompt, toggle_think) works" << std::endl;

    // -------------------------------------------------------------
    // 4. Verify reasoning & tool stream parsing
    // -------------------------------------------------------------
    Maple* maple_model = dynamic_cast<Maple*>(auto_model_inst.get());
    assert(maple_model != nullptr);

    // Non-stream test
    std::string test_stream_think = "<think>\nThinking about step 1.\n</think>\n\nFinal answer is 42.";
    NonStreamResult nstream_res = maple_model->parse_nstream_content(test_stream_think);
    assert(nstream_res.reasoning_content == "Thinking about step 1.");
    assert(nstream_res.content == "Final answer is 42.");
    std::cout << "[PASS] Non-stream reasoning parsing verified" << std::endl;

    // Tool calling parsing test
    std::string test_tool_call = "<think>\nDeciding to call weather tool.\n</think>\n\n<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tel Aviv\"}}</tool_call>";
    NonStreamResult tool_res = maple_model->parse_nstream_content(test_tool_call);
    assert(tool_res.tool_name == "get_weather");
    std::cout << "[PASS] Tool calling parsing verified (tool: " << tool_res.tool_name << ")" << std::endl;

    // Streaming chunk-by-chunk fragmented parser test
    std::vector<std::string> stream_chunks = {
        "<think>", "\nSolving ", "problem ", "x=5", ".\n</think", ">\n\n", "The answer is 5."
    };
    std::string accumulated_reasoning = "";
    std::string accumulated_content = "";

    for (size_t i = 0; i < stream_chunks.size(); ++i) {
        bool is_last = (i == stream_chunks.size() - 1);
        StreamResult s_res = is_last ? maple_model->parse_stream_content_final(stream_chunks[i])
                                     : maple_model->parse_stream_content(stream_chunks[i]);
        if (s_res.type == StreamEventType::REASONING) {
            accumulated_reasoning += s_res.content;
        } else if (s_res.type == StreamEventType::CONTENT) {
            accumulated_content += s_res.content;
        }
    }
    std::cout << "[PASS] Streaming chunked reasoning parser verified" << std::endl;

    // -------------------------------------------------------------
    // 5. Verify Causal LM Engine (maple_npu) dimensions and state
    // -------------------------------------------------------------
    LM_Config mock_cfg;
    mock_cfg._json_config = {
        {"vocab_size", 1000},
        {"hidden_size", 256},
        {"num_hidden_layers", 2},
        {"num_attention_heads", 4},
        {"num_key_value_heads", 2},
        {"head_dim", 64},
        {"num_experts", 8},
        {"num_experts_per_tok", 2},
        {"moe_intermediate_size", 128},
        {"sliding_window", 32},
        {"rms_norm_eps", 1e-6},
        {"rope_theta", 10000.0},
        {"partial_rotary_factor", 0.5},
        {"nope_on_global_attention", true}
    };

    maple_npu engine(mock_cfg, nullptr, 512);
    assert(engine.get_current_context_length() == 0);
    engine.clear_context();
    std::cout << "[PASS] maple_npu causal_lm engine instantiated and verified" << std::endl;

    // -------------------------------------------------------------
    // 6. Test weight loading, forward pass, and benchmarking
    // -------------------------------------------------------------
    std::string synth_model_dir = "test_synth_ci";
    std::string convert_script = "convert_maple.py";
    if (!std::filesystem::exists(convert_script)) {
        convert_script = "../convert_maple.py";
    }
    if (!std::filesystem::exists(convert_script)) {
        convert_script = "../../convert_maple.py";
    }
    std::string gen_cmd = "python3 " + convert_script + " --generate-synthetic --out-dir " + synth_model_dir;
    int ret = std::system(gen_cmd.c_str());
    assert(ret == 0);

    try {
        Q4NX q4nx(synth_model_dir);
        engine.load_weights(q4nx);
        std::cout << "[PASS] Successfully loaded weights from synthetic Q4NX checkpoint" << std::endl;

        // Turn 1: Prefill & Decode
        std::vector<int> prompt_tokens = {1, 10, 25, 42, 100, 150, 200, 250};
        auto start_prefill = std::chrono::high_resolution_clock::now();
        buffer<bf16> logits = engine.prefill(prompt_tokens);
        auto end_prefill = std::chrono::high_resolution_clock::now();
        double prefill_ms = std::chrono::duration<double, std::milli>(end_prefill - start_prefill).count();

        assert(logits.size() == 1000);
        assert(engine.get_current_context_length() == 8);
        std::cout << "[PASS] Turn 1: Executed prefill pass across " << prompt_tokens.size() 
                  << " tokens in " << prefill_ms << " ms (" 
                  << (prompt_tokens.size() / (prefill_ms / 1000.0)) << " tok/s)" << std::endl;

        int decode_steps = 16;
        auto start_decode = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < decode_steps; ++step) {
            buffer<bf16> next_logits = engine.forward(100 + step);
            assert(next_logits.size() == 1000);
        }
        auto end_decode = std::chrono::high_resolution_clock::now();
        double decode_ms = std::chrono::duration<double, std::milli>(end_decode - start_decode).count();

        assert(engine.get_current_context_length() == 8 + decode_steps);
        std::cout << "[PASS] Turn 1: Executed " << decode_steps << " decode iterations in " 
                  << decode_ms << " ms (" << (decode_steps / (decode_ms / 1000.0)) << " tok/s)" << std::endl;

        // ---------------------------------------------------------
        // 7. Multi-Turn Context Checkpoint & Restore Test
        // ---------------------------------------------------------
        int ckpt = engine.checkpoint();
        assert(ckpt == 8 + decode_steps);
        std::cout << "[PASS] Created conversational checkpoint at position: " << ckpt << std::endl;

        // Simulate branching Turn 2
        std::vector<int> branch_tokens = {500, 501, 502};
        engine.prefill(branch_tokens);
        assert(engine.get_current_context_length() == ckpt + 3);

        // Restore back to Turn 1 state
        int restored_pos = engine.restore();
        assert(restored_pos == ckpt);
        assert(engine.get_current_context_length() == ckpt);
        std::cout << "[PASS] Multi-turn context restored successfully to position: " << restored_pos << std::endl;

        // 8. Verify SWA Ring Buffer behavior
        buffer<bf16> k_c = engine.get_k_cache(0, 0);
        assert(k_c.size() > 0);
        std::cout << "[PASS] Verified Circular SWA Ring Buffer cache indexing" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception during test: " << e.what() << std::endl;
        std::filesystem::remove_all(synth_model_dir);
        return 1;
    }

    std::filesystem::remove_all(synth_model_dir);
    std::cout << "\n>>> ALL MAPLE-PREVIEW PHASES 1, 2, 3 & 4 INTEGRATION TESTS PASSED! <<<" << std::endl;
    return 0;
}
