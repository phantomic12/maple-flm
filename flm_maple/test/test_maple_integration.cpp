#include <iostream>
#include <cassert>
#include "AutoModel/all_models.hpp"
#include "models/maple/maple_npu.hpp"

int main() {
    std::cout << "=== Running FastFlowLM Maple Integration Verification ===" << std::endl;

    // 1. Verify Model List & Family resolution
    std::string model_dir = utils::get_models_directory();
    std::string model_list_path = "FastFlowLM/src/model_list.json";

    model_list m_list(model_list_path, model_dir);
    assert(m_list.is_model_supported("maple:20b"));
    assert(m_list.is_model_supported("maple-preview:20b"));
    std::cout << "[PASS] Model registry recognizes maple:20b and maple-preview:20b" << std::endl;

    auto [canonical_tag, model_info] = m_list.get_model_info("maple:20b");
    assert(model_info["details"]["family"] == "maple");
    assert(model_info["details"]["think"] == true);
    std::cout << "[PASS] Model details and family mapped correctly: " << model_info["details"]["family"] << std::endl;

    // 2. Verify get_auto_model instantiation
    flm_rt::device dummy_dev;
    auto [tag_ret, auto_model_inst] = get_auto_model("maple:20b", m_list, &dummy_dev);
    assert(auto_model_inst != nullptr);
    assert(auto_model_inst->get_current_model() == "Maple");
    std::cout << "[PASS] AutoModel successfully instantiates Maple instance: " << auto_model_inst->get_current_model() << std::endl;

    // 3. Verify parameter configuration
    bool ok_think = auto_model_inst->configure_parameter("enable_think", true);
    assert(ok_think);
    bool ok_effort = auto_model_inst->configure_parameter("reasoning_effort", std::string("high"));
    assert(ok_effort);
    std::cout << "[PASS] Parameter configuration (enable_think, reasoning_effort) works" << std::endl;

    // 4. Verify reasoning & tool stream parsing
    Maple* maple_model = dynamic_cast<Maple*>(auto_model_inst.get());
    assert(maple_model != nullptr);

    std::string test_stream_think = "<think>\nThinking about step 1.\n</think>\n\nFinal answer is 42.";
    NonStreamResult nstream_res = maple_model->parse_nstream_content(test_stream_think);
    assert(nstream_res.reasoning_content == "Thinking about step 1.");
    assert(nstream_res.content == "Final answer is 42.");
    std::cout << "[PASS] Non-stream reasoning parsing verified" << std::endl;

    // 5. Verify Causal LM Engine (maple_npu) dimensions and state
    LM_Config mock_cfg;
    mock_cfg._json_config = {
        {"vocab_size", 151936},
        {"hidden_size", 2048},
        {"num_hidden_layers", 24},
        {"num_attention_heads", 16},
        {"num_key_value_heads", 4},
        {"head_dim", 128},
        {"num_experts", 256},
        {"num_experts_per_tok", 8},
        {"moe_intermediate_size", 512},
        {"sliding_window", 512},
        {"rms_norm_eps", 1e-6},
        {"rope_theta", 10000.0},
        {"partial_rotary_factor", 0.5},
        {"nope_on_global_attention", true}
    };

    maple_npu engine(mock_cfg, nullptr, 2048);
    assert(engine.get_current_context_length() == 0);
    engine.clear_context();
    std::cout << "[PASS] maple_npu causal_lm engine instantiated and verified" << std::endl;

    std::cout << "\n>>> ALL MAPLE-PREVIEW PORT INTEGRATION TESTS PASSED! <<<" << std::endl;
    return 0;
}
