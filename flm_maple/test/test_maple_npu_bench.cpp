#include <iostream>
#include <fstream>
#include <chrono>
#include <memory>
#include "AutoModel/modeling_maple.hpp"
#include "model_list.hpp"
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"

flm_rt::device npu_device_global;

int main(int argc, char* argv[]) {
    std::cout << "=================================================================" << std::endl;
    std::cout << "=== FastFlowLM AMD Ryzen AI NPU Benchmark Suite: Maple-20B ===" << std::endl;
    std::cout << "=================================================================" << std::endl;

    arg_utils::po::options_description desc("NPU Benchmark Options");
    desc.add_options()
        ("model,m", arg_utils::po::value<std::string>()->default_value("maple:20b"), "Model tag in model_list.json")
        ("model-path,p", arg_utils::po::value<std::string>()->default_value(""), "Direct model path (overrides tag)")
        ("prompt", arg_utils::po::value<std::string>()->default_value("Explain the difference between sliding window attention and global attention in 3 points."), "Prompt text")
        ("gen-tokens,g", arg_utils::po::value<int>()->default_value(64), "Max generation tokens")
        ("max-len,l", arg_utils::po::value<int>()->default_value(4096), "Context window limit")
        ("effort,e", arg_utils::po::value<std::string>()->default_value("low"), "Reasoning effort (low, medium, high)")
        ("preemption", arg_utils::po::value<bool>()->default_value(false), "Enable hardware preemption");

    arg_utils::po::variables_map vm;
    try {
        arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc), vm);
        arg_utils::po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "Invalid arguments: " << e.what() << std::endl;
        return 1;
    }

    std::string tag = vm["model"].as<std::string>();
    std::string custom_model_path = vm["model-path"].as<std::string>();
    std::string prompt_text = vm["prompt"].as<std::string>();
    int max_gen_tokens = vm["gen-tokens"].as<int>();
    int max_ctx_len = vm["max-len"].as<int>();
    std::string reasoning_effort = vm["effort"].as<std::string>();
    bool preemption = vm["preemption"].as<bool>();

    std::string model_dir = utils::get_models_directory();
    std::string model_list_path = "/home/yoav/slop/maple-flm/FastFlowLM/src/model_list.json";
    model_list m_list(model_list_path, model_dir);

    std::string model_path = custom_model_path.empty() ? m_list.get_model_path(tag) : custom_model_path;
    auto [canonical_tag, model_info] = m_list.get_model_info(tag);

    std::cout << "[NPU Setup] Initializing AMD Ryzen AI NPU device (device index 0)..." << std::endl;
    npu_device_global = flm_rt::device(0);

    std::unique_ptr<AutoModel> chat = std::make_unique<Maple>(&npu_device_global);

    std::cout << "[NPU Load] Loading Maple model from: " << model_path << std::endl;
    std::cout << "[NPU Config] Context Limit: " << max_ctx_len 
              << " | Target Generation Tokens: " << max_gen_tokens 
              << " | Reasoning Effort: " << reasoning_effort << std::endl;

    try {
        chat->load_model(model_path, model_info, max_ctx_len, preemption);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to load Maple model on NPU: " << e.what() << std::endl;
        return 1;
    }

    chat->configure_parameter("reasoning_effort", reasoning_effort);
    chat->configure_parameter("enable_think", true);
    chat->set_max_length(max_ctx_len);
    chat->set_topk(1);

    chat_meta_info_t meta_info;
    lm_uniform_input_t input;
    input.prompt = prompt_text;

    std::cout << "\n-----------------------------------------------------------------" << std::endl;
    std::cout << "[NPU Run] Executing inference on NPU hardware with prompt:\n\"" << input.prompt << "\"" << std::endl;
    std::cout << "-----------------------------------------------------------------\n" << std::endl;

    chat->start_total_timer();
    std::string response = chat->generate_with_prompt(meta_info, input, max_gen_tokens, std::cout);
    chat->stop_total_timer();

    std::cout << "\n\n=================================================================" << std::endl;
    std::cout << "=== NPU HARDWARE PERFORMANCE PROFILE ===" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << chat->show_profile() << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
