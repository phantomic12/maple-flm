/// \file modeling_maple.hpp
/// \brief Maple reasoning MoE AutoModel class
/// \author FastFlowLM Team
/// \date 2026-08-29
/// \version 1.0.0
/// \note This is a header file for the Maple AutoModel class
#pragma once

#include "AutoModel/automodel.hpp"
#include "metrices.hpp"
#include "typedef.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

class Maple : public AutoModel {
private:
    bool enable_think = true;
    bool enable_tool = true;
    std::string reasoning_effort = "medium";
    
    // Token markers for reasoning
    int think_start_id = 151648;
    int think_end_id = 151649;

    void setup_tokenizer(std::string model_path);

public:
    Maple(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_info, int default_context_length = -1, bool enable_preemption = false) override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;
    
    NonStreamResult parse_nstream_content(const std::string response_text);
    StreamResult parse_stream_content(const std::string content);
    StreamResult parse_stream_content_final(const std::string content) override;

private:
    StreamResult parse_stream_content_impl(const std::string content, bool is_final);

public:
    /// \brief Configure a parameter with type-erased value
    bool configure_parameter(std::string parameter_name, const std::any& value) override {
        if (parameter_name == "enable_think") {
            try {
                this->enable_think = std::any_cast<bool>(value);
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        } else if (parameter_name == "reasoning_effort") {
            try {
                std::string effort = std::any_cast<std::string>(value);
                if (effort == "high" || effort == "medium" || effort == "low") {
                    this->enable_think = true;
                    this->reasoning_effort = effort;
                    this->extra_context["reasoning_effort"] = this->reasoning_effort;
                } else if (effort == "none") {
                    this->enable_think = false;
                } else {
                    header_print("WARNING", "Reasoning effort must be 'none', 'low', 'medium' or 'high'!");
                }
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        } else if (parameter_name == "toggle_think") {
            this->enable_think = !this->enable_think;
            return true;
        } else if (parameter_name == "system_prompt") {
            try {
                this->user_system_prompt = std::any_cast<std::string>(value);
                this->extra_context["user_system_prompt"] = this->user_system_prompt;
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }
        return AutoModel::configure_parameter(parameter_name, value);
    }
};
