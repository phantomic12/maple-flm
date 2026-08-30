/// \file modeling_maple.cpp
/// \brief Maple AutoModel class implementation
/// \author FastFlowLM Team
/// \date 2026-08-29
/// \version 1.0.0

#include "AutoModel/modeling_maple.hpp"
#include "models/maple/maple_npu.hpp"

Maple::Maple(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Maple") {}

void Maple::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);

    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    this->lm_engine = std::make_unique<maple_npu>(*this->lm_config, this->npu.get(), this->MAX_L);

    this->lm_engine->load_weights(*this->q4nx);
    this->q4nx.reset();

    this->lm_engine->clear_context();
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    this->enable_tool = true;

    sampler_config config;
    config.top_k = 20;
    config.top_p = 0.8;
    config.min_p = 0.0;
    config.temperature = 0.6;
    config.rep_penalty = 1.05;
    config.freq_penalty = 0.0;
    config.pre_penalty = 1.0f;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

void Maple::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Maple::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    inputs.extra_context["enable_thinking"] = this->enable_think;
    if (!tools.empty() && this->enable_tool) {
        inputs.tools = tools;
    }
    return this->chat_tmpl->apply(inputs);
}

bool Maple::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;

    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }

    if (!input.messages.empty()) {
        templated_text = this->apply_chat_template(input.messages, input.tools);
    } else if (!input.prompt.empty()) {
        nlohmann::ordered_json messages;
        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages);
    }

    std::vector<int> tokens = this->tokenizer->encode(templated_text);
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    int restore_idx = -1;
    maple_npu *maple_engine = dynamic_cast<maple_npu*>(this->lm_engine.get());
    if (meta_info.restore_allowed && maple_engine != nullptr) {
        restore_idx = maple_engine->restore();
        this->total_tokens = restore_idx;
        this->token_history = checkpoint_his;
    }

    bool success = this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);

    if (maple_engine != nullptr) {
        checkpoint_his = token_history;
        maple_engine->checkpoint();
    }

    return success;
}

std::string Maple::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string Maple::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->_shared_generate(meta_info, length_limit, os);
}

NonStreamResult Maple::parse_nstream_content(const std::string response_text) {
    NonStreamResult result;
    const std::string think_start = "<think>\n";
    const std::string think_end = "</think>\n\n";

    size_t start_pos = response_text.find(think_start);
    size_t end_pos = response_text.find(think_end);

    if (start_pos != std::string::npos && end_pos != std::string::npos && end_pos > start_pos) {
        size_t reasoning_start = start_pos + think_start.length();
        result.reasoning_content = response_text.substr(reasoning_start, end_pos - reasoning_start);
        result.content = response_text.substr(end_pos + think_end.length());
    } else {
        result.content = response_text;
    }

    // Check for tool call
    const std::string tool_start = "<tool_call>";
    const std::string tool_end = "</tool_call>";
    size_t t_start = result.content.find(tool_start);
    size_t t_end = result.content.find(tool_end);
    if (t_start != std::string::npos && t_end != std::string::npos && t_end > t_start) {
        std::string json_str = result.content.substr(t_start + tool_start.length(), t_end - t_start - tool_start.length());
        try {
            auto j = nlohmann::json::parse(json_str);
            if (j.contains("name")) result.tool_name = j["name"].get<std::string>();
            if (j.contains("arguments")) result.tool_args = j["arguments"].dump();
        } catch (...) {}
    }

    return result;
}

StreamResult Maple::parse_stream_content(const std::string content) {
    return parse_stream_content_impl(content, false);
}

StreamResult Maple::parse_stream_content_final(const std::string content) {
    return parse_stream_content_impl(content, true);
}

StreamResult Maple::parse_stream_content_impl(const std::string content, bool is_final) {
    buffer_ += content;
    StreamResult result;
    result.type = current_mode_;

    const std::string MARKER_THINK_START = "<think>\n";
    const std::string MARKER_THINK_END = "</think>\n\n";

    while (!buffer_.empty()) {
        if (current_mode_ == StreamEventType::CONTENT) {
            size_t think_pos = buffer_.find(MARKER_THINK_START);
            if (think_pos != std::string::npos) {
                if (think_pos > 0) {
                    result.content = buffer_.substr(0, think_pos);
                    result.type = StreamEventType::CONTENT;
                    buffer_ = buffer_.substr(think_pos);
                    return result;
                }
                buffer_ = buffer_.substr(MARKER_THINK_START.length());
                current_mode_ = StreamEventType::REASONING;
                continue;
            }
        } else if (current_mode_ == StreamEventType::REASONING) {
            size_t think_end = buffer_.find(MARKER_THINK_END);
            if (think_end != std::string::npos) {
                if (think_end > 0) {
                    result.content = buffer_.substr(0, think_end);
                    result.type = StreamEventType::REASONING;
                    buffer_ = buffer_.substr(think_end);
                    return result;
                }
                buffer_ = buffer_.substr(MARKER_THINK_END.length());
                current_mode_ = StreamEventType::CONTENT;
                continue;
            }
        }

        if (!buffer_.empty()) {
            size_t last_lt = buffer_.rfind('<');
            if (!is_final && last_lt != std::string::npos && (buffer_.length() - last_lt) <= 15) {
                if (last_lt > 0) {
                    result.content = buffer_.substr(0, last_lt);
                    result.type = current_mode_;
                    buffer_ = buffer_.substr(last_lt);
                    return result;
                } else {
                    result.type = StreamEventType::WAITING;
                    return result;
                }
            }

            result.content = buffer_;
            result.type = current_mode_;
            buffer_.clear();
            return result;
        }
        break;
    }

    result.type = StreamEventType::WAITING;
    return result;
}
