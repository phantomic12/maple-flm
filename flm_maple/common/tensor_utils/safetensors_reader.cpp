#include "tensor_utils/safe_tensors.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>

SafeTensors::SafeTensors(const std::string& model_path) : model_path(model_path) {
    _open_file();
    _load_tensors();
}

SafeTensors::~SafeTensors() {
    if (file.is_open()) file.close();
}

void SafeTensors::_open_file() {
    if (model_path.empty()) return;
    std::string path = model_path;
    if (std::filesystem::is_directory(path)) {
        if (std::filesystem::exists(path + "/model.q4nx")) path += "/model.q4nx";
        else if (std::filesystem::exists(path + "/model.safetensors")) path += "/model.safetensors";
    }
    file.open(path, std::ios::binary);
}

void SafeTensors::_load_tensors() {
    if (!file.is_open()) return;
    file.seekg(0, std::ios::beg);
    uint64_t header_len = 0;
    file.read(reinterpret_cast<char*>(&header_len), 8);
    if (!file.good() || header_len == 0 || header_len > 100 * 1024 * 1024) return;
    std::string header_str(header_len, '\0');
    file.read(&header_str[0], header_len);
    try {
        metadata = nlohmann::json::parse(header_str);
        tensors_data.clear();
        for (auto& [name, info] : metadata.items()) {
            if (name == "__metadata__") continue;
            tensor_metadata tm;
            tm.name = name;
            if (info.contains("shape")) tm.shape = info["shape"].get<std::vector<size_t>>();
            if (info.contains("dtype")) tm.dtype = info["dtype"].get<std::string>();
            if (info.contains("data_offsets")) tm.offsets = info["data_offsets"].get<std::vector<size_t>>();
            if (tm.offsets.size() == 2) {
                tm.byte_size = tm.offsets[1] - tm.offsets[0];
            }
            tensors_data.push_back(tm);
        }
    } catch (...) {}
}

std::string SafeTensors::load_weights(bytes& weight_buffer, std::string weights_name) {
    if (!file.is_open()) _open_file();
    if (!file.is_open()) return "";
    for (const auto& tm : tensors_data) {
        if (tm.name == weights_name && tm.offsets.size() == 2) {
            uint64_t header_len = 0;
            file.seekg(0, std::ios::beg);
            file.read(reinterpret_cast<char*>(&header_len), 8);
            size_t data_start = 8 + header_len + tm.offsets[0];
            size_t bytes_to_read = tm.offsets[1] - tm.offsets[0];
            weight_buffer.resize(bytes_to_read);
            file.seekg(data_start, std::ios::beg);
            file.read(reinterpret_cast<char*>(weight_buffer.data()), bytes_to_read);
            return tm.name;
        }
    }
    return "";
}

tensor_metadata SafeTensors::get_tensor_metadata(std::string tensor_name) {
    for (const auto& tm : tensors_data) {
        if (tm.name == tensor_name) return tm;
    }
    return {};
}

nlohmann::json SafeTensors::get_metadata() {
    return metadata;
}

void SafeTensors::switch_model(std::string new_model_path) {
    if (file.is_open()) file.close();
    model_path = new_model_path;
    _open_file();
    _load_tensors();
}

void SafeTensors::write_safetensors(std::string output_path) {}

Q4NX::Q4NX(std::string model_path) : SafeTensors(model_path) {}
void Q4NX::convert_model(std::string output_path) {}
size_t Q4NX::get_weight_per_chunk() const { return 128; }
size_t Q4NX::get_block_size() const { return 32; }
