/// \file vulkan_engine.hpp
/// \brief High-Performance Vulkan Compute Engine for Radeon 890M iGPU Acceleration
/// \author FastFlowLM Team
/// \date 2026-08-30
/// \version 1.0.0

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <vulkan/vulkan.h>

class VulkanComputeEngine {
public:
    VulkanComputeEngine();
    ~VulkanComputeEngine();

    bool initialize();
    bool is_available() const { return initialized_; }

    void execute_gemm(
        const float* A,
        const uint16_t* B_bf16,
        float* C,
        uint32_t M,
        uint32_t K,
        uint32_t N
    );

private:
    bool initialized_;
    VkInstance instance_;
    VkPhysicalDevice physical_device_;
    VkDevice device_;
    VkQueue compute_queue_;
    uint32_t queue_family_index_;
    VkCommandPool command_pool_;
    VkShaderModule shader_module_;
    VkDescriptorSetLayout descriptor_set_layout_;
    VkPipelineLayout pipeline_layout_;
    VkPipeline pipeline_;

    void cleanup();
};
