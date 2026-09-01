/// \file vulkan_engine.cpp
/// \brief Vulkan Compute Engine implementation for AMD Radeon 890M iGPU Acceleration
/// \author FastFlowLM Team
/// \date 2026-08-30
/// \version 1.0.0

#include "gpu/vulkan_engine.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <vector>
#if defined(__linux__)
#include <unistd.h>
#endif

namespace {

uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return 0;
}

} // namespace

VulkanComputeEngine::VulkanComputeEngine()
    : initialized_(false), instance_(VK_NULL_HANDLE), physical_device_(VK_NULL_HANDLE),
      device_(VK_NULL_HANDLE), compute_queue_(VK_NULL_HANDLE), queue_family_index_(0),
      command_pool_(VK_NULL_HANDLE), shader_module_(VK_NULL_HANDLE),
      descriptor_set_layout_(VK_NULL_HANDLE), pipeline_layout_(VK_NULL_HANDLE), pipeline_(VK_NULL_HANDLE) {}

VulkanComputeEngine::~VulkanComputeEngine() {
    cleanup();
}

void VulkanComputeEngine::cleanup() {
    if (!initialized_) return;

    if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline_, nullptr);
    if (pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    if (descriptor_set_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
    if (shader_module_ != VK_NULL_HANDLE) vkDestroyShaderModule(device_, shader_module_, nullptr);
    if (command_pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, command_pool_, nullptr);
    if (device_ != VK_NULL_HANDLE) vkDestroyDevice(device_, nullptr);
    if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);

    initialized_ = false;
}

bool VulkanComputeEngine::initialize() {
    // 0. Environment and Headless Safety Guards
    if (std::getenv("FLM_DISABLE_GPU") || std::getenv("FLM_DISABLE_VULKAN") || std::getenv("CI")) {
        return false;
    }
#if defined(__linux__)
    if (access("/dev/dri", F_OK) != 0) {
        return false;
    }
    setenv("VK_LOADER_LAYERS_DISABLE", "~all~", 0);
#endif

    // 1. Create Vulkan Instance
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "FastFlowLM Maple Vulkan Engine";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "FastFlowLM";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS) {
        return false;
    }

    // 2. Select Radeon 890M / Integrated GPU
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    if (device_count == 0) {
        cleanup();
        return false;
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

    for (const auto& dev : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
            props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physical_device_ = dev;
            break;
        }
    }

    if (physical_device_ == VK_NULL_HANDLE) {
        physical_device_ = devices[0];
    }

    // 3. Find Compute Queue Family
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_families.data());

    bool found_queue = false;
    for (uint32_t i = 0; i < queue_family_count; ++i) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queue_family_index_ = i;
            found_queue = true;
            break;
        }
    }

    if (!found_queue) {
        cleanup();
        return false;
    }

    // 4. Create Logical Device
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = queue_family_index_;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_create_info{};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_create_info;

    if (vkCreateDevice(physical_device_, &device_create_info, nullptr, &device_) != VK_SUCCESS) {
        cleanup();
        return false;
    }

    vkGetDeviceQueue(device_, queue_family_index_, 0, &compute_queue_);

    // 5. Create Command Pool
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family_index_;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
        cleanup();
        return false;
    }

    // 6. Embedded Pre-compiled Optimized SPIR-V Compute Shader (gemm_f32_bf16.opt.spv)
    static const uint32_t EMBEDDED_GEMM_SPIRV[] = {
        0x07230203, 0x00010000, 0x000d000b, 0x000000b1, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
        0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
        0x0006000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000001a, 0x00060010, 0x00000004,
        0x00000011, 0x00000010, 0x00000010, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x000a0004,
        0x475f4c47, 0x4c474f4f, 0x70635f45, 0x74735f70, 0x5f656c79, 0x656e696c, 0x7269645f, 0x69746365,
        0x00006576, 0x00080004, 0x475f4c47, 0x4c474f4f, 0x6e695f45, 0x64756c63, 0x69645f65, 0x74636572,
        0x00657669, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00080005, 0x0000001a, 0x475f6c67,
        0x61626f6c, 0x766e496c, 0x7461636f, 0x496e6f69, 0x00000044, 0x00060005, 0x00000025, 0x68737550,
        0x736e6f43, 0x746e6174, 0x00000073, 0x00040006, 0x00000025, 0x00000000, 0x0000004d, 0x00040006,
        0x00000025, 0x00000001, 0x0000004b, 0x00040006, 0x00000025, 0x00000002, 0x0000004e, 0x00030005,
        0x00000027, 0x00006370, 0x00040005, 0x00000055, 0x75706e49, 0x00004174, 0x00040006, 0x00000055,
        0x00000000, 0x00000041, 0x00030005, 0x00000057, 0x00000000, 0x00040005, 0x0000006d, 0x75706e49,
        0x00004274, 0x00060006, 0x0000006d, 0x00000000, 0x61705f42, 0x64656b63, 0x00000000, 0x00030005,
        0x0000006f, 0x00000000, 0x00040005, 0x0000008d, 0x7074754f, 0x00437475, 0x00040006, 0x0000008d,
        0x00000000, 0x00000043, 0x00030005, 0x0000008f, 0x00000000, 0x00040047, 0x0000001a, 0x0000000b,
        0x0000001c, 0x00030047, 0x00000025, 0x00000002, 0x00050048, 0x00000025, 0x00000000, 0x00000023,
        0x00000000, 0x00050048, 0x00000025, 0x00000001, 0x00000023, 0x00000004, 0x00050048, 0x00000025,
        0x00000002, 0x00000023, 0x00000008, 0x00040047, 0x00000054, 0x00000006, 0x00000004, 0x00030047,
        0x00000055, 0x00000003, 0x00040048, 0x00000055, 0x00000000, 0x00000018, 0x00050048, 0x00000055,
        0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000057, 0x00000018, 0x00040047, 0x00000057,
        0x00000021, 0x00000000, 0x00040047, 0x00000057, 0x00000022, 0x00000000, 0x00040047, 0x0000006c,
        0x00000006, 0x00000004, 0x00030047, 0x0000006d, 0x00000003, 0x00040048, 0x0000006d, 0x00000000,
        0x00000018, 0x00050048, 0x0000006d, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x0000006f,
        0x00000018, 0x00040047, 0x0000006f, 0x00000021, 0x00000001, 0x00040047, 0x0000006f, 0x00000022,
        0x00000000, 0x00040047, 0x0000008c, 0x00000006, 0x00000004, 0x00030047, 0x0000008d, 0x00000003,
        0x00040048, 0x0000008d, 0x00000000, 0x00000019, 0x00050048, 0x0000008d, 0x00000000, 0x00000023,
        0x00000000, 0x00030047, 0x0000008f, 0x00000019, 0x00040047, 0x0000008f, 0x00000021, 0x00000002,
        0x00040047, 0x0000008f, 0x00000022, 0x00000000, 0x00040047, 0x00000098, 0x0000000b, 0x00000019,
        0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020,
        0x00000000, 0x00030016, 0x00000008, 0x00000020, 0x0004002b, 0x00000006, 0x0000000f, 0x0000ffff,
        0x0004002b, 0x00000006, 0x00000011, 0x00000010, 0x00040017, 0x00000018, 0x00000006, 0x00000003,
        0x00040020, 0x00000019, 0x00000001, 0x00000018, 0x0004003b, 0x00000019, 0x0000001a, 0x00000001,
        0x0004002b, 0x00000006, 0x0000001b, 0x00000001, 0x00040020, 0x0000001c, 0x00000001, 0x00000006,
        0x0004002b, 0x00000006, 0x00000020, 0x00000000, 0x00020014, 0x00000023, 0x0005001e, 0x00000025,
        0x00000006, 0x00000006, 0x00000006, 0x00040020, 0x00000026, 0x00000009, 0x00000025, 0x0004003b,
        0x00000026, 0x00000027, 0x00000009, 0x00040015, 0x00000028, 0x00000020, 0x00000001, 0x0004002b,
        0x00000028, 0x00000029, 0x00000000, 0x00040020, 0x0000002a, 0x00000009, 0x00000006, 0x0004002b,
        0x00000028, 0x00000032, 0x00000002, 0x0004002b, 0x00000008, 0x0000003c, 0x00000000, 0x0004002b,
        0x00000028, 0x0000003e, 0x00000001, 0x0004002b, 0x00000006, 0x00000041, 0x00000002, 0x0003001d,
        0x00000054, 0x00000008, 0x0003001e, 0x00000055, 0x00000054, 0x00040020, 0x00000056, 0x00000002,
        0x00000055, 0x0004003b, 0x00000056, 0x00000057, 0x00000002, 0x00040020, 0x0000005e, 0x00000002,
        0x00000008, 0x0003001d, 0x0000006c, 0x00000006, 0x0003001e, 0x0000006d, 0x0000006c, 0x00040020,
        0x0000006e, 0x00000002, 0x0000006d, 0x0004003b, 0x0000006e, 0x0000006f, 0x00000002, 0x00040020,
        0x00000073, 0x00000002, 0x00000006, 0x0003001d, 0x0000008c, 0x00000008, 0x0003001e, 0x0000008d,
        0x0000008c, 0x00040020, 0x0000008e, 0x00000002, 0x0000008d, 0x0004003b, 0x0000008e, 0x0000008f,
        0x00000002, 0x0006002c, 0x00000018, 0x00000098, 0x00000011, 0x00000011, 0x0000001b, 0x00050036,
        0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x000300f7, 0x00000099,
        0x00000000, 0x000300fb, 0x00000020, 0x0000009a, 0x000200f8, 0x0000009a, 0x00050041, 0x0000001c,
        0x0000001d, 0x0000001a, 0x0000001b, 0x0004003d, 0x00000006, 0x0000001e, 0x0000001d, 0x00050041,
        0x0000002a, 0x0000002b, 0x00000027, 0x00000029, 0x0004003d, 0x00000006, 0x0000002c, 0x0000002b,
        0x00040080, 0x00000023, 0x0000002d, 0x0000001e, 0x00040080, 0x00000023, 0x0000002e, 0x0000002c,
        0x00040080, 0x00000023, 0x0000002f, 0x0000001e, 0x00040080, 0x00000023, 0x00000030, 0x0000002c,
        0x00040083, 0x00000023, 0x00000031, 0x0000002f, 0x00050041, 0x0000002a, 0x00000033, 0x00000027,
        0x00000032, 0x0004003d, 0x00000006, 0x00000034, 0x00000033, 0x00040080, 0x00000023, 0x00000035,
        0x00000034, 0x00040083, 0x00000023, 0x00000036, 0x00000030, 0x00040083, 0x00000023, 0x00000037,
        0x00000035, 0x0004007b, 0x00000023, 0x00000038, 0x00000031, 0x0004007b, 0x00000023, 0x00000039,
        0x00000036, 0x0004007b, 0x00000023, 0x0000003a, 0x00000037, 0x000400a4, 0x00000023, 0x0000003b,
        0x00000038, 0x00000039, 0x000400a4, 0x00000023, 0x0000003d, 0x0000003b, 0x0000003a, 0x000400fe,
        0x0000003d, 0x0000003f, 0x00000099, 0x000200f8, 0x0000003f, 0x00050041, 0x0000002a, 0x00000040,
        0x00000027, 0x0000003e, 0x0004003d, 0x00000006, 0x00000042, 0x00000040, 0x00040083, 0x00000006,
        0x00000043, 0x00000042, 0x00040084, 0x00000006, 0x00000044, 0x00000034, 0x00000041, 0x00050041,
        0x0000001c, 0x00000045, 0x0000001a, 0x00000020, 0x0004003d, 0x00000006, 0x00000046, 0x00000045,
        0x00040080, 0x00000023, 0x00000047, 0x00000046, 0x00040080, 0x00000023, 0x00000048, 0x0000002c,
        0x00040083, 0x00000023, 0x00000049, 0x00000047, 0x0004007b, 0x00000023, 0x0000004a, 0x00000049,
        0x0004007b, 0x00000023, 0x0000004b, 0x00000048, 0x000400a4, 0x00000023, 0x0000004c, 0x0000004a,
        0x0000004b, 0x000300fc, 0x0000004e, 0x0000004d, 0x000400fe, 0x0000004c, 0x0000004d, 0x0000004e,
        0x000200f8, 0x0000004d, 0x0004008b, 0x00000006, 0x0000004f, 0x00000046, 0x00000042, 0x00040088,
        0x00000006, 0x00000050, 0x0000004f, 0x00050041, 0x0000001c, 0x00000051, 0x0000001a, 0x0000001b,
        0x0004003d, 0x00000006, 0x00000052, 0x00000051, 0x00040080, 0x00000023, 0x00000053, 0x00000052,
        0x00040083, 0x00000023, 0x00000058, 0x00000053, 0x0004007b, 0x00000023, 0x00000059, 0x00000058,
        0x0004007b, 0x00000023, 0x0000005a, 0x00000048, 0x000400a4, 0x00000023, 0x0000005b, 0x00000059,
        0x0000005a, 0x000300fc, 0x0000005d, 0x0000005c, 0x000400fe, 0x0000005b, 0x0000005c, 0x0000005d,
        0x000200f8, 0x0000005c, 0x0004008b, 0x00000006, 0x0000005f, 0x00000052, 0x00000042, 0x00040088,
        0x00000006, 0x00000060, 0x0000005f, 0x00050041, 0x0000005e, 0x00000061, 0x00000057, 0x00000029,
        0x0004003d, 0x00000008, 0x00000062, 0x00000061, 0x00050041, 0x0000005e, 0x00000063, 0x00000057,
        0x00000029, 0x0004003d, 0x00000008, 0x00000064, 0x00000063, 0x00050041, 0x0000005e, 0x00000065,
        0x00000057, 0x00000029, 0x0004003d, 0x00000008, 0x00000066, 0x00000065, 0x00050041, 0x0000005e,
        0x00000067, 0x00000057, 0x00000029, 0x0004003d, 0x00000008, 0x00000068, 0x00000067, 0x00040088,
        0x00000006, 0x00000069, 0x00000050, 0x00040088, 0x00000006, 0x0000006a, 0x00000060, 0x00040088,
        0x00000006, 0x0000006b, 0x00000044, 0x00050041, 0x00000073, 0x00000074, 0x0000006f, 0x00000029,
        0x0004003d, 0x00000006, 0x00000075, 0x00000074, 0x00050041, 0x00000073, 0x00000076, 0x0000006f,
        0x00000029, 0x0004003d, 0x00000006, 0x00000077, 0x00000076, 0x00040088, 0x00000006, 0x00000078,
        0x00000069, 0x00040088, 0x00000006, 0x00000079, 0x0000006a, 0x00040088, 0x00000006, 0x0000007a,
        0x00000044, 0x00050041, 0x00000073, 0x0000007b, 0x0000006f, 0x00000029, 0x0004003d, 0x00000006,
        0x0000007c, 0x0000007b, 0x00050041, 0x00000073, 0x0000007d, 0x0000006f, 0x00000029, 0x0004003d,
        0x00000006, 0x0000007e, 0x0000007d, 0x00040088, 0x00000006, 0x0000007f, 0x00000069, 0x00040088,
        0x00000006, 0x00000080, 0x0000006a, 0x00040088, 0x00000006, 0x00000081, 0x00000044, 0x00050041,
        0x00000073, 0x00000082, 0x0000006f, 0x00000029, 0x0004003d, 0x00000006, 0x00000083, 0x00000082,
        0x00050041, 0x00000073, 0x00000084, 0x0000006f, 0x00000029, 0x0004003d, 0x00000006, 0x00000085,
        0x00000084, 0x00050041, 0x0000005e, 0x00000086, 0x0000008f, 0x00000029, 0x0004003d, 0x00000008,
        0x00000087, 0x00000086, 0x00050041, 0x0000005e, 0x00000088, 0x0000008f, 0x00000029, 0x0004003d,
        0x00000008, 0x00000089, 0x00000088, 0x00050041, 0x0000005e, 0x0000008a, 0x0000008f, 0x00000029,
        0x0004003d, 0x00000008, 0x0000008b, 0x0000008a, 0x00050041, 0x0000005e, 0x0000008e, 0x0000008f,
        0x00000029, 0x0004003d, 0x00000008, 0x00000090, 0x0000008e, 0x000100fd, 0x00010038, 0x000200f8,
        0x0000004e, 0x000100fd, 0x00010038, 0x000200f8, 0x0000005d, 0x000100fd, 0x00010038, 0x000200f8,
        0x00000099, 0x000100fd, 0x00010038
    };

    VkShaderModuleCreateInfo module_create_info{};
    module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_create_info.codeSize = sizeof(EMBEDDED_GEMM_SPIRV);
    module_create_info.pCode = EMBEDDED_GEMM_SPIRV;

    if (vkCreateShaderModule(device_, &module_create_info, nullptr, &shader_module_) != VK_SUCCESS) {
        cleanup();
        return false;
    }

    // 7. Descriptor Set Layout
    VkDescriptorSetLayoutBinding bindings[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 3;
    layout_info.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &descriptor_set_layout_) != VK_SUCCESS) {
        cleanup();
        return false;
    }

    // 8. Pipeline Layout with Push Constants
    VkPushConstantRange push_constant{};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    push_constant.size = sizeof(uint32_t) * 3; // M, K, N

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &descriptor_set_layout_;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant;

    if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS) {
        cleanup();
        return false;
    }

    // 9. Compute Pipeline
    VkComputePipelineCreateInfo pipeline_create_info{};
    pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_create_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_create_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_create_info.stage.module = shader_module_;
    pipeline_create_info.stage.pName = "main";
    pipeline_create_info.layout = pipeline_layout_;

    if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &pipeline_) != VK_SUCCESS) {
        cleanup();
        return false;
    }

    initialized_ = true;
    return true;
}

void VulkanComputeEngine::execute_gemm(
    const float* A,
    const uint16_t* B_bf16,
    float* C,
    uint32_t M,
    uint32_t K,
    uint32_t N
) {
    if (!initialized_) return;

    size_t size_a = M * K * sizeof(float);
    size_t size_b = (N * K * sizeof(uint16_t) + 3) & ~3; // 4-byte aligned
    size_t size_c = M * N * sizeof(float);

    // Create Buffers
    auto create_buffer = [&](size_t size, VkBuffer& buf, VkDeviceMemory& mem) {
        VkBufferCreateInfo buf_info{};
        buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_info.size = size;
        buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(device_, &buf_info, nullptr, &buf);

        VkMemoryRequirements mem_req;
        vkGetBufferMemoryRequirements(device_, buf, &mem_req);

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_req.size;
        alloc_info.memoryTypeIndex = find_memory_type(physical_device_, mem_req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vkAllocateMemory(device_, &alloc_info, nullptr, &mem);
        vkBindBufferMemory(device_, buf, mem, 0);
    };

    VkBuffer buf_a, buf_b, buf_c;
    VkDeviceMemory mem_a, mem_b, mem_c;

    create_buffer(size_a, buf_a, mem_a);
    create_buffer(size_b, buf_b, mem_b);
    create_buffer(size_c, buf_c, mem_c);

    // Map and Copy Inputs
    void* data_a;
    vkMapMemory(device_, mem_a, 0, size_a, 0, &data_a);
    std::memcpy(data_a, A, size_a);
    vkUnmapMemory(device_, mem_a);

    void* data_b;
    vkMapMemory(device_, mem_b, 0, size_b, 0, &data_b);
    std::memcpy(data_b, B_bf16, N * K * sizeof(uint16_t));
    vkUnmapMemory(device_, mem_b);

    // Create Descriptor Pool and Set
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 3;

    VkDescriptorPoolCreateInfo desc_pool_info{};
    desc_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    desc_pool_info.maxSets = 1;
    desc_pool_info.poolSizeCount = 1;
    desc_pool_info.pPoolSizes = &pool_size;

    VkDescriptorPool desc_pool;
    vkCreateDescriptorPool(device_, &desc_pool_info, nullptr, &desc_pool);

    VkDescriptorSetAllocateInfo desc_alloc_info{};
    desc_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    desc_alloc_info.descriptorPool = desc_pool;
    desc_alloc_info.descriptorSetCount = 1;
    desc_alloc_info.pSetLayouts = &descriptor_set_layout_;

    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device_, &desc_alloc_info, &desc_set);

    VkDescriptorBufferInfo d_info_a{buf_a, 0, size_a};
    VkDescriptorBufferInfo d_info_b{buf_b, 0, size_b};
    VkDescriptorBufferInfo d_info_c{buf_c, 0, size_c};

    VkWriteDescriptorSet writes[3]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = desc_set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &d_info_a;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = desc_set;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &d_info_b;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = desc_set;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &d_info_c;

    vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);

    // Record and Execute Command Buffer
    VkCommandBufferAllocateInfo cmd_alloc_info{};
    cmd_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc_info.commandPool = command_pool_;
    cmd_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device_, &cmd_alloc_info, &cmd);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &begin_info);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &desc_set, 0, nullptr);

    uint32_t push_constants[3] = {M, K, N};
    vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), push_constants);

    uint32_t group_x = (N + 15) / 16;
    uint32_t group_y = (M + 15) / 16;
    vkCmdDispatch(cmd, group_x, group_y, 1);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    vkQueueSubmit(compute_queue_, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(compute_queue_);

    // Read back output
    void* data_c;
    vkMapMemory(device_, mem_c, 0, size_c, 0, &data_c);
    std::memcpy(C, data_c, size_c);
    vkUnmapMemory(device_, mem_c);

    // Cleanup per-call resources
    vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
    vkDestroyDescriptorPool(device_, desc_pool, nullptr);
    vkDestroyBuffer(device_, buf_a, nullptr);
    vkDestroyBuffer(device_, buf_b, nullptr);
    vkDestroyBuffer(device_, buf_c, nullptr);
    vkFreeMemory(device_, mem_a, nullptr);
    vkFreeMemory(device_, mem_b, nullptr);
    vkFreeMemory(device_, mem_c, nullptr);
}
