#ifndef VULKAN_LLAMA_CONTEXT_H
#define VULKAN_LLAMA_CONTEXT_H

#include <vulkan/vulkan_raii.hpp>
#include <memory>
#include "ggml_file.h"

namespace vkr = vk::raii;

class llava_context {
public:
    llava_context() = default;
    int run(int argc, char** argv);

private:
    std::unique_ptr<ggml_file> model;
    vkr::PhysicalDevice get_physical_device();
    uint32_t get_queue_family_index();

    std::unique_ptr<vkr::Context> vulkan_context;
    std::unique_ptr<vkr::Instance> vulkan_instance;
    std::unique_ptr<vkr::PhysicalDevice> physicalDevice;
    std::unique_ptr<vkr::Device> device;
    std::unique_ptr<vkr::CommandPool> commandPool;
    std::unique_ptr<vkr::DescriptorPool> descriptorPool;


};

#endif //VULKAN_LLAMA_CONTEXT_H
