#ifndef VULKAN_LLAMA_CONTEXT_H
#define VULKAN_LLAMA_CONTEXT_H

#include <vulkan/vulkan_raii.hpp>
#include <memory>
#include "ggml_file.h"

using namespace std;
namespace vkr = vk::raii;

class llava_context {
public:
    llava_context() = default;
    int run(int argc, char** argv);

private:
    unique_ptr<ggml_file> model;
    vkr::PhysicalDevice get_physical_device();
    uint32_t get_queue_family_index();

    unique_ptr<vkr::Context> vulkan_context;
    unique_ptr<vkr::Instance> vulkan_instance;
    unique_ptr<vkr::PhysicalDevice> physicalDevice;
    unique_ptr<vkr::Device> device;
    unique_ptr<vkr::CommandPool> commandPool;
    unique_ptr<vkr::DescriptorPool> descriptorPool;

    vector<pair<vkr::Buffer, vkr::DeviceMemory>> model_buffers;

private:
    void build_buffers();

};

#endif //VULKAN_LLAMA_CONTEXT_H
