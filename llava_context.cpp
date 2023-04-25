#include "llava_context.h"
#include "llava_pipeline.h"
#include "llava_buffer.h"
#include "llava_layer.h"
#include <iostream>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include "ggml_file.h"

vkr::PhysicalDevice llava_context::get_physical_device() {
    for(vkr::PhysicalDevice const &pd : vulkan_instance->enumeratePhysicalDevices()) {
        if (pd.getProperties().deviceType == vk::PhysicalDeviceType::eIntegratedGpu)
            return pd;
    }
    assert(false);
}

uint32_t llava_context::get_queue_family_index() {
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice->getQueueFamilyProperties();
    // Compute queue family index
    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
        if (queueFamilyProperties.at(i).queueFlags & vk::QueueFlagBits::eCompute) {
            return i;
        }
    }
    assert(false);
}

int llava_context::run(int argc, char **argv) try {
    model = std::make_shared<ggml_file>("/home/denis/Documents/CProj/llm-conv/ggml-model-q4_0-7B.bin");
    model->print_info();

    // initialize the vkr::ApplicationInfo structure
    vk::ApplicationInfo applicationInfo("llm", 1, "llm0", 1, VK_API_VERSION_1_3);

    vector<const char*> enabled_layers = {"VK_LAYER_KHRONOS_validation"};
    // create an Instance
    vulkan_context = make_shared<vkr::Context>();
    vulkan_instance = make_shared<vkr::Instance>(std::move(vkr::Instance(*vulkan_context, {{}, &applicationInfo, static_cast<uint32_t>(enabled_layers.size()), enabled_layers.data(),
                                                0, nullptr})));

    physicalDevice = make_shared<vkr::PhysicalDevice>(get_physical_device());
    uint32_t queueFamilyIndex = get_queue_family_index();

    // create a Device
    float queuePriority = 0.0f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), queueFamilyIndex, 1, &queuePriority);
    device = make_shared<vkr::Device>(std::move(vkr::Device(*physicalDevice, {vk::DeviceCreateFlags(), deviceQueueCreateInfo})));

    // create a CommandPool to allocate a CommandBuffer from
    command_pool = make_shared<vkr::CommandPool>(std::move(vkr::CommandPool(*device, {{}, queueFamilyIndex})));

    // Descriptor pool
    vk::DescriptorPoolSize descriptorPoolSize(vk::DescriptorType::eStorageBuffer, 8);
    descriptor_pool = make_shared<vkr::DescriptorPool>(std::move(vkr::DescriptorPool(*device, vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                                                                                                                           1024, 1, &descriptorPoolSize))));
    // Queue
    queue = make_shared<vkr::Queue>(*device, queueFamilyIndex, 0);

    // Pipeline cache
    pipeline_cache = make_shared<vkr::PipelineCache>(*device, vk::PipelineCacheCreateInfo({}, 0, nullptr));

    llava_pipeline pipeline(shared_from_this(), "built_shaders/q4_0_split.spirv", 3);

    named_buffers.emplace(std::piecewise_construct, forward_as_tuple("current_thought"), forward_as_tuple(this, 4 * model->header.dim, 1024 * 4));
    named_buffers.emplace(std::piecewise_construct, forward_as_tuple("current_Q"), forward_as_tuple(this, 4 * model->header.dim, 1024 * 4));
    named_buffers.emplace(std::piecewise_construct, forward_as_tuple("current_K"), forward_as_tuple(this, 4 * model->header.dim, 1024 * 4));
    named_buffers.emplace(std::piecewise_construct, forward_as_tuple("current_V"), forward_as_tuple(this, 4 * model->header.dim, 1024 * 4));

    for(u32 i = 0; i < model->header.n_layers; ++i) {
        prepare_layer();
    }

    return 0;
} catch (vk::SystemError &err) {
    cerr << "vkr::SystemError: " << err.what() << endl;
    return 1;
} catch (exception &err) {
    cerr << "std::exception: " << err.what() << endl;
    return 1;
}

shared_ptr<vkr::Context> llava_context::get_context() {
    assert(vulkan_context != nullptr);
    return vulkan_context;
}

shared_ptr<vkr::Device> llava_context::get_device() {
    assert(device != nullptr);
    return device;
}

shared_ptr<vkr::CommandPool> llava_context::get_command_pool() {
    assert(command_pool != nullptr);
    return command_pool;
}

shared_ptr<vkr::DescriptorPool> llava_context::get_descriptor_pool() {
    assert(descriptor_pool != nullptr);
    return descriptor_pool;
}

shared_ptr<vkr::PipelineCache> llava_context::get_pipeline_cache() {
    assert(pipeline_cache != nullptr);
    return pipeline_cache;
}

shared_ptr<vkr::Queue> llava_context::get_queue() {
    assert(queue != nullptr);
    return queue;
}

shared_ptr<ggml_file> llava_context::get_model() {
    assert(model != nullptr);
    return model;
}

void llava_context::prepare_layer() {
    u32 layer_id = layers.size();
    assert(layer_id < model->header.n_layers);
    layers.emplace_back(this, layer_id);
}

llava_pipeline *llava_context::get_pipeline(const string &shader_name, u32 argcount) {
    auto it = named_pipelines.find(shader_name);
    if (it != named_pipelines.end()) {
        assert ((argcount == 0) or (it->second.argcount == argcount));
        return &it->second;
    }
    string source_path = string("built_shaders/") + shader_name + ".spirv";
    it = named_pipelines.emplace(std::piecewise_construct, forward_as_tuple(shader_name), forward_as_tuple(shared_from_this(), source_path.c_str(), argcount)).first;
    return &it->second;
}
