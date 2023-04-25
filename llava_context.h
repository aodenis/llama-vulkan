#ifndef VULKAN_LLAMA_CONTEXT_H
#define VULKAN_LLAMA_CONTEXT_H

#include "ggml_file.h"
#include "types.h"

class llava_context : public enable_shared_from_this<llava_context> {
    friend class llava_pipeline;
public:
    llava_context() = default;
    int run(int argc, char** argv);
    shared_ptr<vkr::Context> get_context();
    shared_ptr<vkr::Device> get_device();
    shared_ptr<vkr::CommandPool> get_command_pool();
    shared_ptr<vkr::DescriptorPool> get_descriptorPool();
    shared_ptr<vkr::PipelineCache> get_pipeline_cache();
    shared_ptr<vkr::Queue> get_queue();

private:
    shared_ptr<ggml_file> model;
    vkr::PhysicalDevice get_physical_device();
    uint32_t get_queue_family_index();

    shared_ptr<vkr::Context> vulkan_context;
    shared_ptr<vkr::Instance> vulkan_instance;
    shared_ptr<vkr::PhysicalDevice> physicalDevice;
    shared_ptr<vkr::Device> device;
    shared_ptr<vkr::CommandPool> commandPool;
    shared_ptr<vkr::DescriptorPool> descriptorPool;
    shared_ptr<vkr::PipelineCache> pipelineCache;
    shared_ptr<vkr::Queue> queue;

    vector<pair<vkr::Buffer, vkr::DeviceMemory>> model_buffers;

private:
    void build_base_buffers(llava_pipeline* q4_0_split_pipeline);

};

#endif //VULKAN_LLAMA_CONTEXT_H
