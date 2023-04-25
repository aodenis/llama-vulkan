#ifndef VULKAN_LLAMA_CONTEXT_H
#define VULKAN_LLAMA_CONTEXT_H

#include "ggml_file.h"
#include "types.h"
#include <map>
#include "llava_layer.h"
#include "llava_buffer.h"
#include "llava_pipeline.h"

class llava_context : public enable_shared_from_this<llava_context> {
    friend class llava_pipeline;
public:
    llava_context() = default;
    int run(int argc, char** argv);
    llava_pipeline* get_pipeline(const string& shader_name, u32 argcount);

    shared_ptr<vkr::Context> get_context();
    shared_ptr<vkr::Device> get_device();
    shared_ptr<vkr::CommandPool> get_command_pool();
    shared_ptr<vkr::DescriptorPool> get_descriptor_pool();
    shared_ptr<vkr::PipelineCache> get_pipeline_cache();
    shared_ptr<vkr::Queue> get_queue();
    shared_ptr<ggml_file> get_model();
    u32 backlog_size = 256;

private:
    shared_ptr<ggml_file> model;
    vkr::PhysicalDevice get_physical_device();
    uint32_t get_queue_family_index();

    shared_ptr<vkr::Context> vulkan_context;
    shared_ptr<vkr::Instance> vulkan_instance;
    shared_ptr<vkr::PhysicalDevice> physicalDevice;
    shared_ptr<vkr::Device> device;
    shared_ptr<vkr::CommandPool> command_pool;
    shared_ptr<vkr::DescriptorPool> descriptor_pool;
    shared_ptr<vkr::PipelineCache> pipeline_cache;
    shared_ptr<vkr::Queue> queue;

    vector<pair<vkr::Buffer, vkr::DeviceMemory>> model_buffers;
    void prepare_layer();
    vector<llava_layer> layers;

private:
    map<string, llava_buffer> named_buffers;
    map<string, llava_pipeline> named_pipelines;
};

#endif //VULKAN_LLAMA_CONTEXT_H
