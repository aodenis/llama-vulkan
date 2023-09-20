#ifndef VULKAN_LLAMA_CONTEXT_H
#define VULKAN_LLAMA_CONTEXT_H

#include "ggml_file.h"
#include "types.h"
#include <memory>
#include <set>
#include <map>
#include <list>
#include <random>
#include "llava_layer.h"
#include "llava_buffer.h"
#include "llava_pipeline.h"

class llava_context {
    friend class llava_command_buffer;
    friend class llava_pipeline;
    friend class llava_session;
    friend class llava_layer;
public:
    llava_context();
    ~llava_context();
    int run(int argc, char** argv);
    [[nodiscard]] bool signal_debug_on() const;

public:
    vk::Device& get_device();
    vk::CommandPool& get_command_pool();
    vk::Queue& get_queue();
    vk::DescriptorPool& get_descriptor_pool();
    vk::PipelineCache& get_pipeline_cache();
    vk::PhysicalDevice& get_physical_device();
    [[nodiscard]] uint32_t get_queue_family_index() const;
    [[nodiscard]] ggml_file const* get_model() const;
    [[nodiscard]] vector<llava_layer>& get_layers();
    [[nodiscard]] static string generate_spevar_define_string(specialization_variables_t const* spevars) ;
    [[nodiscard]] pair<u32*, u32> get_shader_spirv_by_name(string const& shader_name);

public:
    llava_pipeline* get_pipeline(const string& shader_name, u32 argument_count, specialization_variables_t const& spevars);
    [[nodiscard]] int get_signal_fd() const;
    [[nodiscard]] u32 pop_signal(bool blocking = false) const;

public:
    u32 workgroup_size = 1024;
    u32 mainMemoryTypeIndex = ~0U;
    mutex descriptor_pool_mutex;
    mutex command_pool_mutex;
    mutex queue_mutex;

private:
    ggml_file* model = nullptr;

    vk::Instance vulkan_instance;
    vk::PhysicalDevice physical_device;
    vk::Device device;
    vk::CommandPool command_pool;
    vk::DescriptorPool descriptor_pool;
    vk::PipelineCache pipeline_cache;
    vk::Queue queue;

    u32 queueFamilyIndex = ~0U;
    u32 verbosity = 0;
    bool signal_debug = false;

    vector<llava_layer> layers;

private: // config
    bool use_prebuilt_shaders = false;

private:
    int sigfd = -1;

private: // pipeline storage
    map<pair<string, specialization_variables_t>, llava_pipeline> named_pipelines;
    map<string, pair<u32*, u32>> embedded_shaders;
    mutex pipeline_mutex;

private:
    u32 find_suitable_memory_type(const vk::PhysicalDevice &_physical_device);
    u32 find_suitable_queue_index();
    vk::PhysicalDevice find_suitable_physical_device();
    bool setup_signal_handling();
};

#endif //VULKAN_LLAMA_CONTEXT_H
