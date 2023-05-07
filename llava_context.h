#ifndef VULKAN_LLAMA_CONTEXT_H
#define VULKAN_LLAMA_CONTEXT_H

#include "ggml_file.h"
#include "types.h"
#include <memory>
#include <map>
#include <list>
#include "llava_command.h"
#include "llava_layer.h"
#include "llava_buffer.h"
#include "llava_pipeline.h"

struct specialization_variables_t {
    u32 head_count; // = 32;
    u32 quarterrot; // = 32;
    u32 backlog; // = 128;
    u32 max_wgs; // = 1024;
    u32 max_wgs_bits; // = 10;
    u32 ff_size; // = 11008;
    u32 softmax_head_per_wavefront; // = 8;
    u32 backlog_bits; // = 7;
    u32 rot_bits; // = 7;
    u32 rot; // = 128;
    u32 matmul_dim_row_per_wavefront; // = 8;
    u32 matmul_dim_row_worker_count; // = 128;
    u32 matmul_dim_row_worker_count_log2; // = 7;
    u32 matmul_dim_q4_block_count_per_worker; // = 1;
    u32 matmul_dim_q4_blocks_per_row; // = 128;
    u32 matmul_ff_row_per_wavefront; // = 8;
    u32 matmul_ff_row_worker_count; // = 128;
    u32 matmul_ff_row_worker_count_log2; // = 7;
    u32 matmul_ff_q4_block_count_per_worker; // = 3;
    u32 matmul_ff_q4_blocks_per_row; // = 344;
};

class llava_context {
    friend class llava_command;
    friend class llava_pipeline;
    friend class llava_layer;
public:
    llava_context() = default;
    ~llava_context();
    int run(int argc, char** argv);
    llava_pipeline* get_pipeline(const string& shader_name, u32 argcount);

    vk::Device& get_device();
    vk::CommandPool& get_command_pool();
    vk::DescriptorPool& get_descriptor_pool();
    vk::PipelineCache& get_pipeline_cache();
    shared_ptr<ggml_file> get_model();
    u32 backlog_size = 128;
    u32 workgroup_size = 1024;
    bool allocate_buffers = true;
    specialization_variables_t specialization_variables{};
    u32 mainMemoryTypeIndex = 3;
    u32 backupMemoryTypeIndex = 3;

public: // various methods
    vk::Event normalize_logit(llava_buffer* outbuf, llava_buffer* inbuf, llava_buffer* weights, initializer_list<vk::Event> events);
    vk::Event row_wise_multiply(llava_buffer* buf, llava_buffer* weights, initializer_list<vk::Event> events);
    vk::Event matmul(llava_buffer* outbuf, llava_buffer*, llava_buffer*, initializer_list<vk::Event> events);
    vk::Event kv_copy(llava_buffer*, llava_buffer*, initializer_list<vk::Event> events);
    vk::Event multi_head_attention(llava_buffer* attn_out, llava_buffer* k_cache, llava_buffer* query, initializer_list<vk::Event> events);
    vk::Event perform_kqv_matching(llava_buffer* v_out, llava_buffer* v_cache, llava_buffer* softmax_out, initializer_list<vk::Event> events);
    vk::Event inplace_softmax(llava_buffer*, initializer_list<vk::Event> events);
    vk::Event add(llava_buffer* outbuf, llava_buffer* delta_buf, initializer_list<vk::Event> events);
    vk::Event silu(llava_buffer* buf, initializer_list<vk::Event> events);
    vk::Event rope(llava_buffer* buf, initializer_list<vk::Event> events);
    vk::Event record_command(llava_pipeline *pipeline, const initializer_list<llava_buffer *> &buffers, const initializer_list<vk::Event> &events, uint32_t countX, uint32_t countY = 1, uint32_t countZ = 1);
    vk::Event record_command(const string& pipeline_name, const initializer_list<llava_buffer *> &buffers, const initializer_list<vk::Event> &events, uint32_t countX, uint32_t countY = 1, uint32_t countZ = 1);
    vk::Event record_execution(vk::Event startEvent);
    [[nodiscard]] string generate_spevar_define_string() const;

private:
    shared_ptr<ggml_file> model;
    vk::PhysicalDevice get_physical_device();
    [[nodiscard]] uint32_t get_queue_family_index() const;

    vk::Instance vulkan_instance;
    vk::PhysicalDevice physical_device;
    vk::Device device;
    vk::CommandPool command_pool;
    vk::DescriptorPool descriptor_pool;
    vk::PipelineCache pipeline_cache;
    vk::Queue queue;

    u32 queueFamilyIndex = ~0U;
    u32 verbosity = 0;

    list<llava_layer> layers;
    void process_token(u32 new_token);
    vector<u32> tokens;
    [[nodiscard]] u32 get_last_predicted_token() const;
    list<llava_command> command_buffer;
    vector<vk::CommandBuffer> command_buffer_raw;

private: // buffers
    llava_buffer* current_thought = nullptr;
    llava_buffer* current_thought_sublayer = nullptr;
    llava_buffer* current_Q = nullptr;
    llava_buffer* current_K = nullptr;
    llava_buffer* current_V = nullptr;
    llava_buffer* attn_result = nullptr;
    llava_buffer* config_buffer = nullptr;
    llava_buffer* norm_w = nullptr;
    llava_buffer* output_w = nullptr;
    llava_buffer* output_probs = nullptr;
    llava_buffer* properties_mask = nullptr;
    llava_buffer* properties_associated_values = nullptr;

private:
    bool use_prebuilt_shaders = false;
    u32 vulkan_target_version = 12;

private:
    map<string, llava_pipeline> named_pipelines;
    void reset_command_buffer_events();
};

#endif //VULKAN_LLAMA_CONTEXT_H
