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
    friend class llava_command_buffer;
    friend class llava_pipeline;
    friend class llava_layer;
public:
    llava_context();
    ~llava_context();
    int run(int argc, char** argv);

public:
    vk::Device& get_device();
    vk::CommandPool& get_command_pool();
    vk::Queue& get_queue();
    vk::DescriptorPool& get_descriptor_pool();
    vk::PipelineCache& get_pipeline_cache();
    vk::PhysicalDevice& get_physical_device();
    [[nodiscard]] uint32_t get_queue_family_index() const;
    [[nodiscard]] shared_ptr<ggml_file> get_model() const;
    [[nodiscard]] specialization_variables_t const& get_spevar_struct() const;
    [[nodiscard]] vector<llava_layer>& get_layers();

public:
    llava_pipeline* get_pipeline(const string& shader_name, u32 argument_count);

public:
    u32 backlog_size = 512;
    u32 batch_size = 0;
    u32 workgroup_size = 1024;
    specialization_variables_t specialization_variables{};
    u32 mainMemoryTypeIndex = ~0U;

public: // various methods
    [[nodiscard]] string generate_spevar_define_string() const;

private:
    shared_ptr<ggml_file> model;

    vk::Instance vulkan_instance;
    vk::PhysicalDevice physical_device;
    vk::Device device;
    vk::CommandPool command_pool;
    vk::DescriptorPool descriptor_pool;
    vk::PipelineCache pipeline_cache;
    vk::Queue queue;

    u32 queueFamilyIndex = ~0U;
    u32 verbosity = 0;

    vector<llava_layer> layers;
    void process_tokens(vector<u32> const& token_ids);
    vector<u32> tokens;
    [[nodiscard]] u32 get_last_predicted_token();
    llava_command_buffer* command_buffer = nullptr;

private: // buffers
    llava_device_memory* main_buffer_memory = nullptr;
    llava_buffer* current_thought = nullptr;
    llava_buffer* current_thought_sublayer = nullptr;
    llava_buffer* current_thought_middle_normd = nullptr;
    llava_buffer* current_Q = nullptr;
    llava_buffer* current_K = nullptr;
    llava_buffer* current_V = nullptr;
    llava_buffer* current_Vout = nullptr;
    llava_buffer* attn_result = nullptr;
    llava_buffer* config_buffer = nullptr;
    llava_buffer* norm_w = nullptr;
    llava_buffer* output_w = nullptr;
    llava_buffer* output_probs = nullptr;
    llava_buffer* properties_mask = nullptr;
    llava_buffer* properties_associated_values = nullptr;

private: // config
    bool use_prebuilt_shaders = false;

private: // storage
    map<string, llava_pipeline> named_pipelines;

private:
    u32 find_suitable_memory_type(const vk::PhysicalDevice &_physical_device);
    u32 find_suitable_queue_index();
    vk::PhysicalDevice find_suitable_physical_device();

private:
    void reset_main_buffers();

private: // command buffer management
    void set_batch_size(u32 _batch_size);
    void recreate_buffers();

private:
    u32 system_prompt_size = 0;
    vector<u32> session;
    void process_system_prompt(const string& system_prompt);
    void process_text(const string& system_prompt);
    void process_text_sync(const string& system_prompt);
    void process_token_rotating(u32 token);
    static string get_user_input();

private:
    std::mt19937 rng;
    const u32 repeat_last_n = 64;
    const float repeat_penalty = 1.1;
    const float alpha_frequency = 0.1;
    const float alpha_presence = 0.0;
    const float temp = 0.8;
    const float mirostat_tau = 5.0;
    const float mirostat_eta = 0.1;
    float mirostat_mu;
};

#endif //VULKAN_LLAMA_CONTEXT_H
