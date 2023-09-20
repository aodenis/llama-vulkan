#ifndef VULKAN_LLAMA_LLAVA_SESSION_H
#define VULKAN_LLAMA_LLAVA_SESSION_H

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
    u32 batch_enabled; // = 1;
};

class llava_session {
    friend class llava_layer;
    friend class llava_layer_session_data;
    friend class llava_command_buffer;
public:
    explicit llava_session(llava_context* ctx);
    ~llava_session();
    ND vector<llava_layer_session_data*> const& get_layer_data() const;
    ND bool set_text(const string &new_text);
    ND u32 get_token_count() const;
    ND vector<u32> const& get_token_buffer() const;
    ND bool push_token(u32);
    ND u32 predict_next_token();
    ND bool start_next_token_prediction();
    ND u32 finish_next_token_prediction();
    ND specialization_variables_t const& get_spevar_struct() const;
    ND bool is_tracing_enabled() const;

    void rewind(u32);
    ReturnCode set_options(u32);
    ReturnCode save_frame(const string& path);
    ReturnCode snapshot(const string &path);
    ReturnCode restore(const string &path);
    ND bool add_text(const string& s);

public:
    llava_context* const ctx;
    ggml_file const* const model;

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

private: // buffers
    llava_device_memory* main_buffer_memory = nullptr;
    llava_buffer* current_thought = nullptr;
    llava_buffer* current_thought_sublayer = nullptr;
    llava_buffer* current_thought_middle_normd = nullptr;
    llava_buffer* current_Q = nullptr;
    llava_buffer* current_K = nullptr;
    llava_buffer* current_V = nullptr;
    llava_buffer* current_Vout = nullptr;
    llava_buffer* main_attn_result = nullptr;
    llava_buffer* config_buffer = nullptr;
    llava_buffer* norm_w = nullptr;
    llava_buffer* output_w = nullptr;
    llava_buffer* output_probs = nullptr;
    llava_buffer* properties_mask = nullptr;
    llava_buffer* main_ff_result = nullptr;
    vector<llava_layer_session_data*> layer_data;

private:
    u32 batch_size = 0;
    u32 backlog_size;
    [[nodiscard]] u32 get_last_predicted_token(bool deterministic);

private:
    void reset_main_buffers();
    void recreate_buffers();
    void set_batch_size(u32 batch_size);
    [[nodiscard]] bool set_backlog_size(u32 new_size);

private:
    llava_command_buffer* command_buffer = nullptr;
    specialization_variables_t specialization_variables{};

private: // Token buffer management
    vector<u32> token_buffer;
    u32 current_tokens_in_gpu = 0;
    void recreate_spevars();
    void ensure_buffers_created();

private:
    u32 options = 0; // For now, non-null = record
    void flush_layers_data_buffers();
};

#endif //VULKAN_LLAMA_LLAVA_SESSION_H
