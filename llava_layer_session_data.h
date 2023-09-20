#ifndef VULKAN_LLAMA_LLAVA_LAYER_SESSION_DATA_H
#define VULKAN_LLAMA_LLAVA_LAYER_SESSION_DATA_H

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

class llava_session;

class llava_layer_session_data {
    friend class llava_layer;
public:
    explicit llava_layer_session_data(llava_session* session);
    llava_layer_session_data(llava_layer_session_data const&) = delete;
    llava_layer_session_data(llava_layer_session_data&) = delete;
    llava_layer_session_data(llava_layer_session_data&&) noexcept;
    ~llava_layer_session_data();
    void dump_tracing_layers(int out_fd, const string& name);
    void flush_buffers_on_gpu();
    void dump_kv_cache(u8 *dst, u32 token_count);
    void restore_kv_cache(const u8 *src, u32 token_count);

public:
    llava_session* const session;

private:
    llava_device_memory* layer_cache_allocation = nullptr;
    llava_buffer* k_cache = nullptr;
    llava_buffer* v_cache = nullptr;

private: // Record buffers
    llava_buffer* attn_result = nullptr;
    llava_buffer* ff_result = nullptr;
    llava_buffer* normalized_input_logit = nullptr;
    llava_buffer* post_attn_logit = nullptr;
    llava_buffer* output_logit = nullptr;
    llava_buffer* post_attn_norm_logit = nullptr;
};

#endif
