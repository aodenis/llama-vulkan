#ifndef VULKAN_LLAMA_LLAVA_COMMAND_BUFFER_H
#define VULKAN_LLAMA_LLAVA_COMMAND_BUFFER_H

#include "types.h"
#include <list>
#include <vulkan/vulkan.hpp>
#include <map>

class llava_wrapped_command {
public:
    llava_wrapped_command(llava_context* context, vk::DescriptorSet descriptorSet, vk::CommandBuffer commandBuffer, vk::Event completionEvent);
    llava_wrapped_command(llava_wrapped_command const&) = delete;
    llava_wrapped_command(llava_wrapped_command&) = delete;
    llava_wrapped_command(llava_wrapped_command&&) = delete;
    ~llava_wrapped_command();
    llava_context* const context;
    const vk::DescriptorSet descriptorSet;
    const vk::CommandBuffer commandBuffer;
    const vk::Event completionEvent;
};

class llava_command_buffer {
public:

public:
    explicit llava_command_buffer(llava_context *context);
    ~llava_command_buffer();
    void record_execution();
    void reset_events();
    void run();
    void normalize_logit(llava_buffer* outbuf, llava_buffer* inbuf, llava_buffer* weights);
    void matmul(llava_buffer* outbuf, llava_buffer*, llava_buffer*);
    void matmul_add_inplace(llava_buffer* outbuf, llava_buffer*, llava_buffer*);
    void kv_copy(llava_buffer*, llava_buffer*);
    void multi_head_attention(llava_buffer* attn_out, llava_buffer* k_cache, llava_buffer* query);
    void perform_kqv_matching(llava_buffer* v_out, llava_buffer* v_cache, llava_buffer* softmax_out);
    void inplace_softmax(llava_buffer*);
    void rope(llava_buffer* buf);
    void matmul_silu_ff(llava_buffer *outbuf, llava_buffer *w3_matrix, llava_buffer *w1_matrix, llava_buffer *inbuf);

public:
    void record_command(const string& pipeline_name, const initializer_list<llava_buffer *> &buffers, uint32_t countX, uint32_t countY = 1, uint32_t countZ = 1);

public:
    llava_context* const context;
    u32 const backlog_size;
    u32 const workgroup_size;

private: // command buffer stuff
    list<llava_wrapped_command> command_buffer;
    vector<vk::CommandBuffer> command_buffer_raw;

private:
    map<llava_buffer*, vk::Event> buffer_to_last_write_event;
    map<llava_buffer*, vk::Event> layer_buffer_to_layer_write_event;
};

#endif
