#ifndef VULKAN_LLAMA_LLAVA_COMMAND_BUFFER_H
#define VULKAN_LLAMA_LLAVA_COMMAND_BUFFER_H

#include "types.h"
#include <list>
#include <vulkan/vulkan.hpp>

class llava_command_buffer {
public:
    explicit llava_command_buffer(llava_context *context);
    ~llava_command_buffer();
    void record_execution();
    void reset_events();
    void run();
    vk::Event normalize_logit(llava_buffer* outbuf, llava_buffer* inbuf, llava_buffer* weights, initializer_list<vk::Event> events);
    vk::Event matmul(llava_buffer* outbuf, llava_buffer*, llava_buffer*, initializer_list<vk::Event> events);
    vk::Event matmul_add_inplace(llava_buffer* outbuf, llava_buffer*, llava_buffer*, initializer_list<vk::Event> events);
    vk::Event kv_copy(llava_buffer*, llava_buffer*, initializer_list<vk::Event> events);
    vk::Event multi_head_attention(llava_buffer* attn_out, llava_buffer* k_cache, llava_buffer* query, initializer_list<vk::Event> events);
    vk::Event perform_kqv_matching(llava_buffer* v_out, llava_buffer* v_cache, llava_buffer* softmax_out, initializer_list<vk::Event> events);
    vk::Event inplace_softmax(llava_buffer*, initializer_list<vk::Event> events);
    vk::Event rope(llava_buffer* buf, initializer_list<vk::Event> events);
    vk::Event matmul_silu_ff(llava_buffer *outbuf, llava_buffer *w3_matrix, llava_buffer *w1_matrix, llava_buffer *inbuf, initializer_list<vk::Event> events);
    vk::Event record_command(llava_pipeline *pipeline, const initializer_list<llava_buffer *> &buffers, const initializer_list<vk::Event> &events, uint32_t countX, uint32_t countY = 1, uint32_t countZ = 1);
    vk::Event record_command(const string& pipeline_name, const initializer_list<llava_buffer *> &buffers, const initializer_list<vk::Event> &events, uint32_t countX, uint32_t countY = 1, uint32_t countZ = 1);

public:
    llava_context* const context;
    u32 const backlog_size;
    u32 const workgroup_size;

private: // command buffer stuff
    list<llava_command> command_buffer;
    vector<vk::CommandBuffer> command_buffer_raw;
};


#endif //VULKAN_LLAMA_LLAVA_COMMAND_BUFFER_H
