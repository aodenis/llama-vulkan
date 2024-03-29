#ifndef VULKAN_LLAMA_LLAVA_LAYER_H
#define VULKAN_LLAMA_LLAVA_LAYER_H

#include "types.h"

class llava_layer {
public:
    llava_layer(llava_context* context, u32 layer_id);
    llava_layer(llava_layer const&) = delete;
    llava_layer(llava_layer&) = delete;
    llava_layer(llava_layer&&) noexcept;
    ~llava_layer();
    llava_buffer* execute(llava_command_buffer *cmd_buf, llava_layer_session_data* layer_data, llava_buffer* raw_input_logit) const;
    void freeze_storage();
    void load_to_gpu();

public:
    u32 const layer_id;
    llava_context* const context;

private:
    llava_device_memory* layer_allocation;
    llava_buffer* attention_wq;
    llava_buffer* attention_wk;
    llava_buffer* attention_wv;
    llava_buffer* attention_wo;
    llava_buffer* feed_forward_w1;
    llava_buffer* feed_forward_w2;
    llava_buffer* feed_forward_w3;
    llava_buffer* attention_norm;
    llava_buffer* ffn_norm;

private:
    u8* raw_layer = nullptr;
};

#endif
