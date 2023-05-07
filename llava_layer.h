#ifndef VULKAN_LLAMA_LLAVA_LAYER_H
#define VULKAN_LLAMA_LLAVA_LAYER_H

#include "types.h"

class llava_layer {
    friend class llava_context;
public:
    llava_layer(llava_context* context, u32 layer_id);
    ~llava_layer();
    vk::Event execute(llava_context* ctx, vk::Event last_event);

private:
    llava_buffer* attention_wq;
    llava_buffer* attention_wk;
    llava_buffer* attention_wv;
    llava_buffer* attention_wo;
    llava_buffer* feed_forward_w1;
    llava_buffer* feed_forward_w2;
    llava_buffer* feed_forward_w3;
    llava_buffer* attention_norm;
    llava_buffer* ffn_norm;
    llava_buffer* k_cache;
    llava_buffer* v_cache;
};

#endif //VULKAN_LLAMA_LLAVA_LAYER_H
