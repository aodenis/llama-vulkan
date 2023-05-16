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
    void execute(llava_command_buffer* cmd_buf) const;
    void freeze_storage();
    void freeze_cache_storage();
    [[nodiscard]] bool is_layer_data_offloaded() const;
    void load_to_host();
    void load_to_gpu();
    void set_offload(u32 other_layer);

public:
    [[nodiscard]] bool is_offload_main_layer() const;
    [[nodiscard]] u32 get_offload_id() const;

public:
    u32 const layer_id;
    llava_context* const context;

private:
    llava_device_memory* layer_allocation;
    llava_device_memory* layer_cache_allocation;
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

private:
    u8* raw_layer = nullptr;
    u32 offload_layer_id;
    bool is_offloaded = false;
};

#endif
