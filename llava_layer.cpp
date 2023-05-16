#include <iostream>
#include "llava_buffer.h"
#include "llava_layer.h"
#include "llava_device_memory.h"
#include "llava_command_buffer.h"
#include "llava_context.h"

llava_layer::llava_layer(llava_context *_context, u32 _layer_id) : layer_id(_layer_id),
                                                                   context(_context),
                                                                   offload_layer_id(layer_id) {
    layer_allocation = new llava_device_memory(context);
    layer_cache_allocation = new llava_device_memory(context);
    string prefix = "layers." + to_string(layer_id) + ".";
    attention_wq = new llava_buffer(context, context->get_model()->get_buffer_descriptor(prefix + "attention.wq"), layer_allocation);
    attention_wk = new llava_buffer(context, context->get_model()->get_buffer_descriptor(prefix + "attention.wk"), layer_allocation);
    attention_wv = new llava_buffer(context, context->get_model()->get_buffer_descriptor(prefix + "attention.wv"), layer_allocation);
    attention_wo = new llava_buffer(context, context->get_model()->get_buffer_descriptor(prefix + "attention.wo"), layer_allocation);
    feed_forward_w1 = new llava_buffer(context, context->get_model()->get_buffer_descriptor(prefix + "feed_forward.w1"), layer_allocation);
    feed_forward_w2 = new llava_buffer(context, context->get_model()->get_buffer_descriptor(prefix + "feed_forward.w2"), layer_allocation);
    feed_forward_w3 = new llava_buffer(context, context->get_model()->get_buffer_descriptor(prefix + "feed_forward.w3"), layer_allocation);
    attention_norm = new llava_buffer(context, context->get_model()->get_buffer_descriptor(prefix + "attention_norm"), layer_allocation);
    ffn_norm = new llava_buffer(context, context->get_model()->get_buffer_descriptor(prefix + "ffn_norm"), layer_allocation);
    k_cache = new llava_buffer(context, ggml_value_type::f32, context->backlog_size, context->get_model()->header.dim, layer_cache_allocation);
    v_cache = new llava_buffer(context, ggml_value_type::f32, context->backlog_size, context->get_model()->header.dim, layer_cache_allocation);
}

llava_layer::~llava_layer() {
    delete attention_wq;
    delete attention_wk;
    delete attention_wv;
    delete attention_wo;
    delete feed_forward_w1;
    delete feed_forward_w2;
    delete feed_forward_w3;
    delete attention_norm;
    delete ffn_norm;
    delete k_cache;
    delete v_cache;
    delete layer_allocation;
    delete layer_cache_allocation;
    delete[] raw_layer;
}

void llava_layer::execute(llava_command_buffer *cmd_buf) const {
    llava_context *ctx = cmd_buf->context;
    cmd_buf->normalize_logit(ctx->current_thought_sublayer, ctx->current_thought, attention_norm);
    cmd_buf->matmul(ctx->current_Q, attention_wq, ctx->current_thought_sublayer);
    cmd_buf->matmul(ctx->current_K, attention_wk, ctx->current_thought_sublayer);
    cmd_buf->rope(ctx->current_Q);
    cmd_buf->rope(ctx->current_K);

    cmd_buf->kv_copy(k_cache, ctx->current_K);
    cmd_buf->multi_head_attention(ctx->attn_result, k_cache, ctx->current_Q);
    cmd_buf->inplace_softmax(ctx->attn_result);
    cmd_buf->matmul(ctx->current_V, attention_wv, ctx->current_thought_sublayer);
    cmd_buf->kv_copy(v_cache, ctx->current_V);

    cmd_buf->perform_kqv_matching(ctx->current_Vout, v_cache, ctx->attn_result);
    cmd_buf->matmul_add_inplace(ctx->current_thought, attention_wo, ctx->current_Vout);
    cmd_buf->normalize_logit(ctx->current_thought_middle_normd, ctx->current_thought, ffn_norm);
    cmd_buf->matmul_silu_ff(ctx->properties_associated_values, feed_forward_w3, feed_forward_w1, ctx->current_thought_middle_normd);
    cmd_buf->matmul_add_inplace(ctx->current_thought, feed_forward_w2, ctx->properties_associated_values);
}

void llava_layer::freeze_storage() {
    layer_allocation->freeze();
}

void llava_layer::freeze_cache_storage() {
    layer_cache_allocation->freeze();
}

void llava_layer::load_to_gpu() {
    void* mapping = layer_allocation->map();
    if (raw_layer) {
        ::memcpy(mapping, raw_layer, layer_allocation->get_size());
    } else {
        attention_wq->load_from_disk(mapping);
        attention_wk->load_from_disk(mapping);
        attention_wv->load_from_disk(mapping);
        attention_wo->load_from_disk(mapping);
        feed_forward_w1->load_from_disk(mapping);
        feed_forward_w2->load_from_disk(mapping);
        feed_forward_w3->load_from_disk(mapping);
        attention_norm->load_from_disk(mapping);
        ffn_norm->load_from_disk(mapping);
    }
    layer_allocation->unmap();
}

void llava_layer::load_to_host() {
    if (raw_layer != nullptr) {
        return;
    }
    raw_layer = new u8[layer_allocation->get_size()];
    attention_wq->load_from_disk(raw_layer);
    attention_wk->load_from_disk(raw_layer);
    attention_wv->load_from_disk(raw_layer);
    attention_wo->load_from_disk(raw_layer);
    feed_forward_w1->load_from_disk(raw_layer);
    feed_forward_w2->load_from_disk(raw_layer);
    feed_forward_w3->load_from_disk(raw_layer);
    attention_norm->load_from_disk(raw_layer);
    ffn_norm->load_from_disk(raw_layer);
}

bool llava_layer::is_layer_data_offloaded() const {
    return is_offloaded;
}

llava_layer::llava_layer(llava_layer && other) noexcept : context(other.context),
                                                          layer_id(other.layer_id),
                                                          offload_layer_id(other.offload_layer_id)
                                                          {
    layer_allocation = other.layer_allocation;
    other.layer_allocation = nullptr;
    layer_cache_allocation = other.layer_cache_allocation;
    other.layer_cache_allocation = nullptr;
    attention_wq = other.attention_wq;
    other.attention_wq = nullptr;
    attention_wk = other.attention_wk;
    other.attention_wk = nullptr;
    attention_wv = other.attention_wv;
    other.attention_wv = nullptr;
    attention_wo = other.attention_wo;
    other.attention_wo = nullptr;
    feed_forward_w1 = other.feed_forward_w1;
    other.feed_forward_w1 = nullptr;
    feed_forward_w2 = other.feed_forward_w2;
    other.feed_forward_w2 = nullptr;
    feed_forward_w3 = other.feed_forward_w3;
    other.feed_forward_w3 = nullptr;
    attention_norm = other.attention_norm;
    other.attention_norm = nullptr;
    ffn_norm = other.ffn_norm;
    other.ffn_norm = nullptr;
    k_cache = other.k_cache;
    other.k_cache = nullptr;
    v_cache = other.v_cache;
    other.v_cache = nullptr;
    raw_layer = other.raw_layer;
    other.raw_layer = nullptr;
}

bool llava_layer::is_offload_main_layer() const {
    return layer_id == offload_layer_id;
}

u32 llava_layer::get_offload_id() const {
    return offload_layer_id;
}

void llava_layer::set_offload(u32 other_layer) {
    is_offloaded = true;
    if (offload_layer_id == other_layer) {
        return;
    }
    offload_layer_id = other_layer;
    context->layers.at(other_layer).layer_allocation->join_memory_pool(layer_allocation);
}
