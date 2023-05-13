#include <iostream>
#include "llava_buffer.h"
#include "llava_layer.h"
#include "llava_device_memory.h"
#include "llava_command_buffer.h"
#include "llava_context.h"

llava_layer::llava_layer(llava_context *context, u32 layer_id) {
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
}

vk::Event llava_layer::execute(llava_command_buffer *cmd_buf, vk::Event event) const {
    llava_context *ctx = cmd_buf->context;
    vk::Event evt_norm = cmd_buf->normalize_logit(ctx->current_thought_sublayer, ctx->current_thought, attention_norm, {event});
    vk::Event evtQ = cmd_buf->matmul(ctx->current_Q, attention_wq, ctx->current_thought_sublayer, {evt_norm});
    vk::Event evtK = cmd_buf->matmul(ctx->current_K, attention_wk, ctx->current_thought_sublayer, {evt_norm});
    vk::Event evtRoPEQ = cmd_buf->rope(ctx->current_Q, {evtQ});
    vk::Event evtRoPEK = cmd_buf->rope(ctx->current_K, {evtK});

    vk::Event evtKc = cmd_buf->kv_copy(k_cache, ctx->current_K, {evtRoPEK});
    vk::Event evtSA = cmd_buf->multi_head_attention(ctx->attn_result, k_cache, ctx->current_Q, {evtRoPEQ, evtKc});
    vk::Event evtSoftmax = cmd_buf->inplace_softmax(ctx->attn_result, {evtSA});
    vk::Event evtV = cmd_buf->matmul(ctx->current_V, attention_wv, ctx->current_thought_sublayer, {evt_norm});
    vk::Event evtVc = cmd_buf->kv_copy(v_cache, ctx->current_V, {evtV});

    vk::Event evtKQV = cmd_buf->perform_kqv_matching(ctx->current_Vout, v_cache, ctx->attn_result, {evtSoftmax, evtVc});
    vk::Event evtSA_out = cmd_buf->matmul_add_inplace(ctx->current_thought, attention_wo, ctx->current_Vout, {evtKQV});
    vk::Event evt_norm_ff = cmd_buf->normalize_logit(ctx->current_thought_middle_normd, ctx->current_thought, ffn_norm, {evtSA_out});
    vk::Event evt_w13 = cmd_buf->matmul_silu_ff(ctx->properties_associated_values, feed_forward_w3, feed_forward_w1, ctx->current_thought_middle_normd, {evt_norm_ff}); // This operation takes forever to complete, TODO optimize it
    vk::Event evt_w2 = cmd_buf->matmul_add_inplace(ctx->current_thought, feed_forward_w2, ctx->properties_associated_values, {evt_w13}); // This operation takes forever to complete, TODO optimize it
    return evt_w2;
}

void llava_layer::freeze_storage() {
    layer_allocation->freeze();
    layer_cache_allocation->freeze();
}

void llava_layer::load_from_disk() {
    attention_wq->load_from_disk();
    attention_wk->load_from_disk();
    attention_wv->load_from_disk();
    attention_wo->load_from_disk();
    feed_forward_w1->load_from_disk();
    feed_forward_w2->load_from_disk();
    feed_forward_w3->load_from_disk();
    attention_norm->load_from_disk();
    ffn_norm->load_from_disk();
}
