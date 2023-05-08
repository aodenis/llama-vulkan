#include <iostream>
#include "llava_buffer.h"
#include "llava_layer.h"
#include "llava_device_memory.h"
#include "llava_context.h"

llava_layer::llava_layer(llava_context *context, u32 layer_id) {
    layer_allocation = new llava_device_memory(context);
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
    k_cache = new llava_buffer(context, ggml_value_type::f32, context->backlog_size, context->get_model()->header.dim);
    v_cache = new llava_buffer(context, ggml_value_type::f32, context->backlog_size, context->get_model()->header.dim);
    if (context->allocate_buffers) {
        layer_allocation->freeze();
    }
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
}

vk::Event llava_layer::execute(llava_context *ctx, vk::Event event) {
    vk::Event evt_norm = ctx->normalize_logit(ctx->current_thought_sublayer, ctx->current_thought, attention_norm, {event});

    vk::Event evtQ = ctx->matmul(ctx->current_Q, attention_wq, ctx->current_thought_sublayer, {evt_norm});
    vk::Event evtK = ctx->matmul(ctx->current_K, attention_wk, ctx->current_thought_sublayer, {evt_norm});
    vk::Event evtRoPEQ = ctx->rope(ctx->current_Q, {evtQ});
    vk::Event evtRoPEK = ctx->rope(ctx->current_K, {evtK});

    vk::Event evtV = ctx->matmul(ctx->current_V, attention_wv, ctx->current_thought_sublayer, {evt_norm});
    vk::Event evtKc = ctx->kv_copy(k_cache, ctx->current_K, {evtRoPEK});
    vk::Event evtVc = ctx->kv_copy(v_cache, ctx->current_V, {evtV});
    vk::Event evtSA = ctx->multi_head_attention(ctx->attn_result, k_cache, ctx->current_Q, {evtRoPEQ, evtKc});
    vk::Event evtSoftmax = ctx->inplace_softmax(ctx->attn_result, {evtSA});
    vk::Event evtKQV = ctx->perform_kqv_matching(ctx->current_V, v_cache, ctx->attn_result, {evtSoftmax, evtVc}); // current_V is used a tmp buffer here
    vk::Event evtSA_out = ctx->matmul(ctx->current_thought_sublayer, attention_wo, ctx->current_V, {evtKQV});
    vk::Event evtSA_add = ctx->add(ctx->current_thought, ctx->current_thought_sublayer, {evtSA_out});

    vk::Event evt_norm_ff = ctx->normalize_logit(ctx->current_thought_sublayer, ctx->current_thought, ffn_norm, {evtSA_add});
    vk::Event evt_w13 = ctx->matmul_silu_ff(ctx->properties_associated_values, feed_forward_w3, feed_forward_w1, ctx->current_thought_sublayer, {evt_norm_ff}); // This operation takes forever to complete, TODO optimize it
    vk::Event evt_w2 = ctx->matmul(ctx->current_thought_sublayer, feed_forward_w2, ctx->properties_associated_values, {evt_w13}); // This operation takes forever to complete, TODO optimize it
    vk::Event evt_ff_add = ctx->add(ctx->current_thought, ctx->current_thought_sublayer, {evt_w2});
    return evt_ff_add;
}
