#include <iostream>
#include "llava_buffer.h"
#include "llava_layer.h"
#include "llava_layer_session_data.h"
#include "llava_device_memory.h"
#include "llava_command_buffer.h"
#include "llava_context.h"
#include "llava_session.h"

llava_layer::llava_layer(llava_context *_context, u32 _layer_id) : layer_id(_layer_id), context(_context) {
    layer_allocation = new llava_device_memory(context);
    auto* model = context->get_model();
    string prefix = "layers." + to_string(layer_id) + ".";
    attention_wq = new llava_buffer(context, model->get_buffer_descriptor(prefix + "attention.wq"), layer_allocation);
    attention_wk = new llava_buffer(context, model->get_buffer_descriptor(prefix + "attention.wk"), layer_allocation);
    attention_wv = new llava_buffer(context, model->get_buffer_descriptor(prefix + "attention.wv"), layer_allocation);
    attention_wo = new llava_buffer(context, model->get_buffer_descriptor(prefix + "attention.wo"), layer_allocation);
    feed_forward_w1 = new llava_buffer(context, model->get_buffer_descriptor(prefix + "feed_forward.w1"), layer_allocation);
    feed_forward_w2 = new llava_buffer(context, model->get_buffer_descriptor(prefix + "feed_forward.w2"), layer_allocation);
    feed_forward_w3 = new llava_buffer(context, model->get_buffer_descriptor(prefix + "feed_forward.w3"), layer_allocation);
    attention_norm = new llava_buffer(context, model->get_buffer_descriptor(prefix + "attention_norm"), layer_allocation);
    ffn_norm = new llava_buffer(context, model->get_buffer_descriptor(prefix + "ffn_norm"), layer_allocation);
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
    delete layer_allocation;
    delete[] raw_layer;
}

llava_buffer* llava_layer::execute(llava_command_buffer *cmd_buf, llava_layer_session_data* layer_data, llava_buffer* raw_input_logit) const {
    // if raw_input_logit is null, not recorded
    llava_session *session = layer_data->session;

    bool record = (session->batch_size == 1) and (layer_data->attn_result != nullptr);

    llava_buffer* c_input_logit = record ? raw_input_logit : session->current_thought;
    llava_buffer* c_input_norm_logit = record ? layer_data->normalized_input_logit : session->current_thought_sublayer;
    llava_buffer* c_post_attn_logit = record ? layer_data->post_attn_logit : session->current_thought;
    llava_buffer* c_post_attn_norm_logit = record ? layer_data->post_attn_norm_logit : session->current_thought_middle_normd;
    llava_buffer* c_output_logit = record ? layer_data->output_logit : session->current_thought;
    llava_buffer* c_attn_result = record ? layer_data->attn_result : session->main_attn_result;
    llava_buffer* c_ff_result = record ? layer_data->ff_result : session->main_ff_result;

    cmd_buf->normalize_logit(c_input_norm_logit, c_input_logit, attention_norm);
    cmd_buf->matmul(session->current_K, attention_wk, c_input_norm_logit);
    cmd_buf->matmul(session->current_Q, attention_wq, c_input_norm_logit);
    cmd_buf->matmul(session->current_V, attention_wv, c_input_norm_logit);

    cmd_buf->kv_copy(layer_data->k_cache, session->current_K);
    cmd_buf->kv_copy(layer_data->v_cache, session->current_V);
    cmd_buf->multi_head_attention(c_attn_result, layer_data->k_cache, session->current_Q);
    cmd_buf->inplace_softmax(c_attn_result);

    cmd_buf->perform_kqv_matching(session->current_Vout, layer_data->v_cache, c_attn_result);
    if (c_post_attn_logit != raw_input_logit) {
        cmd_buf->copy_logit(c_post_attn_logit, raw_input_logit);
    }
    cmd_buf->matmul_add_inplace(c_post_attn_logit, attention_wo, session->current_Vout);
    cmd_buf->normalize_logit(c_post_attn_norm_logit, c_post_attn_logit, ffn_norm);
    cmd_buf->matmul_silu_ff(c_ff_result, feed_forward_w3, feed_forward_w1, c_post_attn_norm_logit);
    if (c_post_attn_logit != c_output_logit) {
        cmd_buf->copy_logit(c_output_logit, c_post_attn_logit);
    }
    cmd_buf->matmul_add_inplace(c_output_logit, feed_forward_w2, c_ff_result);
    return c_output_logit;
}

void llava_layer::freeze_storage() {
    layer_allocation->freeze();
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

llava_layer::llava_layer(llava_layer && other) noexcept : context(other.context),
                                                          layer_id(other.layer_id)
                                                          {
    layer_allocation = other.layer_allocation;
    attention_wq = other.attention_wq;
    attention_wk = other.attention_wk;
    attention_wv = other.attention_wv;
    attention_wo = other.attention_wo;
    feed_forward_w1 = other.feed_forward_w1;
    feed_forward_w2 = other.feed_forward_w2;
    feed_forward_w3 = other.feed_forward_w3;
    attention_norm = other.attention_norm;
    ffn_norm = other.ffn_norm;
    raw_layer = other.raw_layer;
    other.layer_allocation = nullptr;
    other.attention_wq = nullptr;
    other.attention_wk = nullptr;
    other.attention_wv = nullptr;
    other.attention_wo = nullptr;
    other.feed_forward_w1 = nullptr;
    other.feed_forward_w2 = nullptr;
    other.feed_forward_w3 = nullptr;
    other.attention_norm = nullptr;
    other.ffn_norm = nullptr;
    other.raw_layer = nullptr;
}
