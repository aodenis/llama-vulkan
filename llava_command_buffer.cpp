#include "llava_command_buffer.h"
#include "llava_context.h"
#include "utils.h"

llava_command_buffer::llava_command_buffer(llava_context *_context) : context(_context),
                                                                      backlog_size(context->backlog_size),
                                                                      workgroup_size(context->workgroup_size) {

}

llava_command_buffer::~llava_command_buffer() = default;

void llava_command_buffer::record_execution() {
    vk::Event startEvent = nullptr;
    auto model = context->get_model();
    for (auto& layer : context->get_layers()) {
        startEvent = layer.execute(this, {startEvent});
    }

    startEvent = normalize_logit(context->current_thought_sublayer, context->current_thought, context->norm_w, {startEvent});
    matmul(context->output_probs, context->output_w, context->current_thought_sublayer, {startEvent});

    command_buffer_raw.reserve(command_buffer.size());
    for (auto& command : command_buffer) {
        command_buffer_raw.push_back(command.commandBuffer);
    }
}

void llava_command_buffer::reset_events() {
    for (auto& x : command_buffer) {
        context->get_device().resetEvent(x.completionEvent);
    }
}

void llava_command_buffer::run() {
    vk::SubmitInfo submitInfo({}, {}, command_buffer_raw, {});
    context->get_queue().submit(submitInfo);
}

vk::Event llava_command_buffer::normalize_logit(llava_buffer* outbuf, llava_buffer* inbuf, llava_buffer* weights, initializer_list<vk::Event> events) {
    auto model = context->get_model();
    return record_command("normalize", {outbuf, inbuf, weights}, events, 1, 1, context->batch_size);
}

vk::Event llava_command_buffer::matmul(llava_buffer* outbuf, llava_buffer* matrix, llava_buffer* inbuf, initializer_list<vk::Event> events) {
    auto model = context->get_model();
    auto& spevar = context->get_spevar_struct();
    assert(inbuf->shape.first == matrix->shape.second);
    assert(outbuf->shape.first == matrix->shape.first);
    assert(inbuf->shape.second == context->batch_size);
    assert(outbuf->shape.second == context->batch_size);

    if (inbuf->shape.first == model->header.dim) {
        assert(outbuf->shape.first % (4 * spevar.matmul_dim_row_per_wavefront) == 0);
        return record_command("matmul_dim", {outbuf, matrix, inbuf}, events, outbuf->shape.first / (spevar.matmul_dim_row_per_wavefront * 4), 1, context->batch_size);
    } else if (inbuf->shape.first == model->ff_size) {
        assert(outbuf->shape.first % (4 * spevar.matmul_ff_row_per_wavefront) == 0);
        return record_command("matmul_ff", {outbuf, matrix, inbuf}, events, outbuf->shape.first / (spevar.matmul_ff_row_per_wavefront * 4), 1, context->batch_size);
    } else {
        assert(false);
    }
}

vk::Event llava_command_buffer::matmul_add_inplace(llava_buffer* outbuf, llava_buffer* matrix, llava_buffer* inbuf, initializer_list<vk::Event> events) {
    auto model = context->get_model();
    auto& spevar = context->get_spevar_struct();
    assert(inbuf->shape.first == matrix->shape.second);
    assert(outbuf->shape.first == matrix->shape.first);
    assert(inbuf->shape.second == context->batch_size);
    assert(outbuf->shape.second == context->batch_size);

    if (inbuf->shape.first == model->header.dim) {
        assert(outbuf->shape.first % (4 * spevar.matmul_dim_row_per_wavefront) == 0);
        return record_command("matmul_add_dim", {outbuf, matrix, inbuf}, events, outbuf->shape.first / (spevar.matmul_dim_row_per_wavefront * 4), 1, context->batch_size);
    } else if (inbuf->shape.first == model->ff_size) {
        assert(outbuf->shape.first % (4 * spevar.matmul_ff_row_per_wavefront) == 0);
        return record_command("matmul_add_ff", {outbuf, matrix, inbuf}, events, outbuf->shape.first / (spevar.matmul_ff_row_per_wavefront * 4), 1, context->batch_size);
    } else {
        assert(false);
    }
}

vk::Event llava_command_buffer::matmul_silu_ff(llava_buffer* outbuf, llava_buffer* w3_matrix, llava_buffer* w1_matrix, llava_buffer* inbuf, initializer_list<vk::Event> events) {
    auto model = context->get_model();
    auto& spevar = context->get_spevar_struct();
    assert(w3_matrix->shape == w1_matrix->shape);
    assert(inbuf->shape.second == context->batch_size);
    assert(outbuf->shape.second == context->batch_size);
    assert(inbuf->shape.first == model->header.dim);
    assert(outbuf->shape.first == model->ff_size);

    assert(outbuf->shape.first % (4 * spevar.matmul_dim_row_per_wavefront) == 0);
    return record_command("matmul_silu_ff", {outbuf, w3_matrix, w1_matrix, inbuf}, events, outbuf->shape.first / (spevar.matmul_dim_row_per_wavefront * 4), 1, context->batch_size);
}

vk::Event llava_command_buffer::kv_copy(llava_buffer* out_cache, llava_buffer* input_line, initializer_list<vk::Event> events) {
    auto model = context->get_model();

    assert(out_cache->shape.first == backlog_size);
    assert(out_cache->shape.second == model->header.dim);
    assert(input_line->shape.first == model->header.dim);
    assert(input_line->shape.second == context->batch_size);
    return record_command("copy_to_cache", {context->config_buffer, out_cache, input_line}, events, model->header.dim, 1, context->batch_size);
}

vk::Event llava_command_buffer::multi_head_attention(llava_buffer* out_buffer, llava_buffer* cache_buffer, llava_buffer* query, initializer_list<vk::Event> events) {
    auto model = context->get_model();


    assert(cache_buffer->shape.first == backlog_size);
    assert(cache_buffer->shape.second == model->header.dim);

    assert(query->shape.first == model->header.dim);
    assert(query->shape.second == context->batch_size);

    assert(out_buffer->shape.first == backlog_size);
    assert(out_buffer->shape.second == model->header.n_heads * context->batch_size);

    return record_command("mhsa", {out_buffer, cache_buffer, query}, events, updiv(model->header.n_heads * backlog_size, workgroup_size), 1, context->batch_size);
}

vk::Event llava_command_buffer::inplace_softmax(llava_buffer* inout_buffer, initializer_list<vk::Event> events) {
    auto model = context->get_model();



    assert(inout_buffer->shape.first == backlog_size);
    assert(inout_buffer->shape.second == model->header.n_heads * context->batch_size);

    return record_command("softmax", {context->config_buffer, inout_buffer}, events, updiv(model->header.n_heads, context->get_spevar_struct().softmax_head_per_wavefront), 1, context->batch_size);
}

vk::Event llava_command_buffer::rope(llava_buffer* buf, initializer_list<vk::Event> events) {
    auto model = context->get_model();

    assert((model->header.rot % 2) == 0);
    return record_command("rope", {context->config_buffer, buf}, events, updiv(buf->shape.first / 2, workgroup_size), 1, context->batch_size);
}

vk::Event llava_command_buffer::perform_kqv_matching(llava_buffer* v_out, llava_buffer* v_cache, llava_buffer* softmax_out, initializer_list<vk::Event> events) {
    auto model = context->get_model();


    assert(v_out and v_cache and softmax_out);

    assert(v_out->shape.first == model->header.dim);
    assert(v_out->shape.second == context->batch_size);

    assert(v_cache->shape.first == backlog_size);
    assert(v_cache->shape.second == model->header.dim);

    assert(softmax_out->shape.first == backlog_size);
    assert(softmax_out->shape.second == model->header.n_heads * context->batch_size);

    return record_command("kqv_matching", {v_out, v_cache, softmax_out}, events, updiv(model->header.dim, context->get_spevar_struct().softmax_head_per_wavefront), 1, context->batch_size);
}

vk::Event llava_command_buffer::record_command(llava_pipeline *pipeline,
                                               const initializer_list<llava_buffer *> &buffers,
                                               const initializer_list<vk::Event> &events,
                                               u32 countX,
                                               u32 countY,
                                               u32 countZ) {
    return command_buffer.emplace_back(pipeline, buffers, events, countX, countY, countZ).completionEvent;
}

vk::Event llava_command_buffer::record_command(const string &pipeline_name,
                                               const initializer_list<llava_buffer *> &buffers,
                                               const initializer_list<vk::Event> &events,
                                               u32 countX,
                                               u32 countY,
                                               u32 countZ) {
    u32 buffer_count = 0;
    for(llava_buffer * buffer : buffers) {
        buffer_count += buffer->get_sub_buffers().size();
    }
    auto* pipeline = context->get_pipeline(pipeline_name, buffer_count);
    return record_command(pipeline, buffers, events, countX, countY, countZ);
}
