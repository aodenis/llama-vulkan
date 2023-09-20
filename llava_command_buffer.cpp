#include "llava_command_buffer.h"
#include "llava_context.h"
#include "llava_session.h"
#include "utils.h"

llava_command_buffer::llava_command_buffer(llava_session *_session) : session(_session),
                                                                      backlog_size(session->backlog_size),
                                                                      workgroup_size(session->ctx->workgroup_size),
                                                                      batch_size(session->batch_size),
                                                                      fence(session->ctx->device.createFence({})) {

}

llava_command_buffer::~llava_command_buffer() {
    wait_idle();
    lock_guard guard1(session->ctx->command_pool_mutex);
    lock_guard guard2(session->ctx->descriptor_pool_mutex);
    session->ctx->get_device().destroy(fence);
    fence = nullptr;
    buffer_to_last_write_event.clear();
    command_buffer_raw.clear();
    command_buffer.clear();
}

void llava_command_buffer::wait_idle() const {
    (void) session->ctx->get_device().waitForFences(1, &fence, true, 1000000000000UL);
}

void llava_command_buffer::record_execution() {
    assert(command_buffer_raw.empty());

    llava_buffer *current_logit = session->current_thought;
    assert(session->get_layer_data().size() == session->ctx->get_layers().size());
    for (u32 i = 0; i < session->ctx->get_layers().size(); ++i) {
        current_logit = session->ctx->get_layers().at(i).execute(this, session->get_layer_data().at(i), current_logit);
    }

    normalize_logit(session->current_thought_sublayer, current_logit, session->norm_w);
    matmul(session->output_probs, session->output_w, session->current_thought_sublayer);

    command_buffer_raw.reserve(command_buffer.size());
    for (auto &command: command_buffer) {
        command_buffer_raw.push_back(command.commandBuffer);
    }
}

void llava_command_buffer::run() {
    for (auto &x: command_buffer) {
        session->ctx->get_device().resetEvent(x.completionEvent);
    }
    session->ctx->device.resetFences({fence});

    vk::SubmitInfo submitInfo({}, {}, command_buffer_raw, {});
    session->ctx->get_queue().submit(submitInfo, fence);
}

void llava_command_buffer::normalize_logit(llava_buffer *outbuf, llava_buffer *inbuf, llava_buffer *weights) {
    return record_command("normalize", {outbuf, inbuf, weights}, 1, 1, batch_size);
}

void llava_command_buffer::matmul(llava_buffer *outbuf, llava_buffer *matrix, llava_buffer *inbuf) {
    auto const *model = session->model;
    auto &spevar = session->get_spevar_struct();
    assert(inbuf->shape.first == matrix->shape.second);
    assert(outbuf->shape.first == matrix->shape.first);
    assert(inbuf->shape.second == batch_size);
    assert(outbuf->shape.second == batch_size);

    string suffix;
    if (matrix->type == ggml_value_type::q8_0) {
        suffix += "_q8";
    }
    if (matrix->weight_buffer_is_f16()) {
        suffix += "_fp16";
    }
    if (inbuf->shape.first == model->header.dim) {
        assert(outbuf->shape.first % (4 * spevar.matmul_dim_row_per_wavefront) == 0);
        return record_command("matmul_dim" + suffix, {outbuf, matrix, inbuf}, outbuf->shape.first / (spevar.matmul_dim_row_per_wavefront * 4), 1, batch_size);
    } else if (inbuf->shape.first == model->ff_size) {
        assert(outbuf->shape.first % (4 * spevar.matmul_ff_row_per_wavefront) == 0);
        return record_command("matmul_ff" + suffix, {outbuf, matrix, inbuf}, outbuf->shape.first / (spevar.matmul_ff_row_per_wavefront * 4), 1, batch_size);
    } else {
        assert(false);
    }
}

void llava_command_buffer::matmul_add_inplace(llava_buffer *outbuf, llava_buffer *matrix, llava_buffer *inbuf) {
    auto const *model = session->model;
    auto &spevar = session->get_spevar_struct();
    assert(inbuf->shape.first == matrix->shape.second);
    assert(outbuf->shape.first == matrix->shape.first);
    assert(inbuf->shape.second == batch_size);
    assert(outbuf->shape.second == batch_size);

    string suffix;
    if (matrix->type == ggml_value_type::q8_0) {
        suffix += "_q8";
    }
    if (matrix->weight_buffer_is_f16()) {
        suffix += "_fp16";
    }
    if (inbuf->shape.first == model->header.dim) {
        assert(outbuf->shape.first % (4 * spevar.matmul_dim_row_per_wavefront) == 0);
        return record_command("matmul_add_dim" + suffix, {outbuf, matrix, inbuf}, outbuf->shape.first / (spevar.matmul_dim_row_per_wavefront * 4), 1, batch_size);
    } else if (inbuf->shape.first == model->ff_size) {
        assert(outbuf->shape.first % (4 * spevar.matmul_ff_row_per_wavefront) == 0);
        return record_command("matmul_add_ff" + suffix, {outbuf, matrix, inbuf}, outbuf->shape.first / (spevar.matmul_ff_row_per_wavefront * 4), 1, batch_size);
    } else {
        assert(false);
    }
}

void llava_command_buffer::matmul_silu_ff(llava_buffer *outbuf, llava_buffer *w3_matrix, llava_buffer *w1_matrix, llava_buffer *inbuf) {
    auto const *model = session->model;
    auto &spevar = session->get_spevar_struct();
    assert(w3_matrix->shape == w1_matrix->shape);
    assert(w3_matrix->type == w1_matrix->type);
    assert(inbuf->shape.second == batch_size);
    assert(outbuf->shape.second == batch_size);
    assert(inbuf->shape.first == model->header.dim);
    assert(outbuf->shape.first == model->ff_size);

    string suffix;
    if (w3_matrix->type == ggml_value_type::q8_0) {
        suffix += "_q8";
    }
    if (w3_matrix->weight_buffer_is_f16()) {
        suffix += "_fp16";
    }
    assert(outbuf->shape.first % (4 * spevar.matmul_dim_row_per_wavefront) == 0);
    return record_command("matmul_silu_ff" + suffix, {outbuf, w3_matrix, w1_matrix, inbuf}, outbuf->shape.first / (spevar.matmul_dim_row_per_wavefront * 4), 1, batch_size);
}

void llava_command_buffer::kv_copy(llava_buffer *out_cache, llava_buffer *input_line) {
    auto const *model = session->model;

    assert(out_cache->shape.first == backlog_size);
    assert(out_cache->shape.second == model->header.dim);
    assert(input_line->shape.first == model->header.dim);
    assert(input_line->shape.second == batch_size);
    return record_command("copy_to_cache", {out_cache, session->config_buffer, input_line}, model->header.dim, 1, batch_size);
}

void llava_command_buffer::copy_logit(llava_buffer *out_logit, llava_buffer *input_logit) {
    auto const *model = session->model;

    assert(out_logit->shape.first == model->header.dim);
    assert(out_logit->shape.second == batch_size);
    assert(input_logit->shape.first == model->header.dim);
    assert(input_logit->shape.second == batch_size);
    return record_command("copy", {out_logit, input_logit}, model->header.dim, 1, batch_size);
}

void llava_command_buffer::multi_head_attention(llava_buffer *out_buffer, llava_buffer *cache_buffer, llava_buffer *query) {
    auto const *model = session->model;


    assert(cache_buffer->shape.first == backlog_size);
    assert(cache_buffer->shape.second == model->header.dim);

    assert(query->shape.first == model->header.dim);
    assert(query->shape.second == batch_size);

    assert(out_buffer->shape.first == backlog_size);
    assert(out_buffer->shape.second == model->header.n_heads * batch_size);

    return record_command("mhsa", {out_buffer, session->config_buffer, cache_buffer, query}, updiv(model->header.n_heads * backlog_size, workgroup_size), 1, batch_size);
}

void llava_command_buffer::inplace_softmax(llava_buffer *inout_buffer) {
    auto const *model = session->model;

    assert(inout_buffer->shape.first == backlog_size);
    assert(inout_buffer->shape.second == model->header.n_heads * batch_size);

    return record_command("softmax", {inout_buffer, session->config_buffer}, updiv(model->header.n_heads, session->get_spevar_struct().softmax_head_per_wavefront), 1, batch_size);
}

void llava_command_buffer::perform_kqv_matching(llava_buffer *v_out, llava_buffer *v_cache, llava_buffer *softmax_out) {
    auto const *model = session->model;

    assert(v_out and v_cache and softmax_out);

    assert(v_out->shape.first == model->header.dim);
    assert(v_out->shape.second == batch_size);

    assert(v_cache->shape.first == backlog_size);
    assert(v_cache->shape.second == model->header.dim);

    assert(softmax_out->shape.first == backlog_size);
    assert(softmax_out->shape.second == model->header.n_heads * batch_size);

    return record_command("kqv_matching", {v_out, v_cache, softmax_out}, updiv(model->header.dim, session->get_spevar_struct().softmax_head_per_wavefront), 1, batch_size);
}

void llava_command_buffer::record_command(const string &pipeline_name,
                                          const initializer_list<llava_buffer *> &l_buffers,
                                          u32 countX,
                                          u32 countY,
                                          u32 countZ) {
    llava_context *context = session->ctx;
    assert(l_buffers.size() != 0);
    llava_buffer *const output_buffer = *(l_buffers.begin());

    u32 buffer_count = 0;
    for (llava_buffer *buffer: l_buffers) {
        buffer_count += buffer->get_sub_buffers().size();
    }

    vector<vk::DescriptorBufferInfo> buffersInfo;
    vector<vk::WriteDescriptorSet> writes;
    writes.reserve(buffer_count);
    buffersInfo.reserve(buffer_count);

    auto *pipeline = context->get_pipeline(pipeline_name, buffer_count, session->get_spevar_struct());

    vk::DescriptorSet descriptorSet;
    {
        lock_guard guard(context->descriptor_pool_mutex);
        descriptorSet = context->get_device().allocateDescriptorSets({context->get_descriptor_pool(), 1, &pipeline->descriptorSetLayout}).front();
    }
    vk::CommandBuffer commandBuffer;
    {
        lock_guard guard(context->command_pool_mutex);
        commandBuffer = context->get_device().allocateCommandBuffers({context->get_command_pool(), vk::CommandBufferLevel::ePrimary, 1}).front();
    }
    vk::Event completionEvent(context->get_device().createEvent({}));

    vector<pair<vk::Buffer, bool>> buffers;
    for (llava_buffer *buffer: l_buffers) {
        bool is_dynamic_buffer = buffer->backing_buffer_name.empty();
        assert(buffer->is_allocated());
        for (auto &sub_buffer: buffer->get_sub_buffers()) {
            buffers.emplace_back(*(sub_buffer.buffer), is_dynamic_buffer);
        }
    }

    for (llava_buffer *l_buffer: l_buffers) {
        for (auto &buffer: l_buffer->get_sub_buffers()) {
            buffersInfo.emplace_back(*(buffer.buffer), 0, buffer.size);
        }
    }

    for (uint32_t i = 0; i < buffersInfo.size(); ++i) {
        writes.emplace_back(descriptorSet, i, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, buffersInfo.data() + i);
    }

    context->get_device().updateDescriptorSets(writes, {});

    vector<vk::BufferMemoryBarrier> barriers;
    vector<vk::Event> events;
    vector<vk::BufferMemoryBarrier> host_barriers;
    for (llava_buffer *buffer: l_buffers) {
        assert(buffer->is_allocated());
        if (auto it = buffer_to_last_write_event.find(buffer); it != buffer_to_last_write_event.end()) {
            // Wait for it !
            events.emplace_back(it->second);
            for (auto &sub_buffer: buffer->get_sub_buffers()) {
                auto dstAccessMask = (buffer == output_buffer) ? (vk::AccessFlagBits::eShaderWrite) : (vk::AccessFlagBits::eShaderRead);
                auto srcAccessMask = ((buffer == session->config_buffer) or ((buffer == session->current_thought) and (not buffer_to_last_write_event.contains(buffer))))
                                     ? (vk::AccessFlagBits::eHostWrite) : (vk::AccessFlagBits::eShaderWrite);
                barriers.emplace_back(srcAccessMask, dstAccessMask, context->get_queue_family_index(), context->get_queue_family_index(), *(sub_buffer.buffer), 0, sub_buffer.size);
            }
        }
    }

    buffer_to_last_write_event[output_buffer] = completionEvent;

    commandBuffer.begin(vk::CommandBufferBeginInfo());
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline->pipelineLayout, 0, descriptorSet, {});
    if (not events.empty()) {
        commandBuffer.waitEvents(events, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, barriers, {});
    }
    commandBuffer.dispatch(countX, countY, countZ);
    commandBuffer.setEvent(completionEvent, vk::PipelineStageFlagBits::eComputeShader);
    commandBuffer.end();

    command_buffer.emplace_back(context, descriptorSet, commandBuffer, completionEvent);
}

llava_wrapped_command::~llava_wrapped_command() {
    context->get_device().freeCommandBuffers(context->get_command_pool(), commandBuffer);
    context->get_device().freeDescriptorSets(context->get_descriptor_pool(), descriptorSet);
    context->get_device().destroy(completionEvent);
}

llava_wrapped_command::llava_wrapped_command(llava_context *_context,
                                             vk::DescriptorSet _descriptorSet,
                                             vk::CommandBuffer _commandBuffer,
                                             vk::Event _completionEvent) : context(_context),
                                                                           descriptorSet(_descriptorSet),
                                                                           commandBuffer(_commandBuffer),
                                                                           completionEvent(_completionEvent) {

}
