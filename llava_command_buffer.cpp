#include <iostream>
#include "llava_command_buffer.h"
#include "llava_context.h"
#include "utils.h"

struct layer_offload_data {
    llava_layer* layer = nullptr;
    vk::Event eof_event = nullptr;
    vk::Event load_event = nullptr;
};

struct worker_task {
    vector<layer_offload_data> managed_layers;
};

void offload_worker(llava_command_buffer *cmd_buf, worker_task* task) {
    llava_context* context = cmd_buf->context;
    u32 const managed_layer_count = task->managed_layers.size();
    context->get_device().setEvent(task->managed_layers.front().load_event);
    for (u32 i = 1; i < managed_layer_count; ++i) {
        context->get_device().resetEvent(task->managed_layers.at(i).load_event);
    }
    while(true) {
        {
            unique_lock lock(cmd_buf->offload_mutex);
            cmd_buf->offload_cv.wait(lock);
        }
        if (cmd_buf->thread_end_flag.load()) {
            delete task; // TODO thread safe ?
            return;
        }
        for (u32 i = 0; i < managed_layer_count; i++) {
            while (context->get_device().getEventStatus(task->managed_layers.at(i).eof_event) == vk::Result::eEventReset) {
                // TODO horrible
                usleep(10);
            }
            context->get_device().resetEvent(task->managed_layers.at(i).load_event);
            task->managed_layers.at((i + 1) % managed_layer_count).layer->load_to_gpu();
            context->get_device().setEvent(task->managed_layers.at((i+1) % managed_layer_count).load_event);
        }
    }
}

llava_command_buffer::llava_command_buffer(llava_context *_context) : context(_context),
                                                                      backlog_size(context->backlog_size),
                                                                      workgroup_size(context->workgroup_size) {
    thread_end_flag.store(false);
}

llava_command_buffer::~llava_command_buffer() {
    reset();
}

void llava_command_buffer::reset() {
    context->get_queue().waitIdle();
    thread_end_flag.store(true);
    offload_cv.notify_all();
    for (auto& x : offload_threads) {
        x.join();
    }
    offload_threads.clear();

    buffer_to_last_write_event.clear();
    command_buffer_raw.clear();
    layer_process_done_events.clear();
    command_buffer.clear();
    for(auto& event : layer_load_done_events) {
        if (event) {
            context->get_device().destroy(event);
        }
    }
    layer_load_done_events.clear();
    current_layer = ~0;
    thread_end_flag.store(false);
}

void llava_command_buffer::record_execution() {
    reset();
    layer_process_done_events.resize(context->get_layers().size());
    layer_load_done_events.resize(context->get_layers().size());
    current_layer = 0;
    map<u64, worker_task*> offload_mem_to_offload_id;

    for (auto& layer : context->get_layers()) {
        if (layer.is_layer_data_offloaded()) {
            layer_load_done_events[current_layer] = context->get_device().createEvent({});
        }
        layer.execute(this);
        if (not command_buffer.empty()) {
            layer_process_done_events[current_layer] = command_buffer.back().completionEvent;
        }
        if (layer.is_layer_data_offloaded()) {
            worker_task* task;
            if (not offload_mem_to_offload_id.contains(layer.get_offload_id())) {
                task = new worker_task;
                offload_mem_to_offload_id[layer.get_offload_id()] = task;
            } else {
                task = offload_mem_to_offload_id.at(layer.get_offload_id());
            }
            auto& managed_layer = task->managed_layers.emplace_back();
            managed_layer.layer = &layer;
            managed_layer.load_event = layer_load_done_events.at(current_layer);
            managed_layer.eof_event = layer_process_done_events.at(current_layer);
            assert(managed_layer.eof_event);
        }
        current_layer++;
    }
    current_layer = ~0U;

    normalize_logit(context->current_thought_sublayer, context->current_thought, context->norm_w);
    matmul(context->output_probs, context->output_w, context->current_thought_sublayer);

    command_buffer_raw.reserve(command_buffer.size());
    for (auto& command : command_buffer) {
        command_buffer_raw.push_back(command.commandBuffer);
    }

    for (auto& [_, b] : offload_mem_to_offload_id) {
        offload_threads.emplace_back(offload_worker, this, b);
    }
}

void llava_command_buffer::run() {
    for (auto& x : command_buffer) {
        context->get_device().resetEvent(x.completionEvent);
    }
    offload_cv.notify_all();

    vk::SubmitInfo submitInfo({}, {}, command_buffer_raw, {});
    context->get_queue().submit(submitInfo);
}

void llava_command_buffer::normalize_logit(llava_buffer* outbuf, llava_buffer* inbuf, llava_buffer* weights) {
    auto model = context->get_model();
    return record_command("normalize", {outbuf, inbuf, weights}, 1, 1, context->batch_size);
}

void llava_command_buffer::matmul(llava_buffer* outbuf, llava_buffer* matrix, llava_buffer* inbuf) {
    auto model = context->get_model();
    auto& spevar = context->get_spevar_struct();
    assert(inbuf->shape.first == matrix->shape.second);
    assert(outbuf->shape.first == matrix->shape.first);
    assert(inbuf->shape.second == context->batch_size);
    assert(outbuf->shape.second == context->batch_size);

    if (inbuf->shape.first == model->header.dim) {
        assert(outbuf->shape.first % (4 * spevar.matmul_dim_row_per_wavefront) == 0);
        return record_command("matmul_dim", {outbuf, matrix, inbuf}, outbuf->shape.first / (spevar.matmul_dim_row_per_wavefront * 4), 1, context->batch_size);
    } else if (inbuf->shape.first == model->ff_size) {
        assert(outbuf->shape.first % (4 * spevar.matmul_ff_row_per_wavefront) == 0);
        return record_command("matmul_ff", {outbuf, matrix, inbuf}, outbuf->shape.first / (spevar.matmul_ff_row_per_wavefront * 4), 1, context->batch_size);
    } else {
        assert(false);
    }
}

void llava_command_buffer::matmul_add_inplace(llava_buffer* outbuf, llava_buffer* matrix, llava_buffer* inbuf) {
    auto model = context->get_model();
    auto& spevar = context->get_spevar_struct();
    assert(inbuf->shape.first == matrix->shape.second);
    assert(outbuf->shape.first == matrix->shape.first);
    assert(inbuf->shape.second == context->batch_size);
    assert(outbuf->shape.second == context->batch_size);

    if (inbuf->shape.first == model->header.dim) {
        assert(outbuf->shape.first % (4 * spevar.matmul_dim_row_per_wavefront) == 0);
        return record_command("matmul_add_dim", {outbuf, matrix, inbuf}, outbuf->shape.first / (spevar.matmul_dim_row_per_wavefront * 4), 1, context->batch_size);
    } else if (inbuf->shape.first == model->ff_size) {
        assert(outbuf->shape.first % (4 * spevar.matmul_ff_row_per_wavefront) == 0);
        return record_command("matmul_add_ff", {outbuf, matrix, inbuf}, outbuf->shape.first / (spevar.matmul_ff_row_per_wavefront * 4), 1, context->batch_size);
    } else {
        assert(false);
    }
}

void llava_command_buffer::matmul_silu_ff(llava_buffer* outbuf, llava_buffer* w3_matrix, llava_buffer* w1_matrix, llava_buffer* inbuf) {
    auto model = context->get_model();
    auto& spevar = context->get_spevar_struct();
    assert(w3_matrix->shape == w1_matrix->shape);
    assert(inbuf->shape.second == context->batch_size);
    assert(outbuf->shape.second == context->batch_size);
    assert(inbuf->shape.first == model->header.dim);
    assert(outbuf->shape.first == model->ff_size);

    assert(outbuf->shape.first % (4 * spevar.matmul_dim_row_per_wavefront) == 0);
    return record_command("matmul_silu_ff", {outbuf, w3_matrix, w1_matrix, inbuf}, outbuf->shape.first / (spevar.matmul_dim_row_per_wavefront * 4), 1, context->batch_size);
}

void llava_command_buffer::kv_copy(llava_buffer* out_cache, llava_buffer* input_line) {
    auto model = context->get_model();

    assert(out_cache->shape.first == backlog_size);
    assert(out_cache->shape.second == model->header.dim);
    assert(input_line->shape.first == model->header.dim);
    assert(input_line->shape.second == context->batch_size);
    return record_command("copy_to_cache", {out_cache, context->config_buffer, input_line}, model->header.dim, 1, context->batch_size);
}

void llava_command_buffer::multi_head_attention(llava_buffer* out_buffer, llava_buffer* cache_buffer, llava_buffer* query) {
    auto model = context->get_model();


    assert(cache_buffer->shape.first == backlog_size);
    assert(cache_buffer->shape.second == model->header.dim);

    assert(query->shape.first == model->header.dim);
    assert(query->shape.second == context->batch_size);

    assert(out_buffer->shape.first == backlog_size);
    assert(out_buffer->shape.second == model->header.n_heads * context->batch_size);

    return record_command("mhsa", {out_buffer, cache_buffer, query}, updiv(model->header.n_heads * backlog_size, workgroup_size), 1, context->batch_size);
}

void llava_command_buffer::inplace_softmax(llava_buffer* inout_buffer) {
    auto model = context->get_model();



    assert(inout_buffer->shape.first == backlog_size);
    assert(inout_buffer->shape.second == model->header.n_heads * context->batch_size);

    return record_command("softmax", {inout_buffer, context->config_buffer}, updiv(model->header.n_heads, context->get_spevar_struct().softmax_head_per_wavefront), 1, context->batch_size);
}

void llava_command_buffer::rope(llava_buffer* buf) {
    auto model = context->get_model();

    assert((model->header.rot % 2) == 0);
    return record_command("rope", {buf, context->config_buffer}, updiv(buf->shape.first / 2, workgroup_size), 1, context->batch_size);
}

void llava_command_buffer::perform_kqv_matching(llava_buffer* v_out, llava_buffer* v_cache, llava_buffer* softmax_out) {
    auto model = context->get_model();


    assert(v_out and v_cache and softmax_out);

    assert(v_out->shape.first == model->header.dim);
    assert(v_out->shape.second == context->batch_size);

    assert(v_cache->shape.first == backlog_size);
    assert(v_cache->shape.second == model->header.dim);

    assert(softmax_out->shape.first == backlog_size);
    assert(softmax_out->shape.second == model->header.n_heads * context->batch_size);

    return record_command("kqv_matching", {v_out, v_cache, softmax_out}, updiv(model->header.dim, context->get_spevar_struct().softmax_head_per_wavefront), 1, context->batch_size);
}

void llava_command_buffer::record_command(const string &pipeline_name,
                                          const initializer_list<llava_buffer *> &l_buffers,
                                          u32 countX,
                                          u32 countY,
                                          u32 countZ) {
    assert(l_buffers.size() != 0);
    llava_buffer* const output_buffer = *(l_buffers.begin());

    u32 buffer_count = 0;
    for(llava_buffer * buffer : l_buffers) {
        buffer_count += buffer->get_sub_buffers().size();
    }

    vector<vk::DescriptorBufferInfo> buffersInfo;
    vector<vk::WriteDescriptorSet> writes;
    writes.reserve(buffer_count);
    buffersInfo.reserve(buffer_count);

    auto* pipeline = context->get_pipeline(pipeline_name, buffer_count);

    vk::DescriptorSet descriptorSet = context->get_device().allocateDescriptorSets({context->get_descriptor_pool(), 1, &pipeline->descriptorSetLayout}).front();
    vk::CommandBuffer commandBuffer = context->get_device().allocateCommandBuffers({context->get_command_pool(), vk::CommandBufferLevel::ePrimary, 1}).front();
    vk::Event completionEvent(context->get_device().createEvent({}));

    vector<pair<vk::Buffer, bool>> buffers;
    for (llava_buffer* buffer: l_buffers) {
        bool is_dynamic_buffer = buffer->backing_buffer_name.empty();
        assert(buffer->is_allocated());
        for(auto& sub_buffer : buffer->get_sub_buffers()) {
            buffers.emplace_back(sub_buffer.buffer, is_dynamic_buffer);
        }
    }

    for (llava_buffer* l_buffer: l_buffers) {
        for(auto& buffer : l_buffer->get_sub_buffers()) {
            buffersInfo.emplace_back(buffer.buffer, 0, buffer.size);
        }
    }

    for (uint32_t i = 0; i < buffersInfo.size(); ++i) {
        writes.emplace_back(descriptorSet, i, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, buffersInfo.data() + i);
    }

    context->get_device().updateDescriptorSets(writes, {});

    vector<vk::BufferMemoryBarrier> barriers;
    vector<vk::Event> events;
    vector<vk::BufferMemoryBarrier> host_barriers;
    for (llava_buffer* buffer: l_buffers) {
        assert(buffer->is_allocated());
        if (auto it = buffer_to_last_write_event.find(buffer); it != buffer_to_last_write_event.end()) {
            // Wait for it !
            events.emplace_back(it->second);
            for(auto& sub_buffer : buffer->get_sub_buffers()) {
                auto dstAccessMask = (buffer == output_buffer) ? (vk::AccessFlagBits::eShaderWrite) : (vk::AccessFlagBits::eShaderRead);
                barriers.emplace_back(vk::AccessFlagBits::eShaderWrite, dstAccessMask, context->get_queue_family_index(), context->get_queue_family_index(), sub_buffer.buffer, 0, sub_buffer.size);
            }
        } else if ((~current_layer) and layer_load_done_events.at(current_layer) and not buffer->backing_buffer_name.empty()) {
            for (auto& sub_buffer : buffer->get_sub_buffers()) {
                host_barriers.emplace_back(vk::AccessFlagBits::eHostWrite, vk::AccessFlagBits::eShaderRead, context->get_queue_family_index(), context->get_queue_family_index(), sub_buffer.buffer, 0, sub_buffer.size);
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
    if ((current_layer != ~0U) and (layer_load_done_events.at(current_layer)) and (not host_barriers.empty())) {
        commandBuffer.waitEvents({layer_load_done_events.at(current_layer)}, vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eComputeShader, {}, host_barriers, {});
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
