#include "llava_layer_session_data.h"
#include "llava_session.h"
#include "llava_context.h"
#include "llava_buffer.h"
#include "llava_device_memory.h"
#include "utils.h"
#include "llava_command_buffer.h"
#include "ggml_file.h"
#include <set>

llava_layer_session_data::llava_layer_session_data(llava_session* _session) : session(_session) {
    flush_buffers_on_gpu();
}

llava_layer_session_data::llava_layer_session_data(llava_layer_session_data && other) noexcept : session(other.session) {
    layer_cache_allocation = other.layer_cache_allocation;
    other.layer_cache_allocation = nullptr;
    k_cache = other.k_cache;
    other.k_cache = nullptr;
    v_cache = other.v_cache;
    other.v_cache = nullptr;
    attn_result = other.attn_result;
    other.attn_result = nullptr;
    ff_result = other.ff_result;
    other.ff_result = nullptr;
    normalized_input_logit = other.normalized_input_logit;
    post_attn_logit = other.post_attn_logit;
    post_attn_norm_logit = other.post_attn_norm_logit;
    output_logit = other.output_logit;
    other.normalized_input_logit = nullptr;
    other.post_attn_logit = nullptr;
    other.output_logit = nullptr;
    other.post_attn_norm_logit = nullptr;
}

void llava_layer_session_data::dump_tracing_layers(int out_fd, const string& layer_name) {
    this->layer_cache_allocation->dump_buffers(out_fd, layer_name);
}

llava_layer_session_data::~llava_layer_session_data() {
    delete k_cache;
    delete v_cache;
    delete attn_result;
    delete ff_result;
    delete normalized_input_logit;
    delete post_attn_logit;
    delete post_attn_norm_logit;
    delete output_logit;
    delete layer_cache_allocation;
}

void llava_layer_session_data::flush_buffers_on_gpu() {
    llava_context* context = session->ctx;
    ggml_file const* model = session->model;
    bool tracing_was_enabled = (attn_result != nullptr);
    bool tracing_enabled = this->session->is_tracing_enabled();
    u32 old_size = k_cache ? k_cache->shape.first : 0;
    u32 old_batch_size = ff_result ? ff_result->shape.second : 0;

    if ((session->backlog_size == old_size) and (tracing_was_enabled == tracing_enabled) and (not tracing_enabled or (old_batch_size == session->batch_size))) {
        return; // no change
    }

    u32 smallest_size = min(old_size, session->backlog_size);
    u32 dim = model->header.dim;
    u32 n_heads = model->header.n_heads;
    u32 ff_size = model->ff_size;

    auto *next_layer_cache_allocation = new llava_device_memory(context);
    auto *next_k_cache = new llava_buffer(context, ggml_value_type::f16, session->backlog_size, dim, next_layer_cache_allocation, "k_cache");
    auto *next_v_cache = new llava_buffer(context, ggml_value_type::f16, session->backlog_size, dim, next_layer_cache_allocation, "v_cache");
    llava_buffer *next_attn_result = nullptr;
    llava_buffer *next_ff_result = nullptr;
    llava_buffer *next_normalized_input_logit = nullptr;
    llava_buffer *next_post_attn_logit = nullptr;
    llava_buffer *next_output_logit = nullptr;
    llava_buffer *next_post_attn_norm_logit = nullptr;

    if (tracing_enabled and session->batch_size) {
        next_attn_result = new llava_buffer(context, ggml_value_type::f32, session->backlog_size, n_heads * session->batch_size, next_layer_cache_allocation, "attn_result");
        next_ff_result = new llava_buffer(context, ggml_value_type::f32, ff_size, session->batch_size, next_layer_cache_allocation, "ff_result");
        next_normalized_input_logit = new llava_buffer(context, ggml_value_type::f32, dim, session->batch_size, next_layer_cache_allocation, "normalized_input_logit");
        next_post_attn_logit = new llava_buffer(context, ggml_value_type::f32, dim, session->batch_size, next_layer_cache_allocation, "post_attn_logit");
        next_post_attn_norm_logit = new llava_buffer(context, ggml_value_type::f32, dim, session->batch_size, next_layer_cache_allocation, "post_attn_norm_logit");
        next_output_logit = new llava_buffer(context, ggml_value_type::f32, dim, session->batch_size, next_layer_cache_allocation, "output_logit");
    }

    u32 batch_size_to_copy = min(old_batch_size, session->batch_size);
    u32 batch_size_src_offset = old_batch_size - batch_size_to_copy;
    u32 batch_size_dst_offset = session->batch_size - batch_size_to_copy;

    next_layer_cache_allocation->freeze();

    vector<vk::CommandBuffer> commandBuffers;
    {
        lock_guard guard(context->command_pool_mutex);
        commandBuffers = context->get_device().allocateCommandBuffers({context->get_command_pool(), vk::CommandBufferLevel::ePrimary, 1});
    }
    commandBuffers.front().begin(vk::CommandBufferBeginInfo());
    if (layer_cache_allocation != nullptr) {
        commandBuffers.front().copyBuffer(*(k_cache->get_sub_buffers().front().buffer), *(next_k_cache->get_sub_buffers().front().buffer), {{0, 0, 2 * smallest_size * dim}});
        commandBuffers.front().copyBuffer(*(v_cache->get_sub_buffers().front().buffer), *(next_v_cache->get_sub_buffers().front().buffer), {{0, 0, 2 * smallest_size * dim}});
        if(smallest_size < session->backlog_size) {
            commandBuffers.front().fillBuffer( *(next_k_cache->get_sub_buffers().front().buffer), 2 * smallest_size * dim, VK_WHOLE_SIZE, 0);
            commandBuffers.front().fillBuffer( *(next_v_cache->get_sub_buffers().front().buffer), 2 * smallest_size * dim, VK_WHOLE_SIZE, 0);
        }
        /* if (next_ff_result and ff_result) {
            commandBuffers.front().copyBuffer(*(ff_result->get_sub_buffers().front().buffer), *(next_ff_result->get_sub_buffers().front().buffer), {{batch_size_src_offset * 4 * ff_size, batch_size_dst_offset * 4 * ff_size, batch_size_to_copy * 4 * ff_size}});
        }
        if (next_normalized_input_logit and normalized_input_logit) {
            commandBuffers.front().copyBuffer(*(normalized_input_logit->get_sub_buffers().front().buffer), *(next_normalized_input_logit->get_sub_buffers().front().buffer), {{batch_size_src_offset * 4 * dim, batch_size_dst_offset * 4 * dim, batch_size_to_copy * 4 * dim}});
        }
        if (next_post_attn_logit and post_attn_logit) {
            commandBuffers.front().copyBuffer(*(post_attn_logit->get_sub_buffers().front().buffer), *(next_post_attn_logit->get_sub_buffers().front().buffer), {{batch_size_src_offset * 4 * dim, batch_size_dst_offset * 4 * dim, batch_size_to_copy * 4 * dim}});
        }
        if (next_post_attn_norm_logit and post_attn_norm_logit) {
            commandBuffers.front().copyBuffer(*(post_attn_norm_logit->get_sub_buffers().front().buffer), *(next_post_attn_norm_logit->get_sub_buffers().front().buffer), {{batch_size_src_offset * 4 * dim, batch_size_dst_offset * 4 * dim, batch_size_to_copy * 4 * dim}});
        }
        if (next_output_logit and output_logit) {
            commandBuffers.front().copyBuffer(*(output_logit->get_sub_buffers().front().buffer), *(next_output_logit->get_sub_buffers().front().buffer), {{batch_size_src_offset * 4 * dim, batch_size_dst_offset * 4 * dim, batch_size_to_copy * 4 * dim}});
        }
        if (next_attn_result and attn_result) {
            commandBuffers.front().copyBuffer(*(attn_result->get_sub_buffers().front().buffer), *(next_attn_result->get_sub_buffers().front().buffer), {{batch_size_src_offset * 4 * smallest_size * n_heads, batch_size_dst_offset * 4 * smallest_size * n_heads, batch_size_to_copy * 4 * smallest_size * n_heads}});
        } */
    } else {
        commandBuffers.front().fillBuffer( *(next_k_cache->get_sub_buffers().front().buffer), 0, VK_WHOLE_SIZE, 0);
        commandBuffers.front().fillBuffer( *(next_v_cache->get_sub_buffers().front().buffer), 0, VK_WHOLE_SIZE, 0);
    }

    commandBuffers.front().end();

    {
        lock_guard guard(context->queue_mutex);
        vk::SubmitInfo submitInfo({}, {}, commandBuffers, {});
        session->ctx->get_queue().submit(submitInfo);
        session->ctx->get_queue().waitIdle();
    }

    {
        lock_guard guard(context->command_pool_mutex);
        commandBuffers.clear();
    }

    delete attn_result;
    delete ff_result;
    delete normalized_input_logit;
    delete post_attn_logit;
    delete output_logit;
    delete post_attn_norm_logit;
    delete k_cache;
    delete v_cache;
    delete layer_cache_allocation;

    layer_cache_allocation = next_layer_cache_allocation;
    k_cache = next_k_cache;
    v_cache = next_v_cache;
    attn_result = next_attn_result;
    ff_result = next_ff_result;
    normalized_input_logit = next_normalized_input_logit;
    post_attn_logit = next_post_attn_logit;
    output_logit = next_output_logit;
    post_attn_norm_logit = next_post_attn_norm_logit;
}

void llava_layer_session_data::dump_kv_cache(u8 *dst, u32 token_count) {
    void* k_cache_mapping = k_cache->map(0, 0, token_count * 2 * session->model->header.dim);
    memcpy(dst, k_cache_mapping, 2 * session->model->header.dim * token_count);
    k_cache->unmap();
    void* v_cache_mapping = v_cache->map(0, 0, token_count * 2 * session->model->header.dim);
    memcpy(dst + 2 * session->model->header.dim * token_count, v_cache_mapping, 2 * session->model->header.dim * token_count);
    v_cache->unmap();
}

void llava_layer_session_data::restore_kv_cache(u8 const* src, u32 token_count) {
    void* k_cache_mapping = k_cache->map(0, 0, token_count * 2 * session->model->header.dim);
    memcpy(k_cache_mapping, src, 2 * session->model->header.dim * token_count);
    k_cache->unmap();
    void* v_cache_mapping = v_cache->map(0, 0, token_count * 2 * session->model->header.dim);
    memcpy(v_cache_mapping, src + 2 * session->model->header.dim * token_count, 2 * session->model->header.dim * token_count);
    v_cache->unmap();
}
