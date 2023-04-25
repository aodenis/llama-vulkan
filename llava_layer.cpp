#include "llava_layer.h"
#include "llava_context.h"
#include "llava_buffer.h"

llava_layer::llava_layer(llava_context *context, u32 layer_id) {
    string prefix = "layers." + to_string(layer_id) + ".";
    for (auto& buffer : context->get_model()->get_buffers()) {
        if (buffer.name.size() < prefix.size())
            continue;
        if (buffer.name.substr(0, prefix.size()) != prefix) {
            continue;
        }
        named_buffers.emplace(std::piecewise_construct, forward_as_tuple(buffer.name.substr(prefix.size())), forward_as_tuple(context, buffer));
    }

    named_buffers.emplace(std::piecewise_construct, forward_as_tuple("kv_cache"), forward_as_tuple(context, 2 * 2 * context->backlog_size * context->get_model()->header.dim, 2 * 1024));
}
