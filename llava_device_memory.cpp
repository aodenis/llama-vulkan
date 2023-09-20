#include "llava_device_memory.h"
#include "llava_context.h"
#include <iostream>
#include <vulkan/vulkan.hpp>

llava_device_memory::llava_device_memory(llava_context* _context) : context(_context), device_memory(nullptr) {

}

llava_device_memory::~llava_device_memory() {
    assert(buffers.empty());
    if (device_memory) {
        context->get_device().freeMemory(*device_memory);
        delete device_memory;
        device_memory = nullptr;
    }
}

bool llava_device_memory::is_frozen() const {
    return device_memory != nullptr;
}

void llava_device_memory::freeze() {
    assert(not is_frozen());
    device_memory = new vk::DeviceMemory();
    *device_memory = context->get_device().allocateMemory({cursor, context->mainMemoryTypeIndex});
    for (llava_buffer* buffer : buffers) {
        buffer->on_memory_freeze();
    }
}

void llava_device_memory::register_llava_buffer(llava_buffer* buffer) {
    assert(not is_frozen());
    if(not buffers.emplace(buffer).second) {
        cerr << "[?] A buffer was registered twice to a device memory" << endl;
    }
}

void llava_device_memory::forget_llava_buffer(llava_buffer* buffer) {
    buffers.erase(buffer);
}

size_t llava_device_memory::register_buffer(size_t alignment, size_t buffer_size) {
    assert(not is_frozen());
    if (alignment == 0) {
        alignment = 1;
    }
    cursor = cursor + (alignment - 1);
    cursor -= (cursor % alignment);
    size_t ret_cursor = cursor;
    cursor += buffer_size;
    return ret_cursor;
}

vk::DeviceMemory const *llava_device_memory::get_device_memory() const {
    assert(is_frozen());
    return device_memory;
}

void* llava_device_memory::map() const {
    // TODO some checks
    return context->get_device().mapMemory(*get_device_memory(), 0, cursor);
}

void* llava_device_memory::map(size_t offset, size_t size) const {
    // TODO some checks
    return context->get_device().mapMemory(*get_device_memory(), offset, size);
}

void llava_device_memory::unmap() const {
    // TODO some checks
    context->get_device().unmapMemory(*get_device_memory());
}

size_t llava_device_memory::get_size() const {
    assert(is_frozen());
    return cursor;
}

void llava_device_memory::dump_buffers(int out_fd, const string &name) {
    assert(this->is_frozen());
    for (llava_buffer* buffer : buffers) {
        string out_name = name + ".";
        const string& pretty_name = buffer->get_pretty_name();
        if (pretty_name.empty()) {
            stringstream ss;
            ss << ((void*)buffer);
            out_name += ss.str();
        } else {
            if (pretty_name == "k_cache" or pretty_name == "v_cache") {
                continue;
            }
            out_name += pretty_name;
        }
        buffer->dump_raw(out_fd, out_name);
    }
}
