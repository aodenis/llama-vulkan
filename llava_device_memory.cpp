#include "llava_device_memory.h"
#include "llava_context.h"
#include <iostream>

llava_device_memory::llava_device_memory(llava_context* _context) : context(_context) {

}

llava_device_memory::~llava_device_memory() {
    assert(buffers.empty());
    if (device_memory) {
        context->get_device().freeMemory(device_memory);
        device_memory = nullptr;
    }
}

bool llava_device_memory::is_frozen() const {
    return static_cast<bool>(device_memory);
}

void llava_device_memory::freeze() {
    assert(not is_frozen());
    device_memory = context->get_device().allocateMemory({cursor, context->mainMemoryTypeIndex});
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
