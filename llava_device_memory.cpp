#include "llava_device_memory.h"
#include "llava_context.h"
#include <iostream>

llava_device_memory::llava_device_memory(llava_context* _context) : context(_context) {

}

llava_device_memory::~llava_device_memory() {
    assert(buffers.empty());
    if (fallback) {
        assert(fallback->pool_friends.erase(this));
        fallback = nullptr;
    } else if (not pool_friends.empty()) {
        llava_device_memory* new_master = *pool_friends.begin();
        pool_friends.erase(new_master);
        for (llava_device_memory* pool_eq : pool_friends) {
            pool_eq->fallback = new_master;
        }
        new_master->fallback = nullptr;
        new_master->pool_friends = std::move(pool_friends);
        new_master->device_memory = device_memory;
    } else if (device_memory) {
        context->get_device().freeMemory(device_memory);
        device_memory = nullptr;
    }
}

bool llava_device_memory::is_frozen() const {
    return static_cast<bool>(fallback ? fallback->device_memory : device_memory);
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

void llava_device_memory::join_memory_pool(llava_device_memory* new_pool_master) {
    assert(new_pool_master);
    if(new_pool_master > this) {
        return new_pool_master->join_memory_pool(this);
    }
    assert(fallback == nullptr);
    assert(not (new_pool_master->is_frozen() or is_frozen()));
    // TODO assert types are compatible
    fallback = new_pool_master;
    assert(new_pool_master->pool_friends.insert(this).second);
    for (llava_device_memory* pool_friend : pool_friends) {
        new_pool_master->pool_friends.insert(pool_friend);
        pool_friend->fallback = new_pool_master;
    }
    pool_friends.clear();
}

vk::DeviceMemory const &llava_device_memory::get_device_memory() {
    assert(is_frozen());
    return fallback ? fallback->device_memory : device_memory;
}
