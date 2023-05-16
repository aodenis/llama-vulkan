#ifndef VULKAN_LLAMA_LLAVA_DEVICE_MEMORY_H
#define VULKAN_LLAMA_LLAVA_DEVICE_MEMORY_H

#include "types.h"
#include "ggml_file.h"
#include <vulkan/vulkan.hpp>
#include <vector>
#include <set>
#include <atomic>

class llava_device_memory {
public:
    explicit llava_device_memory(llava_context* context);
    ~llava_device_memory();
    [[nodiscard]] bool is_frozen() const;
    void freeze();
    void register_llava_buffer(llava_buffer* buffer);
    void forget_llava_buffer(llava_buffer* buffer);
    size_t register_buffer(size_t alignment, size_t buffer_size);
    llava_context* const context;
    vk::DeviceMemory const& get_device_memory();
    void join_memory_pool(llava_device_memory *new_pool_master);
    void* map();
    void unmap();
    size_t get_size() const;
    bool is_part_of_pool() const;
    llava_device_memory const* get_fallback() const;

public:
    u64 const object_id;

private:
    vk::DeviceMemory device_memory;
    size_t cursor = 0;
    set<llava_buffer*> buffers;
    llava_device_memory* fallback = nullptr;
    set<llava_device_memory*> pool_friends;

private:
    static atomic<u64> object_count;
};

#endif
