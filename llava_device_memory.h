#ifndef VULKAN_LLAMA_LLAVA_DEVICE_MEMORY_H
#define VULKAN_LLAMA_LLAVA_DEVICE_MEMORY_H

#include "types.h"
#include "ggml_file.h"
#include <vulkan/vulkan.hpp>
#include <vector>
#include <set>

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
    vk::DeviceMemory device_memory;

private:
    size_t cursor = 0;
    set<llava_buffer*> buffers;
};

#endif
