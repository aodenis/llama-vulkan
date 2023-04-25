#ifndef VULKAN_LLAMA_LLAVA_BUFFER_H
#define VULKAN_LLAMA_LLAVA_BUFFER_H

#include "types.h"
#include "ggml_file.h"

class llava_buffer {
public:
    llava_buffer(llava_context* context, size_t wanted_size, size_t alignment); // Allocate an untyped buffer
    llava_buffer(llava_context* context, ggml_data_descriptor const&); // From a data descriptor

    const u32 size;

private:
    vector<pair<vkr::Buffer, vkr::DeviceMemory>> storages;
    vector<u32> backing_sizes;
};


#endif //VULKAN_LLAMA_LLAVA_BUFFER_H
