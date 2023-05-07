#ifndef VULKAN_LLAMA_LLAVA_BUFFER_H
#define VULKAN_LLAMA_LLAVA_BUFFER_H

#include "types.h"
#include "ggml_file.h"
#include <vulkan/vulkan.hpp>
#include <vector>

class llava_buffer {
public:
    llava_buffer(llava_context* context, string name, ggml_value_type type, u32 shape1, u32 shape2 = 1, u32 alignment = 1024);
    llava_buffer(llava_context* context, ggml_data_descriptor const&); // From a data descriptor
    llava_buffer(llava_buffer const&) = delete;
    llava_buffer(llava_buffer&&) = delete;
    ~llava_buffer();

    void hexdump(size_t n = 256, size_t offset = 0) const;
    void f32_dump(size_t n = 32, size_t offset = 0, bool in_line = true) const;
    void q40_dump(size_t n = 256, bool in_line = false) const;
    void write_full(void const* in_buf, ggml_value_type intype) const;
    void allocate();
    bool is_allocated() const;
    void fill_f32(float value) const;
    vector<vk::Buffer> buffers;
    vk::DeviceMemory deviceMemory = nullptr;

    bool contains_nan() const;
public:
    llava_context* const context;
    const size_t size;
    const string name;
    const string backing_name;
    const ggml_value_type type;
    const pair<u32, u32> shape;

private:
    size_t buffer_size;
    u32 memory_type;
};

#endif
