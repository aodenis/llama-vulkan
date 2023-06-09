#ifndef VULKAN_LLAMA_LLAVA_BUFFER_H
#define VULKAN_LLAMA_LLAVA_BUFFER_H

#include "types.h"
#include "ggml_file.h"
#include <vulkan/vulkan.hpp>
#include <vector>

struct buffer_record_t {
    buffer_record_t(size_t size, size_t offset, vk::Buffer buffer);
    const size_t size;
    const size_t offset;
    const vk::Buffer buffer;
};

class llava_buffer {
public:
    llava_buffer(llava_context* context, ggml_value_type type, u32 shape1, u32 shape2 = 1, llava_device_memory* device_memory = nullptr);
    llava_buffer(llava_context* context, ggml_data_descriptor const&, llava_device_memory* device_memory = nullptr); // From a data descriptor
    llava_buffer(llava_buffer const&) = delete;
    llava_buffer(llava_buffer&&) = delete;
    ~llava_buffer();

    [[nodiscard]] vector<buffer_record_t> const& get_sub_buffers() const;

    void hexdump(size_t n = 256, size_t offset = 0) const;
    void f32_dump(size_t n = 32, size_t offset = 0, bool in_line = true) const;
    void q40_dump(size_t n = 256, bool in_line = false) const;
    void write_full(void const* in_buf, ggml_value_type intype) const;
    bool is_allocated() const;
    void fill_f32(float value) const;
    bool contains_nan() const;
    void on_memory_freeze();

public:
    llava_context* const context;
    const string backing_buffer_name;
    const ggml_value_type type;
    const pair<u32, u32> shape;
    const bool device_memory_is_shared;
    llava_device_memory* const device_memory;

private:
    void push_buffer(size_t buffer_size);
    vector<buffer_record_t> buffers;
    bool buffers_bound = false;
};

#endif
