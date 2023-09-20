#ifndef VULKAN_LLAMA_LLAVA_BUFFER_H
#define VULKAN_LLAMA_LLAVA_BUFFER_H

#include "types.h"
#include "ggml_file.h"
#include <vector>

struct buffer_record_t {
    buffer_record_t(size_t size, size_t offset, vk::Buffer* buffer);
    const size_t size;
    const size_t offset;
    vk::Buffer* buffer;
};

class llava_buffer {
public:
    llava_buffer(llava_context* context, ggml_value_type type, u32 shape1, u32 shape2 = 1, llava_device_memory* device_memory = nullptr, string name = ""); // Anonymous RW buffer
    llava_buffer(llava_context* context, ggml_data_descriptor const&, llava_device_memory* device_memory = nullptr); // From a data descriptor
    llava_buffer(llava_buffer const&) = delete;
    llava_buffer(llava_buffer&) = delete;
    llava_buffer(llava_buffer&&) = delete;
    llava_buffer& operator=(llava_buffer const&) = delete;
    llava_buffer& operator=(llava_buffer&) = delete;
    llava_buffer& operator=(llava_buffer&&) = delete;
    ~llava_buffer();

    [[nodiscard]] vector<buffer_record_t> const& get_sub_buffers() const;

    void hexdump(size_t n = 256, size_t offset = 0) const;
    void f32_dump(size_t n = 32, size_t offset = 0, bool in_line = true) const;
    void q40_dump(size_t n = 256, bool in_line = false) const;
    void write_full(void const* in_buf, ggml_value_type intype, u32 model_version) const;
    void write_f32(void const* in_buf, ggml_value_type intype, u32 model_version, u32 f32_offset, u32 f32_count) const;
    [[nodiscard]] bool is_allocated() const;
    void fill_f32(float value) const;
    bool contains_nan() const;
    void on_memory_freeze();
    [[nodiscard]] void* map(u32 index = 0, u32 offset = 0, u32 size = ~0U) const;
    void unmap() const;
    void load_from_disk(void* target_buffer);
    void load_to_gpu();
    void dump_raw(int out_fd, const string &name);
    [[nodiscard]] const string &get_pretty_name() const;

    bool weight_buffer_is_f16() const;
public:
    llava_context* const context;
    const string backing_buffer_name;
    const string pretty_name;
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
