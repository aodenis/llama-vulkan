#include <iostream>
#include <utility>
#include <cmath>
#include "llava_buffer.h"
#include "llava_device_memory.h"
#include "llava_context.h"
#include "utils.h"

auto wanted_bits = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;

llava_buffer::llava_buffer(llava_context* _context,
                           ggml_value_type _type,
                           u32 shape1,
                           u32 shape2,
                           llava_device_memory* _device_memory) : context(_context),
                                                                  type(_type),
                                                                  shape(shape1, shape2),
                                                                  device_memory_is_shared(_device_memory != nullptr),
                                                                  device_memory(_device_memory ? _device_memory : new llava_device_memory(context)) {
    assert (not device_memory->is_frozen());
    device_memory->register_llava_buffer(this);
    assert((_type == ggml_value_type::f32) or (_type == ggml_value_type::f16));
    push_buffer(matrix_size(type, shape1, shape2));

    if (not device_memory_is_shared) {
        device_memory->freeze();
    }
}


llava_buffer::llava_buffer(llava_context* _context,
                           ggml_data_descriptor const& table,
                           llava_device_memory* _device_memory) : context(_context),
                                                                  backing_buffer_name(table.name),
                                                                  type(table.ftype),
                                                                  shape(table.shape1, table.shape2),
                                                                  device_memory_is_shared(_device_memory != nullptr),
                                                                  device_memory(_device_memory ? _device_memory : new llava_device_memory(context)) {
    assert (not device_memory->is_frozen());
    device_memory->register_llava_buffer(this);

    if (type == ggml_value_type::f32) {
        push_buffer(matrix_size(type, shape.first, shape.second));
    } else if (type == ggml_value_type::q4_0) {
        assert(shape.first % 32 == 0);
        uint32_t block_count = (shape.second * (shape.first >> 5));
        push_buffer(block_count * 4);
        push_buffer(block_count * 16);
    } else {
        assert(false);
    }

    if (not device_memory_is_shared) {
        device_memory->freeze();
    }
}

void llava_buffer::push_buffer(size_t buffer_size) {
    vk::Buffer buffer = context->get_device().createBuffer({{}, buffer_size, wanted_bits});
    auto alignment = context->get_device().getBufferMemoryRequirements(buffer).alignment;
    auto required_size = context->get_device().getBufferMemoryRequirements(buffer).size;
    assert (required_size >= buffer_size);
    size_t new_buffer_offset = device_memory->register_buffer(alignment, required_size);
    buffers.emplace_back(buffer_size, new_buffer_offset, buffer);
}

llava_buffer::~llava_buffer() {
    for(auto& buffer : buffers) {
        context->get_device().destroy(buffer.buffer);
    }

    device_memory->forget_llava_buffer(this);

    buffers.clear();

    if (not device_memory_is_shared) {
        delete device_memory;
    }
}

void llava_buffer::on_memory_freeze() {
    assert (device_memory->is_frozen());
    assert (not buffers_bound);
    buffers_bound = true;

    for (buffer_record_t& buffer : buffers) {
        context->get_device().bindBufferMemory(buffer.buffer, device_memory->device_memory, buffer.offset);
    }

    if (backing_buffer_name.empty()) {
        return;
    }

    auto& table = context->get_model()->get_buffer_descriptor(backing_buffer_name);
    if (type == ggml_value_type::f32) {
        memcpy(context->get_device().mapMemory(device_memory->device_memory, buffers.front().offset, table.size), context->get_model()->mapping + table.offset, (size_t) table.size);
        context->get_device().unmapMemory(device_memory->device_memory);
    } else if (type == ggml_value_type::q4_0) {
        u32 column_count = shape.second;
        uint32_t* raw_data = reinterpret_cast<uint32_t *>(context->get_model()->mapping + table.offset);
        size_t block_count = (shape.first * shape.second) / 32;

        {
            auto* d_data = static_cast<uint32_t *>(context->get_device().mapMemory(device_memory->device_memory, buffers.front().offset, buffers.front().size));
            for (size_t i = 0; i < block_count; ++i) {
                uint32_t row_id = i / (column_count / 32);
                uint32_t column_id = i % (column_count / 32);
                u32 fpid = row_id >> 2;
                u32 fpsid = row_id & 3;
                d_data[fpid * 4 * (column_count / 32) + 4 * column_id + fpsid] = raw_data[5 * i];
            }
            context->get_device().unmapMemory(device_memory->device_memory);
        }

        {
            auto* q_data = static_cast<uint32_t *>(context->get_device().mapMemory(device_memory->device_memory, buffers.back().offset, buffers.back().size));
            for (size_t i = 0; i < block_count; ++i) {
                uint32_t row_id = i / (column_count / 32);
                uint32_t column_id = i % (column_count / 32);
                u32 fpid = row_id >> 2;
                u32 fpsid = row_id & 3;

                for(u32 sub_block_id = 0; sub_block_id < 4; sub_block_id ++) {
                    q_data[fpid * 4 * (column_count / 32) * 4 + 4 * 4 * column_id + 4 * sub_block_id + fpsid] = raw_data[5 * i + 1 + sub_block_id];
                }
            }
            context->get_device().unmapMemory(device_memory->device_memory);
        }
    } else {
        assert(false);
    }
}

bool llava_buffer::is_allocated() const {
    return buffers_bound;
}

vector<buffer_record_t> const &llava_buffer::get_sub_buffers() const {
    return buffers;
}

void llava_buffer::write_full(const void *in_buf, ggml_value_type input_type) const {
    assert(is_allocated());
    write_f32(in_buf, input_type, 0, shape.first * shape.second);
}

void llava_buffer::write_f32(const void *in_buf, ggml_value_type input_type, u32 f32_offset, u32 f32_count) const {
    assert (is_allocated());
    assert (type == ggml_value_type::f32);
    if (f32_count == 0) {
        return;
    }
    assert (f32_offset < shape.first * shape.second);
    assert ((f32_offset + f32_count) <= (shape.first * shape.second));
    if (input_type == ggml_value_type::q4_0) {
        assert((f32_count % 32) == 0);
        auto* data = (float*)(context->get_device().mapMemory(device_memory->device_memory, buffers.front().offset + 4 * f32_offset, 4 * f32_count));

        for (u32 i = 0; i < f32_count / 32; i++) {
            float d = ((float const*)in_buf)[5 * i];
            u8 const* qbase = ((u8 const*)in_buf) + (20 * i + 4);
            for (u32 j = 0; j < 16; j++) {
                int q = (int)(qbase[j]);
                data[i * 32 + 2 * j] = ((float)((q & 0xf) - 8)) * d;
                data[i * 32 + 2 * j + 1] = ((float)((q >> 4) - 8)) * d;
            }
        }

        context->get_device().unmapMemory(device_memory->device_memory);
        return;
    } else if (input_type == ggml_value_type::f32) {
        void* data = context->get_device().mapMemory(device_memory->device_memory, buffers.front().offset + 4 * f32_offset, 4 * f32_count);
        memcpy(data, in_buf, 4 * f32_count);
        context->get_device().unmapMemory(device_memory->device_memory);
        return;
    }
}

/*
void llava_buffer::hexdump(size_t n, size_t offset) const {
    assert(is_allocated());
    size_t effective_sz = 0;
    if (offset <= buffer_size) {
        effective_sz = min(n, buffer_size - offset);
    }
    cout << name << "\n";
    if (offset >= buffer_size) {
        cout << "...\n";
    } else {
        u8* buffer_memory = (u8*)context->get_device().mapMemory(deviceMemory, offset, effective_sz);
        for (u32 j = 0; j < effective_sz; j+= 16) {
            cout << hex << offset + j << ": ";
            for (u32 k = 0; k < 16; k++) {
                cout << hex << (u32) (buffer_memory[k+j]) << " ";
                if (k == 7) {
                    cout << " ";
                }
            }
            cout << "\n";
        }
        context->get_device().unmapMemory(deviceMemory);
    }
    cout << flush;
}

void llava_buffer::f32_dump(size_t n, size_t offset, bool in_line) const {
    assert(is_allocated());
    auto* buffer_memory = (float*)context->get_device().mapMemory(deviceMemory, 0, buffer_size);
    for (size_t j = 0; j < n; j++) {
        cout << buffer_memory[offset + j] << (in_line ? " " : "\n");
    }
    if (in_line) {
        cout << endl;
    }
    context->get_device().unmapMemory(deviceMemory);
}



 void llava_buffer::fill_f32(float value) const {
    assert(is_allocated() and type == ggml_value_type::f32);
    auto* z = static_cast<float *>(context->get_device().mapMemory(deviceMemory, 0, -1));
    for(u32 i = 0; i < shape.first * shape.second; ++i) {
        z[i] = value;
    }
    context->get_device().unmapMemory(deviceMemory);
}

bool llava_buffer::contains_nan() const {
    assert(is_allocated() and type == ggml_value_type::f32);
    auto* z = static_cast<float *>(context->get_device().mapMemory(deviceMemory, 0, -1));
    for(size_t i = 0; i < shape.first * shape.second; ++i) {
        if(isnan(z[i]))
            return true;
    }
    context->get_device().unmapMemory(deviceMemory);
    return false;
} */

buffer_record_t::buffer_record_t(size_t _size, size_t _offset, vk::Buffer _buffer) : size(_size), offset(_offset), buffer(_buffer) {

}
