#include <vector>
#include <vulkan/vulkan.hpp>
#include <iostream>
#include "llava_buffer.h"
#include "llava_device_memory.h"
#include "llava_context.h"
#include "utils.h"

struct [[maybe_unused]] q4_0_block {
    float base;
    u8 qs[16];
} __attribute__((__packed__));

struct [[maybe_unused]] q4_0_block_f16 {
    u16 base;
    u8 qs[16];
} __attribute__((__packed__));

struct [[maybe_unused]] q8_0_block {
    float base;
    char qs[32];
} __attribute__((__packed__));

struct [[maybe_unused]] q8_0_block_f16 {
    u16 base;
    char qs[32];
};

auto wanted_bits = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;

llava_buffer::llava_buffer(llava_context *_context,
                           ggml_value_type _type,
                           u32 shape1,
                           u32 shape2,
                           llava_device_memory *_device_memory,
                           string _name) : context(_context),
                                           type(_type),
                                           shape(shape1, shape2),
                                           device_memory_is_shared(_device_memory != nullptr),
                                           device_memory(_device_memory ? _device_memory : new llava_device_memory(context)),
                                           pretty_name(std::move(_name)) {
    assert (not device_memory->is_frozen());
    device_memory->register_llava_buffer(this);
    assert((_type == ggml_value_type::f32) or (_type == ggml_value_type::f16));
    push_buffer(shape1 * shape2 * (type == ggml_value_type::f32 ? 4 : 2));

    if (not device_memory_is_shared) {
        device_memory->freeze();
    }
}

llava_buffer::llava_buffer(llava_context *_context,
                           ggml_data_descriptor const &table,
                           llava_device_memory *_device_memory) : context(_context),
                                                                  backing_buffer_name(table.name),
                                                                  type(table.ftype),
                                                                  shape(table.shape1, table.shape2),
                                                                  device_memory_is_shared(_device_memory != nullptr),
                                                                  device_memory(_device_memory ? _device_memory : new llava_device_memory(context)) {
    assert (not device_memory->is_frozen());
    device_memory->register_llava_buffer(this);

    if (type == ggml_value_type::f32) {
        push_buffer(shape.first * shape.second * 4);
    } else if (type == ggml_value_type::q4_0) {
        if (table.model_version == 1) {
            assert(shape.first % 32 == 0);
            u32 block_count = (shape.second * (shape.first >> 5));
            assert(table.size == 20 * block_count);
            push_buffer(block_count * 4);
            push_buffer(block_count * 16);
        } else if (table.model_version == 3) {
            assert(shape.first % 32 == 0);
            u32 block_count = (shape.second * (shape.first >> 5));
            assert(table.size == 18 * block_count);
            push_buffer(block_count * 2);
            push_buffer(block_count * 16);
        } else {
            assert(false);
        }
    } else if (type == ggml_value_type::q8_0) {
        if (table.model_version == 1) {
            assert(shape.first % 32 == 0);
            u32 block_count = (shape.second * (shape.first >> 5));
            assert(table.size == 36 * block_count);
            push_buffer(block_count * 4);
            push_buffer(block_count * 32);
        } else if (table.model_version == 3) {
            assert(shape.first % 32 == 0);
            u32 block_count = (shape.second * (shape.first >> 5));
            assert(table.size == 34 * block_count);
            push_buffer(block_count * 2);
            push_buffer(block_count * 32);
        } else {
            assert(false);
        }
    } else {
        assert(false);
    }

    if (not device_memory_is_shared) {
        device_memory->freeze();
    }
}

void llava_buffer::push_buffer(size_t buffer_size) {
    auto* buffer = new vk::Buffer;
    vk::Buffer _buffer = context->get_device().createBuffer({{}, buffer_size, wanted_bits});
    buffer->operator=(_buffer);
    auto alignment = context->get_device().getBufferMemoryRequirements(*buffer).alignment;
    auto required_size = context->get_device().getBufferMemoryRequirements(*buffer).size;
    assert (required_size >= buffer_size);
    size_t new_buffer_offset = device_memory->register_buffer(alignment, required_size);
    buffers.emplace_back(buffer_size, new_buffer_offset, buffer);
}

llava_buffer::~llava_buffer() {
    for (auto &buffer: buffers) {
        context->get_device().destroy(*(buffer.buffer));
        delete buffer.buffer;
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

    for (buffer_record_t &buffer: buffers) {
        context->get_device().bindBufferMemory(*(buffer.buffer), *(device_memory->get_device_memory()), buffer.offset);
    }
}

float halfToFloat(const uint16_t half) {
    uint32_t single;
    int s = (half >> 15) & 0x00000001;
    int e = (half >> 10) & 0x0000001f;
    int m = half & 0x000003ff;

    if (e == 0) {
        if (m == 0) {
            single = (s << 31);
        } else {
            while (!(m & 0x00000400)) {
                m <<= 1;
                e -= 1;
            }

            e += 1;
            m &= ~0x00000400;
            e = e + (127 - 15);
            m = m << 13;

            single = ((s << 31) | (e << 23) | m);
        }
    } else if (e == 31) {
        if (m == 0) {
            single = ((s << 31) | 0x7f800000);
        } else {
            single = ((s << 31) | 0x7f800000 | (m << 13));
        }
    } else {
        e = e + (127 - 15);
        m = m << 13;
        single = ((s << 31) | (e << 23) | m);
    }

    return *reinterpret_cast<float *>(&single);
}

void llava_buffer::load_from_disk(void *_target_buffer) {
    u8 *target_buffer = (u8 *) _target_buffer;
    if (backing_buffer_name.empty()) {
        return;
    }
    assert(is_allocated());

    auto &table = context->get_model()->get_buffer_descriptor(backing_buffer_name);
    if (type == ggml_value_type::f32) {
        memcpy(target_buffer + buffers.at(0).offset, context->get_model()->mapping + table.offset, (size_t) table.size);
        return;
    }

    auto model_version = context->get_model()->header.file_version;
    if ((type == ggml_value_type::q4_0) and (model_version == 1)) {
        u32 column_count = shape.second;
        auto *raw_data = reinterpret_cast<q4_0_block *>(context->get_model()->mapping + table.offset);
        size_t block_count = (shape.first * shape.second) / 32;

        {
            // TODO not perf
            auto *d_data = (float *) (target_buffer + buffers.at(0).offset);
            for (size_t i = 0; i < block_count; ++i) {
                u32 row_id = i / (column_count / 32);
                u32 column_id = i % (column_count / 32);
                u32 fpid = row_id >> 2;
                u32 fpsid = row_id & 3;
                d_data[fpid * 4 * (column_count / 32) + 4 * column_id + fpsid] = raw_data[i].base;
            }
        }

        {
            auto *q_data = (u32 *) (target_buffer + buffers.at(1).offset);
            for (size_t i = 0; i < block_count; ++i) {
                u32 row_id = i / (column_count / 32);
                u32 column_id = i % (column_count / 32);
                u32 fpid = row_id >> 2;
                u32 fpsid = row_id & 3;

                for (u32 sub_block_id = 0; sub_block_id < 4; sub_block_id++) {
                    memcpy(q_data + (fpid * 4 * (column_count / 32) * 4 + 4 * 4 * column_id + 4 * sub_block_id + fpsid), raw_data[i].qs + 4 * sub_block_id, 4);
                }
            }
        }
        return;
    }

    if ((type == ggml_value_type::q4_0) and (model_version == 3)) {
        u32 column_count = shape.second;
        u32 column_count_d32 = column_count / 32;
        auto *raw_data = reinterpret_cast<q4_0_block_f16 *>(context->get_model()->mapping + table.offset);
        size_t block_count = (shape.first * shape.second) / 32;

        auto *d_data = (u8 *) (target_buffer + buffers.at(0).offset);
        auto *q_data = (u8 *) (target_buffer + buffers.at(1).offset);

        u8 buffer[64];
        auto *d_data_buf = new float[4 * column_count_d32];
        auto *d16_data_buf = new u16[4 * column_count_d32];
        for (u32 fpid = 0; fpid < (block_count / column_count_d32) >> 2; fpid++) {
            for (u32 column_id = 0; column_id < column_count_d32; ++column_id) {
                for (u32 fpsid = 0; fpsid < 4; fpsid++) {
                    u32 row_id = (fpid << 2) | fpsid;
                    u32 i = row_id * column_count_d32 + column_id;

                    d16_data_buf[4 * column_id + fpsid] = raw_data[i].base;

                    u8 const *sub_block_in = raw_data[i].qs;

                    for (u32 sub_block_id = 0; sub_block_id < 4; sub_block_id++) {
                        buffer[fpsid * 4 + sub_block_id] = (sub_block_in[sub_block_id * 2] & 0xf) | (sub_block_in[sub_block_id * 2 + 1] << 4);
                    }

                    for (u32 sub_block_id = 0; sub_block_id < 4; sub_block_id++) {
                        buffer[(4 + fpsid) * 4 + sub_block_id] = (sub_block_in[sub_block_id * 2 + 8] & 0xf) | (sub_block_in[sub_block_id * 2 + 8 + 1] << 4);
                    }

                    for (u32 sub_block_id = 0; sub_block_id < 4; sub_block_id++) {
                        buffer[(8 + fpsid) * 4 + sub_block_id] = (sub_block_in[sub_block_id * 2] >> 4) | (sub_block_in[sub_block_id * 2 + 1] & 0xf0);
                    }

                    for (u32 sub_block_id = 0; sub_block_id < 4; sub_block_id++) {
                        buffer[(12 + fpsid) * 4 + sub_block_id] = (sub_block_in[sub_block_id * 2 + 8] >> 4) | (sub_block_in[sub_block_id * 2 + 8 + 1] & 0xf0);
                    }
                }
                memcpy(q_data + (fpid * column_count_d32 + column_id) * 64, buffer, 64);
            }

            if (weight_buffer_is_f16()) {
                memcpy(d_data + fpid * 4 * column_count_d32 * sizeof(u16), d16_data_buf, 4 * column_count_d32 * sizeof(u16));
            } else {
                for (u32 i = 0; i < 4 * column_count_d32; ++i) {
                    d_data_buf[i] = halfToFloat(d16_data_buf[i]);
                }
                memcpy(d_data + fpid * 4 * column_count_d32 * sizeof(float), d_data_buf, 4 * column_count_d32 * sizeof(float));
            }
        }
        delete[] d_data_buf;
        delete[] d16_data_buf;
        return;
    }

    if ((type == ggml_value_type::q8_0) and (model_version == 3)) {
        u32 column_count = shape.second;
        u32 column_count_d32 = column_count / 32;
        auto *data = reinterpret_cast<q8_0_block_f16 *>(context->get_model()->mapping + table.offset);
        size_t block_count = (shape.first * shape.second) / 32;

        auto *d_data = (u8 *) (target_buffer + buffers.at(0).offset);
        auto *q_data = (u8 *) (target_buffer + buffers.at(1).offset);

        u32 buffer[32];
        auto *d_data_buf = new float[4 * column_count_d32];
        auto *d16_data_buf = new u16[4 * column_count_d32];
        for (u32 fpid = 0; fpid < (block_count / column_count_d32) >> 2; fpid++) {
            for (u32 column_id = 0; column_id < column_count_d32; ++column_id) {
                for (u32 fpsid = 0; fpsid < 4; fpsid++) {
                    u32 row_id = (fpid << 2) | fpsid;
                    u32 i = row_id * column_count_d32 + column_id;

                    d16_data_buf[4 * column_id + fpsid] = data[i].base;

                    for (u32 sub_block_id = 0; sub_block_id < 8; sub_block_id++) {
                        memcpy(buffer + sub_block_id * 4 + fpsid, data[i].qs + 4 * sub_block_id, 4);
                    }
                }
                for (u32& x : buffer) {
                    x ^= 0x80808080U;
                    // x = __builtin_bswap32(x);
                }
                memcpy(q_data + ((fpid * column_count_d32 + column_id) * 128), buffer, 128);
            }
            if (weight_buffer_is_f16()) {
                memcpy(d_data + fpid * 4 * column_count_d32 * sizeof(u16), d16_data_buf, 4 * column_count_d32 * sizeof(u16));
            } else {
                for (u32 i = 0; i < 4 * column_count_d32; ++i) {
                    d_data_buf[i] = halfToFloat(d16_data_buf[i]);
                }
                memcpy(d_data + fpid * 4 * column_count_d32 * sizeof(float), d_data_buf, 4 * column_count_d32 * sizeof(float));
            }
        }
        delete[] d_data_buf;
        delete[] d16_data_buf;
        return;
    }
    assert(false);
}

bool llava_buffer::is_allocated() const {
    return buffers_bound;
}

vector<buffer_record_t> const &llava_buffer::get_sub_buffers() const {
    return buffers;
}

void llava_buffer::write_full(const void *in_buf, ggml_value_type input_type, u32 model_version) const {
    write_f32(in_buf, input_type, model_version, 0, shape.first * shape.second);
}

void llava_buffer::write_f32(const void *in_buf, ggml_value_type input_type, u32 model_version, u32 f32_offset, u32 f32_count) const {
    assert (is_allocated());
    assert (type == ggml_value_type::f32);
    if (f32_count == 0) {
        return;
    }
    assert (f32_offset < shape.first * shape.second);
    assert ((f32_offset + f32_count) <= (shape.first * shape.second));
    if (input_type == ggml_value_type::q4_0) {
        assert((f32_count % 32) == 0);
        auto *data = (float *) (map(0, 4 * f32_offset, 4 * f32_count));
        if (model_version == 1) {
            for (u32 i = 0; i < f32_count / 32; i++) {
                float d = ((float const *) in_buf)[5 * i];
                u8 const *qbase = ((u8 const *) in_buf) + (20 * i + 4);
                for (u32 j = 0; j < 16; j++) {
                    int q = (int) (qbase[j]);
                    data[i * 32 + 2 * j] = ((float) ((q & 0xf) - 8)) * d;
                    data[i * 32 + 2 * j + 1] = ((float) ((q >> 4) - 8)) * d;
                }
            }
        } else if (model_version == 3) {
            for (u32 i = 0; i < f32_count / 32; i++) {
                u16 dbase = ((u16 const *) in_buf)[9 * i];
                float d = halfToFloat(dbase);
                u8 const *qbase = ((u8 const *) in_buf) + (18 * i + 2);
                for (u32 j = 0; j < 16; j++) {
                    int q = (int) (qbase[j]);
                    data[i * 32 + j] = ((float) ((q & 0xf) - 8)) * d;
                    data[i * 32 + j + 16] = ((float) ((q >> 4) - 8)) * d;
                }
            }
        } else {
            assert(false);
        }

        unmap();
        return;
    } else if (input_type == ggml_value_type::f32) {
        void *data = map(0, 4 * f32_offset, 4 * f32_count);
        memcpy(data, in_buf, 4 * f32_count);
        unmap();
        return;
    } else if (input_type == ggml_value_type::q8_0) {
        assert((f32_count % 32) == 0);
        auto *data = (float *) (map(0, 4 * f32_offset, 4 * f32_count));
        auto* blocks = (q8_0_block_f16 *) in_buf;
        assert (model_version == 3);
        for (u32 i = 0; i < f32_count / 32; i++) {
            float d = halfToFloat(blocks[i].base);
            for (u32 j = 0; j < 32; j++) {
                int q = (int) ((u32)(blocks[i].qs[j]));
                if (q >= 128) q -= 256;
                data[i * 32 + j] = ((float) (q)) * d;
            }
        }

        unmap();
        return;
    } else {
        assert(false);
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

buffer_record_t::buffer_record_t(size_t _size, size_t _offset, vk::Buffer* _buffer) : size(_size), offset(_offset), buffer(_buffer) {

}

void *llava_buffer::map(u32 index, u32 offset, u32 size) const {
    auto const &buffer = buffers.at(index);
    assert (offset <= buffer.size);
    if (size > buffer.size - offset) {
        size = buffer.size - offset;
    }
    return device_memory->map(buffer.offset + offset, size);
}

void llava_buffer::unmap() const {
    return device_memory->unmap();
}

void llava_buffer::load_to_gpu() {
    u64 min_offset = ~0UL;
    u64 max_offset = 0;
    for (auto &buffer: buffers) {
        if (buffer.offset < min_offset) {
            min_offset = buffer.offset;
        }
        if (buffer.offset + buffer.size > max_offset) {
            max_offset = buffer.offset + buffer.size;
        }
    }
    u8 *mapping = (u8 *) (device_memory->map(min_offset, max_offset - min_offset));
    load_from_disk(mapping - min_offset); // TODO horrible
    return unmap();
}

void llava_buffer::dump_raw(int out_fd, const string &name) {
    assert(buffers.size() == 1);
    assert(type == ggml_value_type::f32);
    uint32_t header_size = (sizeof(uint32_t) * 3 + name.size() + 1 + 31) & (~31U);
    vector<u8> header(header_size);
    memset(header.data(), 0, header_size);
    memcpy(header.data(), &shape.first, sizeof(shape.first));
    memcpy(header.data() + 4, &shape.second, sizeof(shape.second));
    u32 _type = (u32) type;
    memcpy(header.data() + 8, &_type, sizeof(_type));
    memcpy(header.data() + 12, name.data(), name.size());
    u32 j = 0;
    while (j < header_size) {
        ssize_t d = write(out_fd, header.data() + j, header_size - j);
        assert(d > 0);
        j += d;
    }
    void *buf = map(0, 0, shape.first * shape.second * 4);
    j = 0;
    while (j < shape.first * shape.second * 4) {
        ssize_t d = write(out_fd, ((u8 *) buf) + j, shape.first * shape.second * 4 - j);
        assert(d > 0);
        j += d;
    }
    unmap();
}

const string &llava_buffer::get_pretty_name() const {
    return pretty_name;
}

bool llava_buffer::weight_buffer_is_f16() const {
    assert(buffers.size() == 2); //Only for matrices

    u64 byte_per_element = buffers.front().size / ((((size_t)shape.first) * ((size_t)shape.second)) >> 5);
    assert (byte_per_element == 2 or byte_per_element == 4);
    return byte_per_element == 2;
}
