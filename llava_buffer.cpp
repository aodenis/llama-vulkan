#include <iostream>
#include <utility>
#include <cmath>
#include "llava_buffer.h"
#include "llava_context.h"
#include "utils.h"

bool ends_with(const string& a, const string& b) {
    if (a.size() < b.size())
        return false;
    return memcmp(a.data() + a.size() - b.size(), b.data(), b.size()) == 0;
}

llava_buffer::llava_buffer(llava_context* _context,
                           string _name,
                           ggml_value_type _type,
                           u32 shape1,
                           u32 shape2,
                           u32 alignment) : context(_context),
                                            name(std::move(_name)),
                                            size(matrix_size(_type, shape1, shape2)),
                                            type(_type),
                                            shape(shape1, shape2),
                                            memory_type(context->mainMemoryTypeIndex) {
    auto wantedBits = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;
    assert((_type == ggml_value_type::f32) or (_type == ggml_value_type::f16));
    buffer_size = matrix_overflow_size(_type, shape1, shape2, alignment);

    vk::Buffer buffer = context->get_device().createBuffer({{}, buffer_size, wantedBits});
    buffers.emplace_back(buffer);

    if (context->allocate_buffers) {
        allocate();
    }
}

llava_buffer::llava_buffer(llava_context* _context, ggml_data_descriptor const& table) : context(_context),
                                                                                         name(table.name),
                                                                                         backing_name(table.name),
                                                                                         size(table.size),
                                                                                         type(table.ftype),
                                                                                         shape(table.shape1, table.shape2),
                                                                                         memory_type(context->mainMemoryTypeIndex) {
    auto wantedBits = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;
    if (type == ggml_value_type::q4_0) {
        assert(shape.first % 32 == 0);
        uint32_t element_count = shape.second * shape.first;
        uint32_t block_count = element_count / 32;

        buffer_size = block_count * 20;

        vk::Buffer DBuffer = context->get_device().createBuffer({{}, (buffer_size/20)*4, wantedBits});
        vk::Buffer QBuffer = context->get_device().createBuffer({{}, (buffer_size/20)*16, wantedBits});
        buffers.emplace_back(DBuffer);
        buffers.emplace_back(QBuffer);
    } else if (type == ggml_value_type::f32) {
        uint32_t element_count = shape.second * shape.first;
        element_count = (element_count + 1023) & ~1023U;

        buffer_size = element_count * 4;
        vk::Buffer buffer = context->get_device().createBuffer({{}, buffer_size, wantedBits});
        buffers.emplace_back(buffer);
    } else {
        assert(false);
    }
    if (context->allocate_buffers) {
        allocate();
    }
}

llava_buffer::~llava_buffer() {
    for(auto& buffer : buffers) {
        context->get_device().destroy(buffer);
    }
    buffers.clear();
    context->get_device().freeMemory(deviceMemory);
}

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

void llava_buffer::write_full(const void *in_buf, ggml_value_type intype) const {
    assert(is_allocated());
    if (type == ggml_value_type::f32) {
        if (intype == ggml_value_type::q4_0) {
            assert(shape.second == 1);
            assert(shape.first % 32 == 0);
            auto* data = (float*)(context->get_device().mapMemory(deviceMemory, 0, 4 * shape.first));

            for (u32 i = 0; i < shape.first / 32; i++) {
                float d = ((float const*)in_buf)[5 * i];
                u8 const* qbase = ((u8 const*)in_buf) + (20 * i + 4);
                for (u32 j = 0; j < 16; j++) {
                    int q = (int)(qbase[j]);
                    data[i * 32 + 2 * j] = ((float)((q & 0xf) - 8)) * d;
                    data[i * 32 + 2 * j + 1] = ((float)((q >> 4) - 8)) * d;
                }
            }

            context->get_device().unmapMemory(deviceMemory);
            return;
        } else if (intype == ggml_value_type::f32) {
            void* data = context->get_device().mapMemory(deviceMemory, 0, 4 * shape.second * shape.first);
            memcpy(data, in_buf, 4 * shape.second * shape.first);
            context->get_device().unmapMemory(deviceMemory);
            return;
        }
    }
    assert(false);
}

void llava_buffer::allocate() {
    assert (context->allocate_buffers);
    assert (not is_allocated());
    if (backing_name.empty()) {
        vk::MemoryRequirements requirements = context->get_device().getBufferMemoryRequirements(buffers.front());
        deviceMemory = context->get_device().allocateMemory({requirements.size, memory_type});
        context->get_device().bindBufferMemory(buffers.front(), deviceMemory, 0);
        return;
    }

    auto& table = context->get_model()->get_buffer_descriptor(backing_name);
    if (type == ggml_value_type::q4_0) {
        deviceMemory = context->get_device().allocateMemory({buffer_size, memory_type});
        context->get_device().bindBufferMemory(buffers.front(), deviceMemory, 0);
        context->get_device().bindBufferMemory(buffers.back(), deviceMemory, buffer_size/5);

        u32 column_count = shape.second;

        auto* d_data = static_cast<uint32_t *>(context->get_device().mapMemory(deviceMemory, 0, table.size));
        uint32_t* q_data = d_data + (buffer_size/20);
        uint32_t* raw_data = reinterpret_cast<uint32_t *>(context->get_model()->mapping + table.offset);
        size_t block_count = buffer_size/20;
        for (size_t i = 0; i < block_count; ++i) {
            uint32_t row_id = i / (column_count / 32);
            uint32_t column_id = i % (column_count / 32);
            u32 fpid = row_id >> 2;
            u32 fpsid = row_id & 3;
            d_data[fpid * 4 * (column_count / 32) + 4 * column_id + fpsid] = raw_data[5 * i];

            for(u32 sub_block_id = 0; sub_block_id < 4; sub_block_id ++) {
                q_data[fpid * 4 * (column_count / 32) * 4 + 4 * 4 * column_id + 4 * sub_block_id + fpsid] = raw_data[5 * i + 1 + sub_block_id];
            }
        }
        context->get_device().unmapMemory(deviceMemory);
    } else if (type == ggml_value_type::f32) {
        vk::MemoryRequirements requirements = context->get_device().getBufferMemoryRequirements(buffers.front());
        deviceMemory = context->get_device().allocateMemory({requirements.size, memory_type});
        context->get_device().bindBufferMemory(buffers.front(), deviceMemory, 0);

        memcpy(context->get_device().mapMemory(deviceMemory, 0, table.size), context->get_model()->mapping + table.offset, (size_t)table.size);
        context->get_device().unmapMemory(deviceMemory);
    } else {
        assert(false);
    }
}

bool llava_buffer::is_allocated() const {
    return static_cast<bool>(deviceMemory);
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
}
