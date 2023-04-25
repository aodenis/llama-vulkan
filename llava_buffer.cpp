#include <iostream>
#include "llava_buffer.h"
#include "llava_context.h"

llava_buffer::llava_buffer(llava_context* context, size_t wanted_size, size_t alignment) : size(wanted_size) {
    if (alignment == 0) {
        alignment = 1;
    }
    u32 real_size = wanted_size + alignment - 1;
    real_size -= real_size % alignment;
    auto wantedBits = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;
    vkr::Buffer buffer(*context->get_device(), {{}, real_size, wantedBits});
    vk::MemoryRequirements requirements = buffer.getMemoryRequirements();
    assert((requirements.memoryTypeBits & (1 << memoryTypeIndex)) != 0);
    vkr::DeviceMemory bufferMemory(*context->get_device(), {requirements.size, memoryTypeIndex});
    buffer.bindMemory(*bufferMemory, 0);
}

llava_buffer::llava_buffer(llava_context* context, ggml_data_descriptor const& table) : size(0) {
    cout << "Copying buffer " << table.name << endl;
    auto wantedBits = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;
    if (table.ftype == ggml_value_type::q4_0) {
        uint32_t element_count = table.shape2 * table.shape1;
        element_count = (element_count + 1023) & ~1023U;
        uint32_t block_count = element_count / 16;
        block_count = (block_count + 1023) & ~1023U;


        vkr::Buffer stagingBuffer(*context->get_device(), {{}, block_count * 20, wantedBits});
        vk::MemoryRequirements requirements = stagingBuffer.getMemoryRequirements();
        assert((requirements.memoryTypeBits & (1 << memoryTypeIndex)) != 0);
        vkr::DeviceMemory stagingBufferMemory(*context->get_device(), {requirements.size, memoryTypeIndex});
        stagingBuffer.bindMemory(*stagingBufferMemory, 0);

        vkr::Buffer DBuffer(*context->get_device(), {{}, block_count * 4, wantedBits});
        vkr::DeviceMemory DBufferMemory(*context->get_device(), {DBuffer.getMemoryRequirements().size, memoryTypeIndex});
        DBuffer.bindMemory(*DBufferMemory, 0);

        vkr::Buffer QBuffer(*context->get_device(), {{}, block_count * 16, wantedBits});
        vkr::DeviceMemory QBufferMemory(*context->get_device(), {QBuffer.getMemoryRequirements().size, memoryTypeIndex});
        QBuffer.bindMemory(*QBufferMemory, 0);

        void* data = stagingBufferMemory.mapMemory(0, table.size);
        memcpy(data, context->get_model()->mapping + table.offset, (size_t)table.size);
        stagingBufferMemory.unmapMemory();
        context->get_pipeline("q4_0_split", 3)->simple_call({&stagingBuffer, &QBuffer, &DBuffer}, block_count / 1024);

        storages.emplace_back(std::move(DBuffer), std::move(DBufferMemory));
        backing_sizes.emplace_back(block_count * 4);
        storages.emplace_back(std::move(QBuffer), std::move(QBufferMemory));
        backing_sizes.emplace_back(block_count * 16);
    } else if (table.ftype == ggml_value_type::f32) {
        uint32_t element_count = table.shape2 * table.shape1;
        element_count = (element_count + 1023) & ~1023U;


        vkr::Buffer buffer(*context->get_device(), {{}, element_count * 4, wantedBits});
        vk::MemoryRequirements requirements = buffer.getMemoryRequirements();
        assert((requirements.memoryTypeBits & (1 << memoryTypeIndex)) != 0);
        vkr::DeviceMemory bufferMemory(*context->get_device(), {requirements.size, memoryTypeIndex});
        buffer.bindMemory(*bufferMemory, 0);
        void* data = bufferMemory.mapMemory(0, table.size);
        memcpy(data, context->get_model()->mapping + table.offset, (size_t)table.size);
        bufferMemory.unmapMemory();

        storages.emplace_back(std::move(buffer), std::move(bufferMemory));
        backing_sizes.emplace_back(element_count * 4);
    } else {
        assert(false);
    }
}