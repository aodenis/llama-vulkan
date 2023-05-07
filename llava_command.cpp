#include <iostream>
#include "llava_command.h"
#include "llava_pipeline.h"
#include "llava_context.h"

llava_command::llava_command(llava_pipeline *pipeline,
                             const initializer_list<llava_buffer*> &_buffers,
                             const initializer_list<vk::Event> &events,
                             uint32_t countX,
                             uint32_t countY,
                             uint32_t countZ) :
                             context(pipeline->context),
                             descriptorSet(context->get_device().allocateDescriptorSets({context->get_descriptor_pool(), 1, &pipeline->descriptorSetLayout}).front()),
                             commandBuffer(context->get_device().allocateCommandBuffers({context->get_command_pool(), vk::CommandBufferLevel::ePrimary, 1}).front()),
                             completionEvent(context->get_device().createEvent({}))
{
    vector<pair<vk::Buffer, bool>> buffers;
    for (llava_buffer* buffer: _buffers) {
        for(auto& x : buffer->buffers) {
            buffers.emplace_back(x, buffer->backing_name.empty());
        }
    }

    vector<vk::DescriptorBufferInfo> buffersInfo;
    vector<vk::WriteDescriptorSet> writes;
    writes.reserve(buffers.size());
    buffersInfo.reserve(buffers.size());
    for (auto& buffer : buffers) {
        buffersInfo.emplace_back(buffer.first, 0, -1);
    }

    for (uint32_t i = 0; i < buffersInfo.size(); ++i) {
        writes.emplace_back(descriptorSet, i, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, buffersInfo.data() + i);
    }

    context->get_device().updateDescriptorSets(writes, {});

    vector<vk::Event> _events;
    _events.reserve(events.size());
    for(auto& x : events) {
        if (x) {
            _events.push_back(x);
        }
    }
    bool has_config_buffer = (*_buffers.begin()) == context->config_buffer;
    u32 out_buffer_id = has_config_buffer ? 1 : 0;
    vector<vk::BufferMemoryBarrier> barriers;
    for(u32 i = 0; i < buffers.size(); ++i) {
        auto& [x, is_mutating_buffer] = buffers.at(i);
        if (not is_mutating_buffer) {
            continue;
        }
        auto dstAccessMask = (i == out_buffer_id) ? vk::AccessFlagBits::eMemoryWrite : vk::AccessFlagBits::eMemoryRead;
        barriers.emplace_back(vk::AccessFlagBits::eMemoryWrite, dstAccessMask, context->get_queue_family_index(), context->get_queue_family_index(), x, 0, VK_WHOLE_SIZE);
    }
    commandBuffer.begin(vk::CommandBufferBeginInfo());
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline->pipelineLayout, 0, {descriptorSet}, {});
    if (not _events.empty()) {
        commandBuffer.waitEvents(_events, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, barriers, {});
    }
    commandBuffer.dispatch(countX, countY, countZ);
    commandBuffer.setEvent(completionEvent, vk::PipelineStageFlagBits::eComputeShader);
    commandBuffer.end();
}

llava_command::~llava_command() {
    context->get_device().freeCommandBuffers(context->get_command_pool(), {commandBuffer});
    context->get_device().freeDescriptorSets(context->get_descriptor_pool(), {descriptorSet});
    context->get_device().destroy(completionEvent);
}
