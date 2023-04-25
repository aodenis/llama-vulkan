#include "llava_command.h"
#include "llava_pipeline.h"
#include "llava_context.h"

llava_command::llava_command(llava_pipeline *pipeline,
                             const vector<vkr::Buffer*> &buffers,
                             uint32_t countX,
                             uint32_t countY,
                             uint32_t countZ) :
                             w_context(pipeline->get_context()),
                             descriptorSet(std::move(pipeline->get_context()->get_device()->allocateDescriptorSets({pipeline->get_context()->get_descriptor_pool()->operator*(), 1, &*pipeline->descriptorSetLayout}).front())),
                             commandBuffer(std::move(pipeline->get_context()->get_device()->allocateCommandBuffers({**pipeline->get_context()->get_command_pool(), vk::CommandBufferLevel::ePrimary, 1}).front())),
                             submitInfo(0, nullptr, nullptr, 1, &*commandBuffer)
{
    vector<vk::DescriptorBufferInfo> buffersInfo;
    vector<vk::WriteDescriptorSet> writes;
    writes.reserve(buffers.size());
    buffersInfo.reserve(buffers.size());
    for (auto& buffer : buffers) {
        // TODO pass size with it ?
        buffersInfo.emplace_back(**buffer, 0, buffer->getMemoryRequirements().size);
    }

    for (uint32_t i = 0; i < buffersInfo.size(); ++i) {
        writes.emplace_back(*descriptorSet, i, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, buffersInfo.data() + i);
    }

    pipeline->get_context()->get_device()->updateDescriptorSets(writes, {});

    commandBuffer.begin(vk::CommandBufferBeginInfo());
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline->pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipeline->pipelineLayout, 0, {*descriptorSet}, {});
    commandBuffer.dispatch(countX, countY, countZ);
    commandBuffer.end();
}

void llava_command::run_sync() {
    assert(not w_context.expired());
    auto context = w_context.lock();
    context->get_queue()->submit({submitInfo});
    context->get_queue()->waitIdle();
}
