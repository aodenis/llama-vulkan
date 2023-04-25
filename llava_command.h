#ifndef VULKAN_LLAMA_LLAVA_COMMAND_H
#define VULKAN_LLAMA_LLAVA_COMMAND_H

#include "types.h"

class llava_command {
public:
    llava_command(llava_pipeline *pipeline, const vector<vkr::Buffer*> &buffers, uint32_t countX, uint32_t countY = 1, uint32_t countZ = 1);
    void run_sync();

public:
    const weak_ptr<llava_context> w_context;
    const vkr::DescriptorSet descriptorSet;
    const vkr::CommandBuffer commandBuffer;
    const vk::SubmitInfo submitInfo;
};


#endif //VULKAN_LLAMA_LLAVA_COMMAND_H
