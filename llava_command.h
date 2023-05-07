#ifndef VULKAN_LLAMA_LLAVA_COMMAND_H
#define VULKAN_LLAMA_LLAVA_COMMAND_H

#include "types.h"
#include <vulkan/vulkan.hpp>

class llava_command {
public:
    llava_command(llava_pipeline *pipeline, const initializer_list<llava_buffer*> &buffers, const initializer_list<vk::Event> &events, uint32_t countX, uint32_t countY = 1, uint32_t countZ = 1);
    ~llava_command();

public:
    llava_context* const context;
    const vk::DescriptorSet descriptorSet;
    const vk::CommandBuffer commandBuffer;
    const vk::Event completionEvent;
};

#endif
