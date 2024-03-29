#ifndef VULKAN_LLAMA_LLAVA_PIPELINE_H
#define VULKAN_LLAMA_LLAVA_PIPELINE_H

#include "types.h"
#include <vector>
#include <vulkan/vulkan.hpp>

// A dumb wrapper around vulkan nightmarish Pipeline/PipelineLayout/DescriptorSet/whatever
class llava_pipeline {
    friend class llava_command_buffer;
public:
    llava_pipeline(llava_context* ctx, string shader_name, specialization_variables_t const& spevar, bool use_prebuilt_shader, uint32_t argument_count);
    ~llava_pipeline();

public:
    const u32 argument_count;
    llava_context* const context;
    const string shader_name;

private:
    vk::ShaderModule shaderModule;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;
};


#endif //VULKAN_LLAMA_LLAVA_PIPELINE_H
