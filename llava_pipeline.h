#ifndef VULKAN_LLAMA_LLAVA_PIPELINE_H
#define VULKAN_LLAMA_LLAVA_PIPELINE_H

#include "types.h"

// A dumb wrapper around vulkan nightmarish Pipeline/PipelineLayout/DescriptorSet/whatever
class llava_pipeline {
public:
    llava_pipeline(const shared_ptr<llava_context>& ctx, const char* shader_source, uint32_t argument_count);
    shared_ptr<llava_command> prepare_call(const vector<vkr::Buffer*> &buffers, uint32_t countX, uint32_t countY = 1, uint32_t countZ = 1);
    void simple_call(const vector<vkr::Buffer*> &buffers, uint32_t countX, uint32_t countY = 1, uint32_t countZ = 1);
    shared_ptr<llava_context> get_context() const;

public:
    const weak_ptr<llava_context> w_context;
    const vkr::ShaderModule shaderModule;
    const vkr::DescriptorSetLayout descriptorSetLayout;
    const vkr::PipelineLayout pipelineLayout;
    const vkr::Pipeline pipeline;
};


#endif //VULKAN_LLAMA_LLAVA_PIPELINE_H
