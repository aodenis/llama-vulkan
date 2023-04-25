#include "llava_pipeline.h"
#include "llava_context.h"
#include "llava_command.h"
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <csignal>
#include <fcntl.h>

void slurp_file(vector<uint8_t>& out, const char* path) {
    int fd = open(path, O_RDONLY);
    assert(fd >= 0);
    while(true) {
        uint32_t cur = out.size();
        out.resize(out.size() + 1024);
        ssize_t r = read(fd, out.data() + cur, 1024);
        assert(r >= 0);
        out.resize(cur + r);
        if (r == 0) {
            close(fd);
            return;
        }
    }
}

vkr::ShaderModule createShaderModule(const shared_ptr<llava_context>& ctx, const char* shader_source) {
    vector<uint8_t> shader_code;
    slurp_file(shader_code, shader_source);
    return {*(ctx->get_device()), vk::ShaderModuleCreateInfo({}, shader_code.size(), (uint32_t const*)shader_code.data())};
}

vkr::DescriptorSetLayout createDescriptorSetLayout(const shared_ptr<llava_context>& ctx, uint32_t argument_count) {
    vector<vk::DescriptorSetLayoutBinding> bindings;
    bindings.resize(argument_count);
    while(bindings.size() < argument_count) {
        bindings.emplace_back(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
    }
    return {*(ctx->get_device()), {{}, argument_count, bindings.data()}};
}

llava_pipeline::llava_pipeline(const shared_ptr<llava_context>& ctx,
                               const char* shader_source,
                               uint32_t argument_count) : w_context(ctx),
                                                          shaderModule(createShaderModule(ctx, shader_source)),
                                                          descriptorSetLayout(createDescriptorSetLayout(ctx, argument_count)),
                                                          pipelineLayout(*(ctx->get_device()), {{}, 1, &*descriptorSetLayout}),
                                                          pipeline(*(ctx->get_device()), *(ctx->get_pipeline_cache()),
                                                                   {{}, {{}, vk::ShaderStageFlagBits::eCompute, *shaderModule, "main"}, *pipelineLayout}) {

}

shared_ptr<llava_command> llava_pipeline::prepare_call(const vector<vkr::Buffer*> &buffers, uint32_t countX, uint32_t countY, uint32_t countZ) {
    return make_shared<llava_command>(this, buffers, countX, countY, countZ);
}

void llava_pipeline::simple_call(const vector<vkr::Buffer *> &buffers, uint32_t countX, uint32_t countY, uint32_t countZ) {
    llava_command command(this, buffers, countX, countY, countZ);
    command.run_sync();
}

shared_ptr<llava_context> llava_pipeline::get_context() const {
    return w_context.lock();
}
