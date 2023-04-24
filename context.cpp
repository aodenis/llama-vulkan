#include "llava_context.h"
#include <iostream>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <csignal>
#include <fcntl.h>
#include "ggml_file.h"

using namespace std;
namespace vkr = vk::raii;

// Constant for now !
const uint32_t heapIndex = 1;
const uint32_t memoryTypeIndex = 3;

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

vkr::PhysicalDevice llava_context::get_physical_device() {
    for(vkr::PhysicalDevice const &pd : vulkan_instance->enumeratePhysicalDevices()) {
        if (pd.getProperties().deviceType == vk::PhysicalDeviceType::eIntegratedGpu)
            return pd;
    }
    assert(false);
}

uint32_t llava_context::get_queue_family_index() {
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice->getQueueFamilyProperties();
    // Compute queue family index
    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
        if (queueFamilyProperties.at(i).queueFlags & vk::QueueFlagBits::eCompute) {
            return i;
        }
    }
    assert(false);
}

int llava_context::run(int argc, char **argv) try {
    model = std::make_unique<ggml_file>("/home/denis/Documents/CProj/llm-conv/ggml-model-q4_0-7B.bin");
    model->print_info();
    vector<uint32_t> tokens;
    model->tokenize(tokens, " Yohoho", true);
    vector<uint8_t> shader_code;
    slurp_file(shader_code, (argc > 1) ? argv[argc-1] : "../shader.spirv");

    // initialize the vkr::ApplicationInfo structure
    vk::ApplicationInfo applicationInfo("llm", 1, "llm0", 1, VK_API_VERSION_1_3);

    vector<const char*> enabled_layers = {"VK_LAYER_KHRONOS_validation"};
    // create an Instance
    vulkan_context = make_unique<vkr::Context>();
    vulkan_instance = make_unique<vkr::Instance>(std::move(vkr::Instance(*vulkan_context, {{}, &applicationInfo, static_cast<uint32_t>(enabled_layers.size()), enabled_layers.data(),
                                                0, nullptr})));

    physicalDevice = make_unique<vkr::PhysicalDevice>(get_physical_device());
    uint32_t queueFamilyIndex = get_queue_family_index();

    // create a Device
    float queuePriority = 0.0f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), queueFamilyIndex, 1, &queuePriority);
    device = make_unique<vkr::Device>(std::move(vkr::Device(*physicalDevice, {vk::DeviceCreateFlags(), deviceQueueCreateInfo})));

    // create a CommandPool to allocate a CommandBuffer from
    commandPool = make_unique<vkr::CommandPool>(std::move(vkr::CommandPool(*device, {{}, queueFamilyIndex})));

    // Descriptor pool
    vk::DescriptorPoolSize descriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1);
    descriptorPool = make_unique<vkr::DescriptorPool>(std::move(vkr::DescriptorPool(*device, vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                                                                            1024, 1, &descriptorPoolSize))));

    // Queue
    vkr::Queue queue(*device, queueFamilyIndex, 0);

    // Pipeline cache
    vkr::PipelineCache pipelineCache(*device, {{}, 0, nullptr});
    vkr::ShaderModule computeShaderModule(*device, vk::ShaderModuleCreateInfo({}, shader_code.size(), (uint32_t const*)shader_code.data()));

    // Create buffers
    vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice->getMemoryProperties();
    assert(memoryProperties.memoryTypeCount > memoryTypeIndex);
    assert(memoryProperties.memoryHeapCount > heapIndex);
    assert((memoryProperties.memoryTypes[memoryTypeIndex].propertyFlags & (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)) == (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));

    build_buffers();
    // end

/*
    // Descriptor set
    vk::DescriptorSetLayoutBinding layoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
    vkr::DescriptorSetLayout descriptorSetLayout(*device, {{}, 1, &layoutBinding});
    vkr::DescriptorSet descriptorSet(std::move(device->allocateDescriptorSets({**descriptorPool, 1, &*descriptorSetLayout}).front()));

    // Update set
    vk::DescriptorBufferInfo uniformBufferInfo(*stagingBuffer, 0, bufferSize);
    vk::WriteDescriptorSet descWrite(*descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, {}, &uniformBufferInfo);
    device->updateDescriptorSets({descWrite}, {});
    // end

    // Pipeline cache
    vkr::PipelineLayout pipelineLayout(*device, {{}, 1, &*descriptorSetLayout});
    vkr::Pipeline pipeline(*device, pipelineCache, {{}, {{}, vk::ShaderStageFlagBits::eCompute, *computeShaderModule, "main"}, *pipelineLayout});
    // end

    vkr::CommandBuffer commandBuffer(std::move(device->allocateCommandBuffers({**commandPool, vk::CommandBufferLevel::ePrimary, 1}).front()));
    commandBuffer.begin(vk::CommandBufferBeginInfo());
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, {*descriptorSet}, {});
    commandBuffer.dispatch(256, 4, 1);
    commandBuffer.end();

    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &*commandBuffer);

    queue.submit({submitInfo});

    queue.waitIdle();
    */
    return 0;
} catch (vk::SystemError &err) {
    cerr << "vkr::SystemError: " << err.what() << endl;
    return 1;
} catch (exception &err) {
    cerr << "std::exception: " << err.what() << endl;
    return 1;
}

void llava_context::build_buffers() {
    for (auto& table : this->model->get_buffers()) {
        vkr::Buffer stagingBuffer(*device, {{}, table.size, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer});
        vk::MemoryRequirements requirements = stagingBuffer.getMemoryRequirements();
        assert((requirements.memoryTypeBits & (1 << memoryTypeIndex)) != 0);
        vkr::DeviceMemory stagingBufferMemory(*device, {requirements.size, memoryTypeIndex});
        stagingBuffer.bindMemory(*stagingBufferMemory, 0);

        void* data = stagingBufferMemory.mapMemory(0, table.size);
        memcpy(data, this->model->mapping + table.offset, (size_t)table.size);
        stagingBufferMemory.unmapMemory();
        model_buffers.emplace_back(std::move(stagingBuffer), std::move(stagingBufferMemory));
    }
}
