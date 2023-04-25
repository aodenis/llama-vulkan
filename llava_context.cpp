#include "llava_context.h"
#include "llava_pipeline.h"
#include <iostream>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include "ggml_file.h"

using namespace std;
namespace vkr = vk::raii;

// Constant for now !
const uint32_t heapIndex = 1;
const uint32_t memoryTypeIndex = 3;


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
    model = std::make_shared<ggml_file>("/home/denis/Documents/CProj/llm-conv/ggml-model-q4_0-7B.bin");
    model->print_info();

    // initialize the vkr::ApplicationInfo structure
    vk::ApplicationInfo applicationInfo("llm", 1, "llm0", 1, VK_API_VERSION_1_3);

    vector<const char*> enabled_layers = {"VK_LAYER_KHRONOS_validation"};
    // create an Instance
    vulkan_context = make_shared<vkr::Context>();
    vulkan_instance = make_shared<vkr::Instance>(std::move(vkr::Instance(*vulkan_context, {{}, &applicationInfo, static_cast<uint32_t>(enabled_layers.size()), enabled_layers.data(),
                                                0, nullptr})));

    physicalDevice = make_shared<vkr::PhysicalDevice>(get_physical_device());
    uint32_t queueFamilyIndex = get_queue_family_index();

    // create a Device
    float queuePriority = 0.0f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), queueFamilyIndex, 1, &queuePriority);
    device = make_shared<vkr::Device>(std::move(vkr::Device(*physicalDevice, {vk::DeviceCreateFlags(), deviceQueueCreateInfo})));

    // create a CommandPool to allocate a CommandBuffer from
    command_pool = make_shared<vkr::CommandPool>(std::move(vkr::CommandPool(*device, {{}, queueFamilyIndex})));

    // Descriptor pool
    vk::DescriptorPoolSize descriptorPoolSize(vk::DescriptorType::eStorageBuffer, 8);
    descriptor_pool = make_shared<vkr::DescriptorPool>(std::move(vkr::DescriptorPool(*device, vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                                                                                                                           1024, 1, &descriptorPoolSize))));

    // Queue
    queue = make_shared<vkr::Queue>(*device, queueFamilyIndex, 0);

    // Pipeline cache
    pipeline_cache = make_shared<vkr::PipelineCache>(*device, vk::PipelineCacheCreateInfo({}, 0, nullptr));

    // Create buffers
    vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice->getMemoryProperties();
    assert(memoryProperties.memoryTypeCount > memoryTypeIndex);
    assert(memoryProperties.memoryHeapCount > heapIndex);
    assert((memoryProperties.memoryTypes[memoryTypeIndex].propertyFlags & (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)) == (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));

    llava_pipeline pipeline(shared_from_this(), "built_shaders/q4_0_split.spirv", 3);

    build_base_buffers(&pipeline);

    return 0;
} catch (vk::SystemError &err) {
    cerr << "vkr::SystemError: " << err.what() << endl;
    return 1;
} catch (exception &err) {
    cerr << "std::exception: " << err.what() << endl;
    return 1;
}

void llava_context::build_base_buffers(llava_pipeline* q4_0_split_pipeline) {
    for (auto& table : this->model->get_buffers()) {
        cout << table.name << " (" << table.shape1 << ", " << table.shape2 << "), size in file: " << table.size_in_file() << " type " << table.ftype << endl;

        if (table.ftype != GGML_TYPE_Q4_0) {
            cout << "Bad type ! " << endl;
            continue;
        }
        uint32_t element_count = table.shape2 * table.shape1;
        element_count = (element_count + 1023) & ~1023U;
        uint32_t block_count = element_count / 16;
        block_count = (block_count + 1023) & ~1023U;


        auto wantedBits = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;
        vkr::Buffer stagingBuffer(*device, {{}, block_count * 20, wantedBits});
        vk::MemoryRequirements requirements = stagingBuffer.getMemoryRequirements();
        assert((requirements.memoryTypeBits & (1 << memoryTypeIndex)) != 0);
        vkr::DeviceMemory stagingBufferMemory(*device, {requirements.size, memoryTypeIndex});
        stagingBuffer.bindMemory(*stagingBufferMemory, 0);

        vkr::Buffer DBuffer(*device, {{}, block_count * 4, wantedBits});
        vkr::DeviceMemory DBufferMemory(*device, {DBuffer.getMemoryRequirements().size, memoryTypeIndex});
        DBuffer.bindMemory(*DBufferMemory, 0);

        vkr::Buffer QBuffer(*device, {{}, block_count * 16, wantedBits});
        vkr::DeviceMemory QBufferMemory(*device, {QBuffer.getMemoryRequirements().size, memoryTypeIndex});
        QBuffer.bindMemory(*QBufferMemory, 0);

        void* data = stagingBufferMemory.mapMemory(0, table.size);
        memcpy(data, this->model->mapping + table.offset, (size_t)table.size);
        stagingBufferMemory.unmapMemory();
        q4_0_split_pipeline->simple_call({&stagingBuffer, &QBuffer, &DBuffer}, block_count / 1024);

        model_buffers.emplace_back(std::move(DBuffer), std::move(DBufferMemory));
        model_buffers.emplace_back(std::move(QBuffer), std::move(QBufferMemory));
    }
}

shared_ptr<vkr::Context> llava_context::get_context() {
    assert(vulkan_context != nullptr);
    return vulkan_context;
}

shared_ptr<vkr::Device> llava_context::get_device() {
    assert(device != nullptr);
    return device;
}

shared_ptr<vkr::CommandPool> llava_context::get_command_pool() {
    assert(command_pool != nullptr);
    return command_pool;
}

shared_ptr<vkr::DescriptorPool> llava_context::get_descriptor_pool() {
    assert(descriptor_pool != nullptr);
    return descriptor_pool;
}

shared_ptr<vkr::PipelineCache> llava_context::get_pipeline_cache() {
    assert(pipeline_cache != nullptr);
    return pipeline_cache;
}

shared_ptr<vkr::Queue> llava_context::get_queue() {
    assert(queue != nullptr);
    return queue;
}
