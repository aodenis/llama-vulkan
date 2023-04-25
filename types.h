#ifndef VULKAN_LLAMA_TYPES_H
#define VULKAN_LLAMA_TYPES_H

#include <vulkan/vulkan_raii.hpp>
#include <memory>

namespace vkr = vk::raii;
using namespace std;

class llava_command;
class llava_context;
class llava_pipeline;
class llava_layer;
class llava_buffer;

using u32 = uint32_t;
using u64 = uint64_t;
using u16 = uint16_t;

enum class ggml_value_type : uint16_t {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
};

// Constant for now !
const u32 heapIndex = 1;
const u32 memoryTypeIndex = 3;



#endif //VULKAN_LLAMA_TYPES_H
