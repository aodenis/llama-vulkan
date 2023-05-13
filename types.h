#ifndef VULKAN_LLAMA_TYPES_H
#define VULKAN_LLAMA_TYPES_H

// #define RUNTIME_BUILD_ENABLED

#include <cstdint>

using namespace std;

class llava_command;
class llava_context;
class llava_pipeline;
class llava_layer;
class llava_buffer;
class llava_device_memory;
class llava_command_buffer;

using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;
using u8 = uint8_t;

enum class ggml_value_type : u16 {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
};

#endif //VULKAN_LLAMA_TYPES_H
