#ifndef VULKAN_LLAMA_TYPES_H
#define VULKAN_LLAMA_TYPES_H

// #define RUNTIME_BUILD_ENABLED

#include <cstdint>
#include <string>

#define ND [[nodiscard]]

using namespace std;

class llava_context;
class llava_pipeline;
class llava_layer;
class llava_buffer;
class llava_device_memory;
class llava_command_buffer;
class llava_session;
class llava_layer_session_data;
struct specialization_variables_t;

using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;
using u8 = uint8_t;

enum class ggml_value_type : u16 {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_K = 10,
    q3_K = 11,
    q4_K = 12,
    q5_K = 13,
    q6_K = 14,
    q8_K = 15,
};

enum class ReturnCode : u32 {
    ok = 0,
    nok = 1,
    no_such_session = 2,
    tick_already_requested = 3,
    already_running = 4,
    not_tracing = 5,
    batched_tick = 6,
    bad_arguments = 7,
    unknown_command = 8,
};

namespace vk {
    class DeviceMemory;
    class Buffer;
    class ShaderModule;
    class DescriptorSetLayout;
    class PipelineLayout;
    class Pipeline;
}

#endif //VULKAN_LLAMA_TYPES_H
