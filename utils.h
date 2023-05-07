#ifndef VULKAN_LLAMA_UTILS_H
#define VULKAN_LLAMA_UTILS_H

#include "types.h"

size_t matrix_size(ggml_value_type type, u32 shape1, u32 shape2);
size_t matrix_overflow_size(ggml_value_type type, u32 shape1, u32 shape2, u32 alignment);
const char* ftype_name(ggml_value_type type);

inline u64 updiv(u64 a, u64 b) {
    return (a + b - 1) / b;
}

#endif //VULKAN_LLAMA_UTILS_H
