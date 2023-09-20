#ifndef VULKAN_LLAMA_UTILS_H
#define VULKAN_LLAMA_UTILS_H

#include "types.h"
#include <string>

const char* ftype_name(ggml_value_type type);

inline u64 updiv(u64 a, u64 b) {
    return (a + b - 1) / b;
}

u32 ulog2(u32 n);

bool ends_with(const string& a, const string& b);
bool starts_with(const string& a, const string& b);
size_t read_noshort(int fd, void* dst, size_t sz);
size_t write_noshort(int fd, void* src, size_t sz);

#ifdef EMBEDDED_SPV

struct packed_data_entry_t {
    u32 name_offset;
    u32 data_offset;
    u32 data_length;
} __attribute__((packed));

struct packed_data_t {
    u32 count;
    packed_data_entry_t entries[];
} __attribute__((packed));

extern const unsigned char raw_packed_shaders[];
#endif

#endif //VULKAN_LLAMA_UTILS_H
