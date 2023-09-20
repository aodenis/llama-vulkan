#include "utils.h"
#include <cassert>
#include <cstring>
#include <unistd.h>

const char* ftype_name(ggml_value_type type) {
    switch(type) {
        case ggml_value_type::f32:
            return "f32";
        case ggml_value_type::f16:
            return "f16";
        case ggml_value_type::q4_0:
            return "q4_0";
        case ggml_value_type::q4_1:
            return "q4_1";
        case ggml_value_type::q5_0:
            return "q5_0";
        case ggml_value_type::q5_1:
            return "q5_1";
        case ggml_value_type::q8_0:
            return "q8_0";
        case ggml_value_type::q8_1:
            return "q8_1";
        case ggml_value_type::q2_K:
            return "q2_K";
        case ggml_value_type::q3_K:
            return "q3_K";
        case ggml_value_type::q4_K:
            return "q4_K";
        case ggml_value_type::q5_K:
            return "q5_K";
        case ggml_value_type::q6_K:
            return "q6_K";
        case ggml_value_type::q8_K:
            return "q8_K";
    }
    return nullptr;
}

bool ends_with(const string& a, const string& b) {
    if (a.size() < b.size())
        return false;
    return memcmp(a.data() + a.size() - b.size(), b.data(), b.size()) == 0;
}

bool starts_with(const string& a, const string& b) {
    if (a.size() < b.size())
        return false;
    return memcmp(a.data(), b.data(), b.size()) == 0;
}

u32 ulog2(u32 n) {
    assert (n != 0);
    u32 i = 0;
    while (((n & 1) == 0)) {
        n >>= 1;
        i++;
    }
    assert (n == 1);
    return i;
}

size_t read_noshort(int fd, void* dst, size_t sz) {
    size_t r = 0;
    errno = 0;
    while(r < sz) {
        long s = read(fd, ((u8*)dst)+r, sz - r);
        if (s <= 0) {
            if (errno) {
                perror("read_noshort");
            }
            return r;
        }
        r += ((size_t)s);
    }
    return r;
}

size_t write_noshort(int fd, void* src, size_t sz) {
    size_t r = 0;
    while(r < sz) {
        long s = write(fd, ((u8*)src)+r, sz - r);
        if (s <= 0) {
            perror("write_noshort");
            return r;
        }
        r += ((size_t)s);
    }
    return r;
}
