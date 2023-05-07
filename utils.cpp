#include "utils.h"
#include <cassert>

size_t matrix_size(ggml_value_type type, u32 shape1, u32 shape2) {
    if (type == ggml_value_type::f16) {
        return shape1 * shape2 * 2;
    }
    if (type == ggml_value_type::f32) {
        return shape1 * shape2 * 4;
    }
    size_t base_size = shape1 * shape2 * ((type == ggml_value_type::q4_0) ? 20 : 24);
    assert((base_size % 32) == 0);
    return base_size / 32;
}

size_t matrix_overflow_size(ggml_value_type type, u32 shape1, u32 shape2, u32 alignment) {
    return matrix_size(type, shape1, shape2);
}

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
    }
    return nullptr;
}
