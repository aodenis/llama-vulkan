#version 450

#extension GL_GOOGLE_include_directive:enable

#define LOCAL_SUM_BITS BACKLOG_BITS

#include "common.glsl"

layout (binding = 0) buffer readonly ConfigBuffer {
    uint token_count;
    uint dim;
    uint pad1;
    uint pad2;
} config;

layout (binding = 1) buffer InOutBuffer {
    float values[];
} iobuf;

#ifdef USE_SPEVAR
layout (local_size_x_id = SOFTMAX_HEAD_PER_WAVEFRONT_CID, local_size_y_id = BACKLOG_CID, local_size_z = 1) in;
#else
layout (local_size_x = SOFTMAX_HEAD_PER_WAVEFRONT, local_size_y = BACKLOG, local_size_z = 1) in;
#endif

void main()
{
    const float main_factor = inversesqrt(float(ROT));
    const uint head_id = gl_GlobalInvocationID.x;
    const uint cache_entry_id = gl_GlobalInvocationID.y;

    const float input_value = (head_id < HEAD_COUNT) ? iobuf.values[cache_entry_id * HEAD_COUNT + head_id] : 0.;
    float a = exp(input_value * main_factor);
    if (cache_entry_id > config.token_count) {
        a = 0;
    }

    const float head_exp_sum = local_sum(gl_LocalInvocationID.x, cache_entry_id, a);

    if (head_id < HEAD_COUNT) {
        iobuf.values[cache_entry_id * HEAD_COUNT + head_id] = a / head_exp_sum;
    }
}
