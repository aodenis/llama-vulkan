#version 450

#extension GL_GOOGLE_include_directive:enable

#define LOCAL_SUM_BITS ROT_BITS

#include "common.glsl"

layout (binding = 0) buffer OutBuffer {
    float values[DIM];
} outp;

layout (binding = 1) buffer readonly VCacheBuffer {
    float values[];
} vcache;

layout (binding = 2) buffer readonly AttnBuffer {
    float values[];
} attn;

#ifdef USE_SPEVAR
layout (local_size_x_id = SOFTMAX_HEAD_PER_WAVEFRONT_CID, local_size_y_id = BACKLOG_CID, local_size_z = 1) in;
#else
layout (local_size_x = SOFTMAX_HEAD_PER_WAVEFRONT, local_size_y = BACKLOG, local_size_z = 1) in;
#endif

void main()
{
    const uint v_row_id = gl_GlobalInvocationID.x;
    const uint local_v_row_id = gl_LocalInvocationID.x;
    const uint backlog_id = gl_GlobalInvocationID.y;

    const float this_match = attn.values[backlog_id * HEAD_COUNT + (v_row_id >> BACKLOG_BITS)] * vcache.values[backlog_id * DIM + v_row_id];

    outp.values[v_row_id] = local_sum(local_v_row_id, backlog_id, this_match);
}
