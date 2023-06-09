#version 450

#extension GL_GOOGLE_include_directive:enable

#include "common.glsl"

layout (binding = 0) buffer OutBuffer {
    float values[BACKLOG * HEAD_COUNT];
} outp;

layout (binding = 1) buffer readonly MatrixDBuffer {
    vec4 values[BACKLOG * HEAD_COUNT * QUARTERROT];  // [BACKLOG][HEAD_COUNT][QUARTERROT]
} matd;

layout (binding = 2) buffer readonly InFBuffer {
    vec4 values[HEAD_COUNT * QUARTERROT]; // [HEAD_COUNT][QUARTERROT], logit
} inp;

#ifdef USE_SPEVAR
layout (local_size_x_id = MAX_WGS_CID, local_size_y = 1, local_size_z = 1) in;
#else
layout (local_size_x = MAX_WGS, local_size_y = 1, local_size_z = 1) in;
#endif

void main()
{
    const uint row_id = gl_GlobalInvocationID.x;
    const uint head_id = row_id % HEAD_COUNT;

    const uint clamped_row_id = min(row_id, BACKLOG * HEAD_COUNT - 1);
    float result = 0;
    // TODO This could be parallel
    for (int i = 0; i < QUARTERROT; i++) {
        result += dot(inp.values[head_id * QUARTERROT + i], matd.values[clamped_row_id * QUARTERROT + i]);
    }

    if (clamped_row_id == row_id) {
        outp.values[row_id] = result;
    }
}
