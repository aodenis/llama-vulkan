#version 450

#extension GL_GOOGLE_include_directive:enable

#include "common.glsl"

layout (binding = 0) buffer readonly ConfigBuffer {
    uint token_count;
    uint pad0;
    uint pad1;
    uint pad2;
} config;

layout (binding = 1) buffer CacheBuffer {
     float values[];
} cache;

layout (binding = 2) buffer readonly InputBuffer {
     float values[DIM];
} inp;


#ifdef USE_SPEVAR
layout (local_size_x_id = MAX_WGS_CID, local_size_y = 1, local_size_z = 1) in;
#else
layout (local_size_x = MAX_WGS, local_size_y = 1, local_size_z = 1) in;
#endif

void main()
{
    const uint i = gl_GlobalInvocationID.x;
    cache.values[config.token_count * DIM + i] = inp.values[i];
}
