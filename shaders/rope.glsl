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
    const uint head_id = i / (2 * QUARTERROT);
    const uint i0 = i % (2 * QUARTERROT);

    float theta = float(config.token_count) * pow(10000.0, -float(i0)/float(2 * QUARTERROT));

    if (2 * i < DIM) {
        float x = inp.values[2 * i];
        float y = inp.values[2 * i + 1];

        float cr = cos(theta);
        float sr = sin(theta);
        inp.values[2 * i] = x*cr - y*sr;
        inp.values[2 * i + 1] = x*sr + y*cr;
    }
}
