#version 450

#extension GL_GOOGLE_include_directive:enable

#include "common.glsl"

layout (binding = 0) buffer InputBuffer {
    float values[FF_SIZE];
} inp;

#ifdef USE_SPEVAR
layout (local_size_x_id = MAX_WGS_CID, local_size_y = 1, local_size_z = 1) in;
#else
layout (local_size_x = MAX_WGS, local_size_y = 1, local_size_z = 1) in;
#endif

void main()
{
    const uint i = gl_GlobalInvocationID.x;
    const float x = inp.values[i];
    const float sig = 1.0f/(exp(-x) + 1);
    inp.values[i] = x * sig;
}
