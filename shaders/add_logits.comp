#version 450

#extension GL_GOOGLE_include_directive:enable

#include "common.glsl"

layout (binding = 0) buffer InputBuffer {
    float values[];
} inp;

layout (binding = 1) buffer readonly DeltaBuffer {
    float values[];
} delta;

#ifdef USE_SPEVAR
layout (local_size_x_id = MAX_WGS_CID, local_size_y = 1, local_size_z = 1) in;
#else
layout (local_size_x = MAX_WGS, local_size_y = 1, local_size_z = 1) in;
#endif

void main()
{
    const uint i = gl_GlobalInvocationID.x;
    const uint z_id = gl_GlobalInvocationID.z;
    if (i < DIM) {
        inp.values[i + z_id * DIM] += delta.values[i + z_id * DIM];
    }
}
