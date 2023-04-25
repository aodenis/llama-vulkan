#version 450

// q4_0 matrix are an array of "blocks"
// A block is : a float (4 bytes), followed by 16 bytes of char
// This shader splits the float and the chars apart for easier use later by other shaders

layout (binding = 0) buffer InputBuffer {
    uint values[];
} inp;

layout (binding = 1) buffer OutDBuffer {
    uint values[];
} outD;

layout (binding = 2) buffer OutQBuffer {
    uint values[];
} outQ;

layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

void main()
{
    uint i = gl_GlobalInvocationID.x;
    outD.values[i] = inp.values[5 * i];
    outQ.values[4 * i    ] = inp.values[5 * i + 1];
    outQ.values[4 * i + 1] = inp.values[5 * i + 2];
    outQ.values[4 * i + 2] = inp.values[5 * i + 3];
    outQ.values[4 * i + 3] = inp.values[5 * i + 4];
}
