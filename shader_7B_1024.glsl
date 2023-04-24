#version 450

struct q4_0 {
    float d;
    int values[4];
};

layout (binding = 0) buffer MatrixBuffer {
    q4_0 values[];
} buf_a;

layout (binding = 1) buffer RowBuffer {
    q4_0 values[];
} buf_b;

layout (binding = 2) buffer OutRowBuffer {
    float values[];
} buf_c;

layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

// 16 bytes of data -> uvec4

float compute_q4_0_dot(q4_0 a, q4_0 b) {
    int total = 0;
    for (uint i = 0; i < 4; i++) {
        int x = a.values[i];
        int y = b.values[i];
        for(uint j = 0; i < 4; ++j) {
            total += (((x >> (4*j))   & 0xf) - 8) * (((y >> (4*j))   & 0xf) - 8);
            total += (((x >> (4*j+4)) & 0xf) - 8) * (((y >> (4*j+4)) & 0xf) - 8);
        }
    }
    return a.d * b.d * total;
}

void main()
{
    uint row = gl_GlobalInvocationID.x;
    float res = 0;
    for (uint i = 0; i < 128; i++) {
        res += compute_q4_0_dot(buf_a.values[128 * row + i], buf_b.values[i]);
    }
    buf_c.values[row] = res;
}
