#ifndef MATMUL_X
#error "This file should be included, not compiled"
#endif

#define LOCAL_SUM_BITS MATMUL_ROW_WORKER_COUNT_LOG2

#include "common.glsl"

layout (binding = 0) buffer OutBuffer {
    float values[];
} outp;

layout (binding = 1) buffer readonly MatrixDBuffer {
    float values[];
} matd;

layout (binding = 2) buffer readonly MatrixQBuffer {
    int values[][4];
} matq;

layout (binding = 3) buffer readonly InFBuffer {
    vec2 values[][4][4];
} inp;


#ifdef USE_SPEVAR
layout (local_size_x_id = MATMUL_X_CID, local_size_y_id = MATMUL_Y_CID, local_size_z = 1) in;
#else
layout (local_size_x = MATMUL_X, local_size_y = MATMUL_Y, local_size_z = 1) in;
#endif

void main()
{
    const uint row_id = gl_GlobalInvocationID.x;
    const uint local_row_id = gl_LocalInvocationID.x;
    const uint row_worker_id = gl_GlobalInvocationID.y;

    float worker_sum = 0;
    #pragma unroll
    for (int t = 0; t < MATMUL_Q4_BLOCK_COUNT_PER_WORKER; t++) {
        float block_mat_value = 0;
        uint block_id = MATMUL_Q4_BLOCK_COUNT_PER_WORKER * row_worker_id + t;
        const bool skipped = (block_id >= MATMUL_Q4_BLOCKS_PER_ROW);
        block_id = min(block_id, MATMUL_Q4_BLOCKS_PER_ROW - 1);
        #pragma unroll 4
        for (int block_block_id = 0; block_block_id < 4; block_block_id++) {
            int b_base = matq.values[row_id * MATMUL_Q4_BLOCKS_PER_ROW + block_id][block_block_id];
            #pragma unroll 4
            for (int block_byte_id = 0; block_byte_id < 4; block_byte_id++) {
                vec2 v1 = inp.values[block_id][block_block_id][block_byte_id];
                ivec2 v2 = ivec2(b_base, b_base >> 4);
                v2 &= 0xf;
                v2 -= 8;
                block_mat_value += dot(vec2(v2), v1);
                b_base >>= 8;
            }
        }
        worker_sum += block_mat_value * (skipped ? 0. : matd.values[row_id * MATMUL_Q4_BLOCKS_PER_ROW + block_id]);
    }

    outp.values[row_id] = local_sum(local_row_id, row_worker_id, worker_sum);
}
