#version 450

#extension GL_GOOGLE_include_directive:enable

#define MATMUL_Q4_BLOCKS_PER_ROW MATMUL_DIM_Q4_BLOCKS_PER_ROW
#define MATMUL_Q4_BLOCK_COUNT_PER_WORKER MATMUL_DIM_Q4_BLOCK_COUNT_PER_WORKER
#define MATMUL_X_CID MATMUL_DIM_ROW_PER_WAVEFRONT_CID
#define MATMUL_Y_CID MATMUL_DIM_ROW_WORKER_COUNT_CID
#define MATMUL_X MATMUL_DIM_ROW_PER_WAVEFRONT
#define MATMUL_Y MATMUL_DIM_ROW_WORKER_COUNT
#define MATMUL_ROW_WORKER_COUNT_LOG2 MATMUL_DIM_ROW_WORKER_COUNT_LOG2

#define LOCAL_SUM_BITS MATMUL_ROW_WORKER_COUNT_LOG2

#include "common.glsl"

layout (binding = 0) buffer OutBuffer {
    float values[];
} outp;

layout (binding = 1) buffer readonly Matrix1DBuffer {
    float values[];
} mat1d;

layout (binding = 2) buffer readonly Matrix1QBuffer {
    int values[][4];
} mat1q;

layout (binding = 3) buffer readonly Matrix2DBuffer {
    float values[];
} mat2d;

layout (binding = 4) buffer readonly Matrix2QBuffer {
    int values[][4];
} mat2q;

layout (binding = 5) buffer readonly InFBuffer {
    vec4 values[][4][2];
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
    [[unroll]] for (int t = 0; t < MATMUL_Q4_BLOCK_COUNT_PER_WORKER; t++) {
        float block_mat_value = 0;
        uint block_id = MATMUL_Q4_BLOCK_COUNT_PER_WORKER * row_worker_id + t;
        const bool skipped = (block_id >= MATMUL_Q4_BLOCKS_PER_ROW);
        block_id = min(block_id, MATMUL_Q4_BLOCKS_PER_ROW - 1);
        [[unroll]] for (int block_block_id = 0; block_block_id < 4; block_block_id++) {
            ivec4 v2 = ivec4(mat1q.values[row_id * MATMUL_Q4_BLOCKS_PER_ROW + block_id][block_block_id]);
            v2 >>= ivec4(0, 4, 8, 12);
            block_mat_value += dot(vec4((v2 & 0xf) - 8), inp.values[block_id][block_block_id][0]);
            v2 >>= 16;
            block_mat_value += dot(vec4((v2 & 0xf) - 8), inp.values[block_id][block_block_id][1]);
        }
        worker_sum += block_mat_value * (skipped ? 0. : mat1d.values[row_id * MATMUL_Q4_BLOCKS_PER_ROW + block_id]);
    }

    float mat1res = local_sum(local_row_id, row_worker_id, worker_sum);

    worker_sum = 0;
    [[unroll]] for (int t = 0; t < MATMUL_Q4_BLOCK_COUNT_PER_WORKER; t++) {
        float block_mat_value = 0;
        uint block_id = MATMUL_Q4_BLOCK_COUNT_PER_WORKER * row_worker_id + t;
        const bool skipped = (block_id >= MATMUL_Q4_BLOCKS_PER_ROW);
        block_id = min(block_id, MATMUL_Q4_BLOCKS_PER_ROW - 1);
        [[unroll]] for (int block_block_id = 0; block_block_id < 4; block_block_id++) {
            ivec4 v2 = ivec4(mat2q.values[row_id * MATMUL_Q4_BLOCKS_PER_ROW + block_id][block_block_id]);
            v2 >>= ivec4(0, 4, 8, 12);
            block_mat_value += dot(vec4((v2 & 0xf) - 8), inp.values[block_id][block_block_id][0]);
            v2 >>= 16;
            block_mat_value += dot(vec4((v2 & 0xf) - 8), inp.values[block_id][block_block_id][1]);
        }
        worker_sum += block_mat_value * (skipped ? 0. : mat2d.values[row_id * MATMUL_Q4_BLOCKS_PER_ROW + block_id]);
    }

    float mat2res = local_sum(local_row_id, row_worker_id, worker_sum);

    if (row_worker_id == 0) {
        outp.values[row_id] = mat1res * mat2res / (exp(-mat2res) + 1);
    }
}
