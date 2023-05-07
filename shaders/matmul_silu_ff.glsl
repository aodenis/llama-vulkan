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
#define LOCAL_SUM_VEC4

#include "common.glsl"

layout (binding = 0) buffer OutBuffer {
    vec4 values[];
} outp;

layout (binding = 1) buffer readonly Matrix1DBuffer {
    vec4 values[];
} mat1d;

layout (binding = 2) buffer readonly Matrix1QBuffer {
    ivec4 values[][4];
} mat1q;

layout (binding = 3) buffer readonly Matrix2DBuffer {
    vec4 values[];
} mat2d;

layout (binding = 4) buffer readonly Matrix2QBuffer {
    ivec4 values[][4];
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
    const uint worker_id = gl_GlobalInvocationID.y;

    vec4 worker_sum = vec4(0);
    [[unroll]] for (int t = 0; t < MATMUL_Q4_BLOCK_COUNT_PER_WORKER; t++) {
        vec4 block_mat_value = vec4(0.);
        uint block_id = MATMUL_Q4_BLOCK_COUNT_PER_WORKER * worker_id + t;
        block_id = min(block_id, MATMUL_Q4_BLOCKS_PER_ROW - 1);

        [[unroll]] for (int block_block_id = 0; block_block_id < 4; block_block_id++) {
            ivec4 sub_block = mat1q.values[row_id * MATMUL_Q4_BLOCKS_PER_ROW + block_id][block_block_id];
            mat4 m = mat4(vec4(sub_block & 0xf), vec4((sub_block >> 4) & 0xf), vec4((sub_block >> 8) & 0xf), vec4((sub_block >> 12) & 0xf));
            block_mat_value += (m - 8.) * inp.values[block_id][block_block_id][0];
            sub_block >>= 16;
            m = mat4(vec4(sub_block & 0xf), vec4((sub_block >> 4) & 0xf), vec4((sub_block >> 8) & 0xf), vec4((sub_block >> 12) & 0xf));
            block_mat_value += (m - 8.) * inp.values[block_id][block_block_id][1];
        }
        if (MATMUL_Q4_BLOCK_COUNT_PER_WORKER * worker_id + t < MATMUL_Q4_BLOCKS_PER_ROW) {
            worker_sum += block_mat_value * mat1d.values[row_id * MATMUL_Q4_BLOCKS_PER_ROW + block_id];
        }
    }

    const vec4 result1 = local_sum(local_row_id, worker_id, worker_sum);

    worker_sum = vec4(0);
    [[unroll]] for (int t = 0; t < MATMUL_Q4_BLOCK_COUNT_PER_WORKER; t++) {
        vec4 block_mat_value = vec4(0.);
        uint block_id = MATMUL_Q4_BLOCK_COUNT_PER_WORKER * worker_id + t;
        block_id = min(block_id, MATMUL_Q4_BLOCKS_PER_ROW - 1);

        [[unroll]] for (int block_block_id = 0; block_block_id < 4; block_block_id++) {
            ivec4 sub_block = mat2q.values[row_id * MATMUL_Q4_BLOCKS_PER_ROW + block_id][block_block_id];
            mat4 m = mat4(vec4(sub_block & 0xf), vec4((sub_block >> 4) & 0xf), vec4((sub_block >> 8) & 0xf), vec4((sub_block >> 12) & 0xf));
            block_mat_value += (m - 8.) * inp.values[block_id][block_block_id][0];
            sub_block >>= 16;
            m = mat4(vec4(sub_block & 0xf), vec4((sub_block >> 4) & 0xf), vec4((sub_block >> 8) & 0xf), vec4((sub_block >> 12) & 0xf));
            block_mat_value += (m - 8.) * inp.values[block_id][block_block_id][1];
        }
        if (MATMUL_Q4_BLOCK_COUNT_PER_WORKER * worker_id + t < MATMUL_Q4_BLOCKS_PER_ROW) {
            worker_sum += block_mat_value * mat2d.values[row_id * MATMUL_Q4_BLOCKS_PER_ROW + block_id];
        }
    }

    const vec4 result2 = local_sum(local_row_id, worker_id, worker_sum);
    if (worker_id == 0) {
        outp.values[row_id] = result1 * result2 / (exp(-result2) + 1);
    }
}
