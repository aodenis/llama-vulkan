#ifndef MATMUL_X
#error "This file should be included, not compiled"
#endif

#define LOCAL_SUM_BITS MATMUL_ROW_WORKER_COUNT_LOG2
#define LOCAL_SUM_VEC4

#include "common.glsl"

#ifdef MATMUL_ADD
layout (binding = 0) buffer OutBuffer {
#else
layout (binding = 0) buffer writeonly OutBuffer {
#endif
    vec4 values[]; // [Z][MATMUL_Q4_BLOCKS_PER_ROW * 8]
} outp;

layout (binding = 1) buffer readonly MatrixDBuffer {
#ifdef USE_FP16_DBASE
    f16vec4 values[];
#else
    vec4 values[];
#endif
} matd;

layout (binding = 2) buffer readonly MatrixQBuffer {
    uvec4 values[][8];
} matq;

layout (binding = 3) buffer readonly InFBuffer {
    vec4 values[][8]; // [Z][MATMUL_Q4_BLOCKS_PER_ROW][4][2]
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
    const uint z_id = gl_GlobalInvocationID.z * BATCH_ENABLED;

    vec4 worker_sum = vec4(0);
    [[unroll]] for (int t = 0; t < MATMUL_Q4_BLOCK_COUNT_PER_WORKER; t++) {
        vec4 block_mat_value = vec4(0.);
        uint block_id = min(t * MATMUL_Y + worker_id, MATMUL_Q4_BLOCKS_PER_ROW - 1);

        [[unroll]] for (int block_block_id = 0; block_block_id < 8; block_block_id++) {
            uvec4 sub_block = matq.values[row_id * MATMUL_Q4_BLOCKS_PER_ROW + block_id][block_block_id];
            // sub_block ^= 0x80808080;
            mat4 m = mat4(vec4(sub_block & 0xff), vec4((sub_block >> 8) & 0xff), vec4((sub_block >> 16) & 0xff), vec4((sub_block >> 24) & 0xff));
            block_mat_value += (m - 128.) * inp.values[z_id * MATMUL_Q4_BLOCKS_PER_ROW + block_id][block_block_id];
        }
        if (t * MATMUL_Y + worker_id < MATMUL_Q4_BLOCKS_PER_ROW) {
            worker_sum += block_mat_value * vec4(matd.values[row_id * MATMUL_Q4_BLOCKS_PER_ROW + block_id]);
        }
    }

    const vec4 result = local_sum(local_row_id, worker_id, worker_sum);
    if (worker_id == 0) {
    #ifdef MATMUL_ADD
        outp.values[z_id * gl_NumWorkGroups.x * MATMUL_X + row_id] += result;
    #else
        outp.values[z_id * gl_NumWorkGroups.x * MATMUL_X + row_id] = result;
    #endif
    }
}

