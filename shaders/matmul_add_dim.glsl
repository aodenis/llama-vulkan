#version 450

#extension GL_GOOGLE_include_directive:enable

#define MATMUL_Q4_BLOCKS_PER_ROW MATMUL_DIM_Q4_BLOCKS_PER_ROW
#define MATMUL_Q4_BLOCK_COUNT_PER_WORKER MATMUL_DIM_Q4_BLOCK_COUNT_PER_WORKER
#define MATMUL_X_CID MATMUL_DIM_ROW_PER_WAVEFRONT_CID
#define MATMUL_Y_CID MATMUL_DIM_ROW_WORKER_COUNT_CID
#define MATMUL_X MATMUL_DIM_ROW_PER_WAVEFRONT
#define MATMUL_Y MATMUL_DIM_ROW_WORKER_COUNT
#define MATMUL_ROW_WORKER_COUNT_LOG2 MATMUL_DIM_ROW_WORKER_COUNT_LOG2
#define MATMUL_ADD

#include "matmul.glsl"