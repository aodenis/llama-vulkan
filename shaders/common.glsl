#extension GL_EXT_control_flow_attributes:enable
#extension GL_EXT_shader_16bit_storage:enable

#ifdef USE_SPEVAR

layout (constant_id = 0) const uint HEAD_COUNT = 32;
layout (constant_id = 1) const uint QUARTERROT = 32;
layout (constant_id = 2) const uint BACKLOG = 128;
layout (constant_id = 3) const uint MAX_WGS = 1024;
layout (constant_id = 4) const uint MAX_WGS_BITS = 10;
layout (constant_id = 5) const uint FF_SIZE = 11008;
layout (constant_id = 6) const uint SOFTMAX_HEAD_PER_WAVEFRONT = 8;
layout (constant_id = 7) const uint BACKLOG_BITS = 7;
layout (constant_id = 8) const uint ROT_BITS = 7;
layout (constant_id = 9) const uint ROT = 128;
layout (constant_id = 10) const uint MATMUL_DIM_ROW_PER_WAVEFRONT = 8;
layout (constant_id = 11) const uint MATMUL_DIM_ROW_WORKER_COUNT = 128;
layout (constant_id = 12) const uint MATMUL_DIM_ROW_WORKER_COUNT_LOG2 = 7;
layout (constant_id = 13) const uint MATMUL_DIM_Q4_BLOCK_COUNT_PER_WORKER = 1;
layout (constant_id = 14) const uint MATMUL_DIM_Q4_BLOCKS_PER_ROW = 128;
layout (constant_id = 15) const uint MATMUL_FF_ROW_PER_WAVEFRONT = 8;
layout (constant_id = 16) const uint MATMUL_FF_ROW_WORKER_COUNT = 128;
layout (constant_id = 17) const uint MATMUL_FF_ROW_WORKER_COUNT_LOG2 = 7;
layout (constant_id = 18) const uint MATMUL_FF_Q4_BLOCK_COUNT_PER_WORKER = 3;
layout (constant_id = 19) const uint MATMUL_FF_Q4_BLOCKS_PER_ROW = 344;

#define HEAD_COUNT_CID 0
#define QUARTERROT_CID 1
#define BACKLOG_CID 2
#define MAX_WGS_CID 3
#define MAX_WGS_BITS_CID 4
#define FF_SIZE_CID 5
#define SOFTMAX_HEAD_PER_WAVEFRONT_CID 6
#define BACKLOG_BITS_CID 7
#define ROT_BITS_CID 8
#define ROT_CID 9
#define MATMUL_DIM_ROW_PER_WAVEFRONT_CID 10
#define MATMUL_DIM_ROW_WORKER_COUNT_CID 11
#define MATMUL_DIM_ROW_WORKER_COUNT_LOG2_CID 12
#define MATMUL_DIM_Q4_BLOCK_COUNT_PER_WORKER_CID 13
#define MATMUL_DIM_Q4_BLOCKS_PER_ROW_CID 14
#define MATMUL_FF_ROW_PER_WAVEFRONT_CID 15
#define MATMUL_FF_ROW_WORKER_COUNT_CID 16
#define MATMUL_FF_ROW_WORKER_COUNT_LOG2_CID 17
#define MATMUL_FF_Q4_BLOCK_COUNT_PER_WORKER_CID 18
#define MATMUL_FF_Q4_BLOCKS_PER_ROW_CID 19

#else

#include "constants.glsl"

#endif

#define DIM (ROT * HEAD_COUNT)

#ifdef LOCAL_SUM_BITS

#ifndef LOCAL_SUM_VEC4
shared float sum_buffer[MAX_WGS];
float local_sum(const uint subgroup_id, const uint self_id, const float element) {
    sum_buffer[(subgroup_id << LOCAL_SUM_BITS) + self_id] = element;
    barrier();
    uint curmax = (1 << LOCAL_SUM_BITS);
    [[unroll]] for (int j = 0; j < LOCAL_SUM_BITS; j++) {
        curmax >>= 1;
        if (self_id < curmax) {
            sum_buffer[(subgroup_id << LOCAL_SUM_BITS) + self_id] += sum_buffer[(subgroup_id << LOCAL_SUM_BITS) + curmax + self_id];
        }
        barrier();
    }
    return sum_buffer[subgroup_id << LOCAL_SUM_BITS];
}
#else
shared vec4 sum_buffer[MAX_WGS];
vec4 local_sum(const uint subgroup_id, const uint self_id, const vec4 element) {
    sum_buffer[(subgroup_id << LOCAL_SUM_BITS) + self_id] = element;
    barrier();
    uint curmax = (1 << LOCAL_SUM_BITS);
    [[unroll]] for (int j = 0; j < LOCAL_SUM_BITS; j++) {
        curmax >>= 1;
        if (self_id < curmax) {
            sum_buffer[(subgroup_id << LOCAL_SUM_BITS) + self_id] += sum_buffer[(subgroup_id << LOCAL_SUM_BITS) + curmax + self_id];
        }
        barrier();
    }
    return sum_buffer[subgroup_id << LOCAL_SUM_BITS];
}
#endif

#endif
