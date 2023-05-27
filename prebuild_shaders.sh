#!/bin/bash

mkdir -p prebuilt_shaders
rm -f prebuilt_shaders/*.spv
glslangValidator --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/copy_to_cache.comp -o prebuilt_shaders/copy_to_cache.spv
glslangValidator --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/kqv_matching.comp -o prebuilt_shaders/kqv_matching.spv
glslangValidator --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/matmul_ff.comp -o prebuilt_shaders/matmul_ff.spv
glslangValidator --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/matmul_dim.comp -o prebuilt_shaders/matmul_dim.spv
glslangValidator --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/mhsa.comp -o prebuilt_shaders/mhsa.spv
glslangValidator --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/normalize.comp -o prebuilt_shaders/normalize.spv
glslangValidator --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/softmax.comp -o prebuilt_shaders/softmax.spv
glslangValidator --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/matmul_silu_ff.comp -o prebuilt_shaders/matmul_silu_ff.spv
glslangValidator --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/matmul_add_dim.comp -o prebuilt_shaders/matmul_add_dim.spv
glslangValidator --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/matmul_add_ff.comp -o prebuilt_shaders/matmul_add_ff.spv
