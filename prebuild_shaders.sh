#!/bin/bash

mkdir -p prebuilt_shaders
rm -f prebuilt_shaders/*.spv
glslangValidator  -S comp --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/copy_to_cache.glsl -o prebuilt_shaders/copy_to_cache.spv
glslangValidator  -S comp --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/kqv_matching.glsl -o prebuilt_shaders/kqv_matching.spv
glslangValidator  -S comp --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/matmul_ff.glsl -o prebuilt_shaders/matmul_ff.spv
glslangValidator  -S comp --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/matmul_dim.glsl -o prebuilt_shaders/matmul_dim.spv
glslangValidator  -S comp --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/mhsa.glsl -o prebuilt_shaders/mhsa.spv
glslangValidator  -S comp --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/normalize.glsl -o prebuilt_shaders/normalize.spv
glslangValidator  -S comp --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/rope.glsl -o prebuilt_shaders/rope.spv
glslangValidator  -S comp --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/softmax.glsl -o prebuilt_shaders/softmax.spv
glslangValidator  -S comp --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/matmul_silu_ff.glsl -o prebuilt_shaders/matmul_silu_ff.spv
glslangValidator  -S comp --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/matmul_add_dim.glsl -o prebuilt_shaders/matmul_add_dim.spv
glslangValidator  -S comp --target-env vulkan1.2 -DUSE_SPEVAR=1 -e main shaders/matmul_add_ff.glsl -o prebuilt_shaders/matmul_add_ff.spv
