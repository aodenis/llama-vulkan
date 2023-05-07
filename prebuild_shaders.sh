#!/bin/bash

mkdir prebuilt_shaders
rm -f prebuilt_shaders/*.spv
glslc -fshader-stage=comp -DUSE_SPEVAR=1 shaders/add_logits.glsl -o prebuilt_shaders/add_logits.spv
glslc -fshader-stage=comp -DUSE_SPEVAR=1 shaders/copy_to_cache.glsl -o prebuilt_shaders/copy_to_cache.spv
glslc -fshader-stage=comp -DUSE_SPEVAR=1 shaders/kqv_matching.glsl -o prebuilt_shaders/kqv_matching.spv
glslc -fshader-stage=comp -DUSE_SPEVAR=1 shaders/matmul_ff.glsl -o prebuilt_shaders/matmul_ff.spv
glslc -fshader-stage=comp -DUSE_SPEVAR=1 shaders/matmul_dim.glsl -o prebuilt_shaders/matmul_dim.spv
glslc -fshader-stage=comp -DUSE_SPEVAR=1 shaders/mhsa.glsl -o prebuilt_shaders/mhsa.spv
glslc -fshader-stage=comp -DUSE_SPEVAR=1 shaders/multiply.glsl -o prebuilt_shaders/multiply.spv
glslc -fshader-stage=comp -DUSE_SPEVAR=1 shaders/normalize.glsl -o prebuilt_shaders/normalize.spv
glslc -fshader-stage=comp -DUSE_SPEVAR=1 shaders/silu.glsl -o prebuilt_shaders/silu.spv
glslc -fshader-stage=comp -DUSE_SPEVAR=1 shaders/rope.glsl -o prebuilt_shaders/rope.spv
glslc -fshader-stage=comp -DUSE_SPEVAR=1 shaders/softmax.glsl -o prebuilt_shaders/softmax.spv
