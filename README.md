Llama for Vulkan's compute pipelines

### Working
* Tokenizer
* GGML parsing and mapping
* Vulkan initialization
* Basic evaluation of the 7G model

### TODO
* Support for memory swapping and dynamic memory types
* Faster initialization
* Proper interface
* Support for more models
* Support for more GPUs
* Optimizations

## Setup

```
cmake --no-warn-unused-cli -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Debug -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++ -B build -G Ninja
```

Building prebuilt shaders:
```
glslc -fshader-stage=comp shaders/add_f32_1024.glsl -o prebuilt_shaders/add_f32_1024.spv
```


## Build

```
cmake --build build --config Debug --target all
build/vulkan_llama
```
