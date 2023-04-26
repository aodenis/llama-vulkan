Llama for Vulkan's compute pipelines

### Working
* Tokenizer
* GGML parsing and mapping
* Vulkan initialization

### TODO
* Everything else

## Setup

```
cmake --no-warn-unused-cli -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Debug -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++ -B build -G Ninja
```

```
glslc -fshader-stage=comp shaders/q4_0_split.glsl -o build/built_shaders/q4_0_split.spirv
```


## Build

```
cmake --build build --config Debug --target all
build/vulkan_llama
```
