# vulkan-llama

LLM evaluator based on Vulkan

This project is mostly based on [Georgi Gerganov's llama.cpp](https://github.com/ggerganov/llama.cpp).
It supports both using prebuilt SpirV shaders and building them at runtime. The latter option is disabled by default
as it requires extra libraries and does not produce faster shaders.

Vulkan 1.2 is used and no extension is required.

## Currently working
* Tokenizer
* GGML parsing and mapping for q4_0 models
* Evaluation of 7B and 13B models

## TODO
* Support for memory swapping
* Faster initialization
* Proper interface
* Support for more models
* Support for more GPUs
* Optimizations

## Setup

Building prebuilt shaders:
```bash
./prebuild_shaders.sh
```


## Build

```bash
mkdir build && cd build
cmake .. && make -j
cd .. && ./build/vulkan_llama --help
```

## Testing hardware

* AMD Ryzen 7 6800U with Radeon Graphics (AMD Radeon 680M)
