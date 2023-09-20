# vulkan-llama

LLM evaluator based on Vulkan

This project is mostly based on [Georgi Gerganov's llama.cpp](https://github.com/ggerganov/llama.cpp).
It supports both using prebuilt SpirV shaders and building them at runtime. The latter option is disabled by default
as it requires extra libraries and does not produce faster shaders.

Vulkan 1.2 is used and no extension is required.

## Currently working

* Tokenizer
* GGML parsing and mapping for q4_0, q8_0 models
* Evaluation of 7B models

## Known issues

* Evaluation 13B models fail due to a NaN value corrupting the shader values. This is being fixed
* Threading (for server mode) uses mutex and may deadlock
* Too many asserts can fire

## Setup

Building prebuilt shaders:
```bash
./prebuild_shaders.py
```


## Build

```bash
mkdir build && cd build
cmake .. && make -j
cd .. && ./build/vulkan_llama --help
```

## Testing hardware

* AMD Ryzen 7 6800U with Radeon Graphics (AMD Radeon 680M)
* AMD Radeon RX 6900 XT
