# vulkan-llama

LLM evaluator based on Vulkan

This project is mostly based on [Georgi Gerganov's llama.cpp](https://github.com/ggerganov/llama.cpp).

## Working
* Tokenizer
* GGML parsing and mapping for q4_0 models
* Basic evaluation of the 7G model

## TODO
* Support for memory swapping and dynamic memory types
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
