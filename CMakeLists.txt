cmake_minimum_required(VERSION 3.25)
project(vulkan_llama)

set(CMAKE_CXX_STANDARD 17)
add_executable(vulkan_llama main.cpp ggml_file.cpp ggml_file.h context.cpp llava_context.h)
target_link_libraries(vulkan_llama PRIVATE vulkan)
