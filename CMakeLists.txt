cmake_minimum_required(VERSION 3.25)
project(vulkan_llama)

set(CMAKE_CXX_STANDARD 17)
add_executable(vulkan_llama main.cpp ggml_file.cpp ggml_file.h llava_context.cpp llava_context.h llava_pipeline.cpp llava_pipeline.h llava_command.cpp llava_command.h types.h)
target_link_libraries(vulkan_llama PRIVATE vulkan)
add_link_options(-fsanitizer=address)
add_compile_options(-fsanitizer=address)
