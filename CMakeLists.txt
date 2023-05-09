cmake_minimum_required(VERSION 3.18)
project(vulkan_llama)

find_package(Vulkan REQUIRED)
if(NOT DEFINED RUNTIME_SHADER_BUILD)
    set(RUNTIME_SHADER_BUILD 0)
endif ()

if (RUNTIME_SHADER_BUILD)
    find_package(Threads REQUIRED)
    find_package(glslang REQUIRED)
endif ()

set(CMAKE_CXX_STANDARD 20)
add_executable(vulkan_llama main.cpp ggml_file.cpp ggml_file.h llava_context.cpp llava_context.h llava_pipeline.cpp llava_pipeline.h llava_command.cpp llava_command.h types.h llava_buffer.cpp llava_buffer.h llava_layer.cpp llava_layer.h utils.h utils.cpp llava_device_memory.cpp llava_device_memory.h)

if (RUNTIME_SHADER_BUILD)
    add_compile_definitions(RUNTIME_BUILD_ENABLED)
    target_link_libraries(vulkan_llama PRIVATE ${Vulkan_LIBRARIES} glslang::glslang glslang::SPIRV)
else ()
    target_link_libraries(vulkan_llama PRIVATE ${Vulkan_LIBRARIES})
endif ()

# add_link_options(-fsanitizer=address)
# add_compile_options(-fsanitizer=address)
