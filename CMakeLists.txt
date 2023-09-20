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
# add_link_options(-fsanitize=undefined)
# add_compile_options(-fsanitize=undefined)

if (EMBEDDED_SPV)
    # A generated blob generation command
    file(GLOB SHADER_SOURCES ${PROJECT_SOURCE_DIR}/shaders/*.*)
    add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/generated/packed_spv.c
                       COMMAND python3 ${CMAKE_SOURCE_DIR}/prebuild_shaders.py
                       DEPENDS ${SHADER_SOURCES}
                       WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set(EXTRA_FILE ${CMAKE_SOURCE_DIR}/generated/packed_spv.c)
    add_compile_definitions(EMBEDDED_SPV)
else ()
    set(EXTRA_FILE )
endif ()


add_executable(vulkan_llama main.cpp ggml_file.cpp ggml_file.h llava_context.cpp llava_context.h llava_pipeline.cpp llava_pipeline.h types.h llava_buffer.cpp llava_buffer.h llava_layer.cpp llava_layer.h utils.h utils.cpp llava_device_memory.cpp llava_device_memory.h llava_command_buffer.cpp llava_command_buffer.h llava_session.cpp llava_session.h server/server.cpp server/server.h server/client.h server/session_wrapper.h server/client.cpp server/session_wrapper.cpp server/types.h ${EXTRA_FILE}
        llava_layer_session_data.h
        llava_layer_session_data.cpp)

if (RUNTIME_SHADER_BUILD)
    add_compile_definitions(RUNTIME_BUILD_ENABLED)
    target_link_libraries(vulkan_llama PRIVATE ${Vulkan_LIBRARIES} glslang::glslang glslang::SPIRV)
else ()
    target_link_libraries(vulkan_llama PRIVATE ${Vulkan_LIBRARIES})
endif ()
