cmake_minimum_required(VERSION 3.20)

project(VolumeShadowRenderer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan 1.3 REQUIRED)

find_program(VULKAN_GLSLC
    NAMES glslc
    HINTS ENV VULKAN_SDK
    PATH_SUFFIXES Bin bin
)
find_program(VULKAN_GLSLANG
    NAMES glslangValidator
    HINTS ENV VULKAN_SDK
    PATH_SUFFIXES Bin bin
)

if(VULKAN_GLSLC)
    set(SHADER_COMPILER "${VULKAN_GLSLC}")
    set(SHADER_COMPILER_KIND "glslc")
elseif(VULKAN_GLSLANG)
    set(SHADER_COMPILER "${VULKAN_GLSLANG}")
    set(SHADER_COMPILER_KIND "glslang")
else()
    message(FATAL_ERROR
        "No GLSL compiler found. Install the Vulkan SDK or set VULKAN_SDK; "
        "glslc/glslangValidator must be available.")
endif()

set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SHADER_OUTPUT_DIR}")

set(SHADER_OUTPUTS)
function(add_vulkan_shader SOURCE)
    get_filename_component(source_name "${SOURCE}" NAME)
    set(output "${SHADER_OUTPUT_DIR}/${source_name}.spv")
    if(SHADER_COMPILER_KIND STREQUAL "glslc")
        add_custom_command(
            OUTPUT "${output}"
            COMMAND "${SHADER_COMPILER}"
                --target-env=vulkan1.3
                -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
                -o "${output}"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SOURCE}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SOURCE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/common.glsl"
            COMMENT "Compiling ${SOURCE}"
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT "${output}"
            COMMAND "${SHADER_COMPILER}"
                -V
                --target-env vulkan1.3
                -I"${CMAKE_CURRENT_SOURCE_DIR}/shaders"
                -o "${output}"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SOURCE}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SOURCE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/common.glsl"
            COMMENT "Compiling ${SOURCE}"
            VERBATIM
        )
    endif()
    set(SHADER_OUTPUTS "${SHADER_OUTPUTS};${output}" PARENT_SCOPE)
endfunction()

add_vulkan_shader(shadow.vert)
add_vulkan_shader(gbuffer.vert)
add_vulkan_shader(gbuffer.frag)
add_vulkan_shader(fullscreen.vert)
add_vulkan_shader(composite.frag)
add_vulkan_shader(volume.comp)

add_custom_target(renderer_shaders ALL DEPENDS ${SHADER_OUTPUTS})

add_library(volume_renderer_core
    src/errors.cpp
    src/json.cpp
    src/hash.cpp
    src/png.cpp
    src/scene.cpp
    src/renderer.cpp
)
target_include_directories(volume_renderer_core PUBLIC src)
target_link_libraries(volume_renderer_core PUBLIC Vulkan::Vulkan)
target_compile_definitions(volume_renderer_core
    PRIVATE RENDERER_DEFAULT_SHADER_DIR="${SHADER_OUTPUT_DIR}"
)

if(MSVC)
    target_compile_options(volume_renderer_core PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(volume_renderer_core PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-missing-field-initializers
        -Wno-dangling-reference)
endif()

add_executable(volume_renderer src/main.cpp)
target_link_libraries(volume_renderer PRIVATE volume_renderer_core)
add_dependencies(volume_renderer renderer_shaders)

add_executable(scene_gen src/scene_gen.cpp)
target_link_libraries(scene_gen PRIVATE volume_renderer_core)

if(MSVC)
    target_compile_options(volume_renderer PRIVATE /W4 /permissive- /utf-8)
    target_compile_options(scene_gen PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(volume_renderer PRIVATE
        -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)
    target_compile_options(scene_gen PRIVATE
        -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)
endif()

install(TARGETS volume_renderer scene_gen RUNTIME DESTINATION bin)
install(DIRECTORY "${SHADER_OUTPUT_DIR}/" DESTINATION bin/shaders)

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_FOUND)
        add_test(
            NAME volume_renderer_acceptance
            COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_renderer.py"
                --renderer "$<TARGET_FILE:volume_renderer>"
                --scene-gen "$<TARGET_FILE:scene_gen>"
                --shader-dir "${SHADER_OUTPUT_DIR}"
                --work-dir "${CMAKE_BINARY_DIR}/test-work"
        )
        set_tests_properties(volume_renderer_acceptance PROPERTIES
            TIMEOUT 600
            LABELS "acceptance"
        )
    else()
        message(STATUS "Python 3 not found; acceptance CTest target is skipped")
    endif()
endif()
