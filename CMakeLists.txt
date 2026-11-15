cmake_minimum_required(VERSION 3.20)

project(SkeletalMaterialSync VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

find_package(glfw3 3.3 CONFIG QUIET)
if(TARGET glfw)
    set(DEMO_GLFW_TARGET glfw)
else()
    find_path(GLFW_INCLUDE_DIR NAMES GLFW/glfw3.h)
    find_library(GLFW_LIBRARY NAMES glfw3 glfw)
    if(NOT GLFW_INCLUDE_DIR OR NOT GLFW_LIBRARY)
        message(FATAL_ERROR
            "GLFW 3.3+ was not found. Install it with vcpkg and pass "
            "-DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake, "
            "or set CMAKE_PREFIX_PATH.")
    endif()
    add_library(demo_glfw UNKNOWN IMPORTED)
    set_target_properties(demo_glfw PROPERTIES
        IMPORTED_LOCATION "${GLFW_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GLFW_INCLUDE_DIR}")
    set(DEMO_GLFW_TARGET demo_glfw)
endif()

find_package(glm CONFIG QUIET)
if(TARGET glm::glm)
    set(DEMO_GLM_TARGET glm::glm)
elseif(TARGET glm::glm-header-only)
    set(DEMO_GLM_TARGET glm::glm-header-only)
else()
    find_path(GLM_INCLUDE_DIR NAMES glm/glm.hpp)
    if(NOT GLM_INCLUDE_DIR)
        message(FATAL_ERROR
            "GLM 0.9.9+ was not found. Install it with vcpkg or set "
            "CMAKE_PREFIX_PATH.")
    endif()
    add_library(demo_glm INTERFACE)
    target_include_directories(demo_glm INTERFACE "${GLM_INCLUDE_DIR}")
    set(DEMO_GLM_TARGET demo_glm)
endif()

function(demo_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

add_library(demo_scene STATIC
    src/scene.cpp
    src/frame_data.cpp
)
target_include_directories(demo_scene PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(demo_scene PUBLIC ${DEMO_GLM_TARGET})
demo_enable_warnings(demo_scene)

add_library(demo_render STATIC
    src/gl_loader.cpp
    src/renderer.cpp
)
target_link_libraries(demo_render PUBLIC demo_scene ${DEMO_GLFW_TARGET})
demo_enable_warnings(demo_render)

add_executable(skeletal_material_demo
    src/main.cpp
    src/application.cpp
)
target_link_libraries(skeletal_material_demo PRIVATE demo_render)
demo_enable_warnings(skeletal_material_demo)

if(WIN32)
    # The loader obtains GL 4.5 entry points through GLFW. opengl32 remains
    # linked for the WGL backend used by static GLFW builds.
    target_link_libraries(skeletal_material_demo PRIVATE opengl32)
endif()

if(BUILD_TESTING)
    add_executable(cpu_tests tests/cpu_tests.cpp)
    target_link_libraries(cpu_tests PRIVATE demo_scene)
    demo_enable_warnings(cpu_tests)
    add_test(NAME cpu_scene_contract COMMAND cpu_tests)

    option(DEMO_ENABLE_GPU_TESTS
        "Register tests that require a desktop OpenGL 4.5 context" OFF)
    if(DEMO_ENABLE_GPU_TESTS)
        add_test(NAME gpu_self_test
            COMMAND skeletal_material_demo --self-test)
        add_test(NAME gpu_shader_compile_fallback
            COMMAND skeletal_material_demo --self-test
                    --inject-shader-compile-failure)
        add_test(NAME gpu_shader_link_fallback
            COMMAND skeletal_material_demo --self-test
                    --inject-shader-link-failure)
        add_test(NAME gpu_vsync_off_fps_cap
            COMMAND skeletal_material_demo --benchmark-frames 120 --vsync-off)
        set_tests_properties(
            gpu_self_test
            gpu_shader_compile_fallback
            gpu_shader_link_fallback
            gpu_vsync_off_fps_cap
            PROPERTIES TIMEOUT 40)
    endif()
endif()
