cmake_minimum_required(VERSION 3.20)
project(deferred_fog LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DEFERRED_ENABLE_GPU "Build the OpenGL 4.5 real-time renderer" ON)
option(DEFERRED_BUILD_TESTS "Build deterministic core tests" ON)
set(GLFW_ROOT "" CACHE PATH "Optional local GLFW installation prefix")

add_library(deferred_core
    src/config.cpp
    src/scene.cpp
    src/sorting.cpp
    src/lights.cpp
    src/noise.cpp
    src/image_io.cpp
    src/renderer_cpu.cpp
    src/offline.cpp
)
target_include_directories(deferred_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
target_compile_features(deferred_core PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(deferred_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(deferred_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(deferred_fog src/main.cpp)
target_link_libraries(deferred_fog PRIVATE deferred_core)
target_compile_definitions(deferred_fog PRIVATE
    DF_DEFAULT_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/render.json")

if(DEFERRED_ENABLE_GPU)
    find_package(glfw3 3.3 QUIET CONFIG)
    if(NOT glfw3_FOUND)
        find_package(glfw3 QUIET MODULE)
    endif()
    if(NOT glfw3_FOUND)
        find_path(GLFW_INCLUDE_DIR GLFW/glfw3.h
            HINTS "${GLFW_ROOT}/include"
                  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/glfw/include")
        find_library(GLFW_LIBRARY NAMES glfw3 glfw
            HINTS "${GLFW_ROOT}/lib" "${GLFW_ROOT}/lib64")
        if(GLFW_INCLUDE_DIR AND GLFW_LIBRARY)
            add_library(glfw INTERFACE)
            target_include_directories(glfw INTERFACE "${GLFW_INCLUDE_DIR}")
            target_link_libraries(glfw INTERFACE "${GLFW_LIBRARY}")
            set(glfw3_FOUND TRUE)
        endif()
    endif()
    if(glfw3_FOUND)
        find_package(OpenGL REQUIRED)
        add_library(deferred_glad STATIC third_party/glad/src/gl.c)
        target_include_directories(deferred_glad PUBLIC third_party/glad/include)
        add_library(deferred_gpu STATIC src/gpu_renderer.cpp)
        target_link_libraries(deferred_gpu PUBLIC deferred_core deferred_glad glfw OpenGL::GL)
        target_compile_definitions(deferred_gpu PUBLIC DEFERRED_HAS_GPU=1)
        target_compile_definitions(deferred_gpu PRIVATE
            DF_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders")
        target_link_libraries(deferred_fog PRIVATE deferred_gpu)
        message(STATUS "OpenGL renderer enabled with GLFW")
    else()
        message(WARNING "GLFW not found; building CPU reference backend only. "
                        "Use vcpkg or set glfw3_DIR to enable the OpenGL renderer.")
    endif()
endif()

if(DEFERRED_BUILD_TESTS)
    enable_testing()
    add_executable(deferred_core_tests tests/test_core.cpp)
    target_link_libraries(deferred_core_tests PRIVATE deferred_core)
    target_compile_definitions(deferred_core_tests PRIVATE
        DF_TEST_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config")
    add_test(NAME core_tests COMMAND deferred_core_tests)
    set_tests_properties(core_tests PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    add_executable(deferred_verify tools/verify_artifacts.cpp)
    target_link_libraries(deferred_verify PRIVATE deferred_core)
endif()

