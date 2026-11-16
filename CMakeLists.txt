cmake_minimum_required(VERSION 3.20)

project(skin_batch_renderer VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SKINRENDER_USE_GLFW "Build the optional GLFW desktop backend" ON)
option(SKINRENDER_BUILD_TESTS "Register the deterministic validation test" ON)

find_package(glm CONFIG QUIET)
find_package(nlohmann_json CONFIG QUIET)

if(SKINRENDER_USE_GLFW)
    find_package(glfw3 3.3 CONFIG QUIET)
endif()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/shaders/lut.vert" LUT_VERTEX_SHADER)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/shaders/lut.frag" LUT_FRAGMENT_SHADER)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/shaders/surface.vert" SURFACE_VERTEX_SHADER)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/shaders/surface.frag" SURFACE_FRAGMENT_SHADER)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders_embedded.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/shaders_embedded.hpp"
    @ONLY
)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/lut.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/lut.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/surface.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/surface.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders_embedded.hpp.in"
)

add_executable(skin_renderer
    src/main.cpp
    src/config.cpp
    src/gl_backend.cpp
    src/gl_resources.cpp
    src/logger.cpp
    src/ppm.cpp
    src/renderer.cpp
    third_party/glad/src/gl.c
)

target_include_directories(skin_renderer
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/glad/include"
)

target_compile_definitions(skin_renderer
    PRIVATE
        SKINRENDER_VERSION="${PROJECT_VERSION}"
)

if(glm_FOUND)
    target_link_libraries(skin_renderer PRIVATE glm::glm)
endif()

if(nlohmann_json_FOUND)
    target_link_libraries(skin_renderer PRIVATE nlohmann_json::nlohmann_json)
endif()

if(glfw3_FOUND)
    target_compile_definitions(skin_renderer PRIVATE SKINRENDER_HAVE_GLFW=1)
    target_link_libraries(skin_renderer PRIVATE glfw)
elseif(WIN32)
    message(FATAL_ERROR
        "GLFW is required on Windows. Install it with vcpkg and configure CMake "
        "with the vcpkg toolchain file.")
endif()

if(UNIX AND NOT APPLE)
    find_library(EGL_LIBRARY NAMES EGL libEGL.so.1)
    if(EGL_LIBRARY)
        target_compile_definitions(skin_renderer PRIVATE SKINRENDER_HAVE_EGL=1)
        target_link_libraries(skin_renderer PRIVATE "${EGL_LIBRARY}" ${CMAKE_DL_LIBS})
        find_library(GL_DISPATCH_LIBRARY NAMES GLdispatch libGLdispatch.so.0)
        if(GL_DISPATCH_LIBRARY)
            target_link_libraries(skin_renderer PRIVATE "${GL_DISPATCH_LIBRARY}")
        endif()
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(skin_renderer PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(skin_renderer PRIVATE /W4 /permissive- /EHsc)
endif()

if(SKINRENDER_BUILD_TESTS)
    enable_testing()
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME batch_renderer_validation
            COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate.py"
                --exe "$<TARGET_FILE:skin_renderer>"
        )
    endif()
endif()

