cmake_minimum_required(VERSION 3.20)

project(pcss_shadow_renderer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
if(UNIX AND NOT APPLE)
    set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)
endif()

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.8
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 0.9.9.8
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(glfw glm)

find_package(OpenGL REQUIRED)

add_library(glad STATIC third_party/glad/src/gl.c)
target_include_directories(glad PUBLIC third_party/glad/include)

add_executable(pcss_shadow_renderer
    src/main.cpp
    src/camera.cpp
    src/gl_utils.cpp
    src/gpu_timer.cpp
    src/mesh.cpp
    src/scene.cpp
    src/shadow_renderer.cpp
    src/text_overlay.cpp
)

target_include_directories(pcss_shadow_renderer PRIVATE
    src
    third_party/stb
)

target_link_libraries(pcss_shadow_renderer PRIVATE
    glad
    glfw
    glm::glm
    OpenGL::GL
)

if(MSVC)
    target_compile_options(pcss_shadow_renderer PRIVATE /W4 /permissive-)
else()
    target_compile_options(pcss_shadow_renderer PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(pcss_shadow_renderer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

