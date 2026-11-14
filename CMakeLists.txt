cmake_minimum_required(VERSION 3.20)

project(deferred_snapshot LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.local-deps/usr/include/GL/gl.h")
    list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.local-deps/usr")
    set(OPENGL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.local-deps/usr/include")
    set(OPENGL_opengl_LIBRARY
        "${CMAKE_CURRENT_SOURCE_DIR}/.local-deps/usr/lib/x86_64-linux-gnu/libGL.so")
    set(OPENGL_glx_LIBRARY
        "${CMAKE_CURRENT_SOURCE_DIR}/.local-deps/usr/lib/x86_64-linux-gnu/libGLX.so")
    set(ZLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.local-deps/usr/include")
    set(ZLIB_LIBRARY
        "${CMAKE_CURRENT_SOURCE_DIR}/.local-deps/usr/lib/x86_64-linux-gnu/libz.so")
    set(ZLIB_LIBRARY_RELEASE "${ZLIB_LIBRARY}")
    set(DEFERRED_LOCAL_RPATH_LINK "/usr/lib/x86_64-linux-gnu")
endif()

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(PNG REQUIRED)

add_library(glad STATIC third_party/glad/src/glad.c)
target_include_directories(glad PUBLIC third_party/glad/include)
target_compile_definitions(glad PRIVATE GLAD_GL_IMPLEMENTATION)

add_executable(deferred_renderer
    src/main.cpp
    src/cli.cpp
    src/scene.cpp
    src/snapshot.cpp
    src/renderer.cpp
    src/gl_resources.cpp
    src/image_io.cpp
    src/math.cpp
)

target_include_directories(deferred_renderer PRIVATE
    include
    third_party
    third_party/glad/include
)

target_link_libraries(deferred_renderer PRIVATE
    glad
    glfw
    PNG::PNG
    OpenGL::GL
    ${CMAKE_DL_LIBS}
)

if(DEFERRED_LOCAL_RPATH_LINK)
    target_link_options(deferred_renderer PRIVATE
        "-Wl,-rpath-link,${DEFERRED_LOCAL_RPATH_LINK}")
endif()

if(MSVC)
    target_compile_options(deferred_renderer PRIVATE /W4 /permissive-)
else()
    target_compile_options(deferred_renderer PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(ENABLE_ASAN)
    if(MSVC)
        target_compile_options(deferred_renderer PRIVATE /fsanitize=address)
        target_link_options(deferred_renderer PRIVATE /fsanitize=address)
    else()
        target_compile_options(deferred_renderer PRIVATE -fsanitize=address -fno-omit-frame-pointer)
        target_link_options(deferred_renderer PRIVATE -fsanitize=address)
    endif()
endif()

if(WIN32)
    add_custom_command(TARGET deferred_renderer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:deferred_renderer> $<TARGET_FILE_DIR:deferred_renderer>
        COMMAND_EXPAND_LISTS
    )
endif()
