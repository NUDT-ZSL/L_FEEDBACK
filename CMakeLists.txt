cmake_minimum_required(VERSION 3.20)

project(uid166_d3d11_scene LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(UID166_BUILD_TESTS "Build deterministic core tests" ON)
option(UID166_BUILD_WIN32 "Build the native DirectX 11 application" ON)

add_library(uid166_core
    src/core/error.cpp
    src/core/json.cpp
    src/core/scene.cpp
    src/core/animation.cpp
    src/core/bvh.cpp
    src/core/culling.cpp
    src/core/shadows.cpp
    src/core/adaptive.cpp
    src/core/command_recorder.cpp
    src/core/benchmark.cpp
)

target_include_directories(uid166_core PUBLIC src)
target_compile_definitions(uid166_core PUBLIC UID166_CORE_VERSION="0.1.0")

if(MSVC)
    target_compile_options(uid166_core PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(uid166_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(UID166_BUILD_WIN32 AND WIN32)
    add_executable(uid166
        src/win/main.cpp
        src/win/d3d11_renderer.cpp
    )
    target_link_libraries(uid166 PRIVATE
        uid166_core
        d3d11
        dxgi
        d3dcompiler
        user32
        ole32
    )
    target_compile_definitions(uid166 PRIVATE UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
    if(MSVC)
        target_compile_options(uid166 PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(uid166 PRIVATE -Wall -Wextra -Wpedantic)
        if(MINGW)
            target_link_options(uid166 PRIVATE -static-libgcc -static-libstdc++ -static)
        endif()
    endif()
elseif(UID166_BUILD_WIN32)
    add_executable(uid166 src/posix_main.cpp)
    target_link_libraries(uid166 PRIVATE uid166_core)
endif()

if(UID166_BUILD_TESTS)
    enable_testing()
    add_executable(uid166_core_tests tests/core_tests.cpp)
    target_link_libraries(uid166_core_tests PRIVATE uid166_core)
    if(MINGW)
        target_link_options(uid166_core_tests PRIVATE -static-libgcc -static-libstdc++ -static)
    endif()
    add_test(NAME uid166_core_tests COMMAND uid166_core_tests)
endif()

