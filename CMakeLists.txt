cmake_minimum_required(VERSION 3.15)
project(canyon_rasterizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_executable(canyon_renderer
    src/core.cpp
    src/json.cpp
    src/scene.cpp
    src/renderer.cpp
    src/main.cpp
)

target_include_directories(canyon_renderer PRIVATE src)

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME canyon_acceptance
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tests/acceptance.py
                $<TARGET_FILE:canyon_renderer>
                ${CMAKE_SOURCE_DIR}/scenes
                ${CMAKE_BINARY_DIR}/acceptance_out
        )
    endif()
endif()
