cmake_minimum_required(VERSION 3.20)

project(cpu_terrain LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cpu_terrain_core
    src/config.cpp
    src/heightfield.cpp
    src/image.cpp
    src/json.cpp
    src/renderer.cpp
    src/terrain.cpp
    src/window.cpp
)

target_include_directories(cpu_terrain_core PUBLIC src)

if(MSVC)
    target_compile_options(cpu_terrain_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(cpu_terrain_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(terrain_renderer src/main.cpp)
target_link_libraries(terrain_renderer PRIVATE cpu_terrain_core)

if(WIN32)
    target_link_libraries(cpu_terrain_core PUBLIC user32 gdi32)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(cpu_terrain_core PUBLIC stdc++fs)
endif()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME end_to_end_verification
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify.py
                    $<TARGET_FILE:terrain_renderer>
                    ${CMAKE_CURRENT_SOURCE_DIR}/configs/default.json
                    ${CMAKE_CURRENT_BINARY_DIR}/test-output
        )
    endif()
endif()
