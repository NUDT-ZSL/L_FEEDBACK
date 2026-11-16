cmake_minimum_required(VERSION 3.20)

project(tilegen VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TILEGEN_BUILD_TESTS "Build deterministic validation tests" ON)
option(TILEGEN_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

add_library(tilegen_core
    src/json.cpp
    src/png.cpp
    src/scene.cpp
    src/render.cpp
    src/pipeline.cpp
)

target_include_directories(tilegen_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(MSVC)
    target_compile_options(tilegen_core PRIVATE /W4 /permissive-)
    if(TILEGEN_WARNINGS_AS_ERRORS)
        target_compile_options(tilegen_core PRIVATE /WX)
    endif()
else()
    target_compile_options(tilegen_core PRIVATE -Wall -Wextra -Wpedantic)
    if(TILEGEN_WARNINGS_AS_ERRORS)
        target_compile_options(tilegen_core PRIVATE -Werror)
    endif()
endif()

add_executable(tilegen src/main.cpp)
target_link_libraries(tilegen PRIVATE tilegen_core)

install(TARGETS tilegen RUNTIME DESTINATION bin)

if(TILEGEN_BUILD_TESTS)
    enable_testing()
    add_executable(tilegen_tests tests/test_main.cpp)
    target_link_libraries(tilegen_tests PRIVATE tilegen_core)
    target_compile_definitions(tilegen_tests PRIVATE
        TILEGEN_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME tilegen_unit COMMAND tilegen_tests)
endif()
