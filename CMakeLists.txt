cmake_minimum_required(VERSION 3.20)

project(character_baker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BAKER_BUILD_TESTS "Build the focused C++ tests" ON)

add_library(baker_core
    src/atomic_io.cpp
    src/async_writer.cpp
    src/interrupt.cpp
    src/render.cpp
    src/scene.cpp
    src/scene_generator.cpp
    src/snapshot.cpp
)

target_include_directories(baker_core PUBLIC include)

if(MSVC)
    target_compile_options(baker_core PRIVATE /W4 /permissive- /EHsc)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(baker_core PRIVATE /fp:strict)
    else()
        target_compile_options(baker_core PRIVATE /fp:precise
            /clang:-ffp-contract=off)
    endif()
else()
    target_compile_options(baker_core PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow
        -fno-fast-math -ffp-contract=off
    )
endif()

if(WIN32)
    target_compile_definitions(baker_core PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

find_package(Threads REQUIRED)
target_link_libraries(baker_core PUBLIC Threads::Threads)

add_executable(baker src/main.cpp)
target_link_libraries(baker PRIVATE baker_core)

add_executable(scene_gen src/scene_gen_main.cpp)
target_link_libraries(scene_gen PRIVATE baker_core)

if(BAKER_BUILD_TESTS)
    enable_testing()
    add_executable(baker_tests tests/baker_tests.cpp)
    target_link_libraries(baker_tests PRIVATE baker_core)
    add_test(NAME baker_unit COMMAND baker_tests)

    find_package(Python3 COMPONENTS Interpreter)
    if(UNIX AND Python3_Interpreter_FOUND)
        add_test(
            NAME baker_integration
            COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_integration.sh"
                    "$<TARGET_FILE:baker>"
                    "$<TARGET_FILE:scene_gen>"
                    "${CMAKE_CURRENT_BINARY_DIR}/integration-work"
        )
        set_tests_properties(baker_integration PROPERTIES TIMEOUT 120)

        add_test(
            NAME baker_kill_recovery
            COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/kill_recovery.sh"
                    "$<TARGET_FILE:baker>"
                    "$<TARGET_FILE:scene_gen>"
                    "${CMAKE_CURRENT_BINARY_DIR}/kill-work"
        )
        set_tests_properties(baker_kill_recovery PROPERTIES TIMEOUT 120)
    endif()
endif()
