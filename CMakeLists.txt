cmake_minimum_required(VERSION 3.20)

project(dungeon_deferred_validator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(dungeon_deferred
    src/main.cpp
    src/scene.cpp
    src/renderer.cpp
    src/shadow_cache.cpp
)

target_include_directories(dungeon_deferred PRIVATE src)

if(MSVC)
    target_compile_options(dungeon_deferred PRIVATE /W4 /permissive- /fp:strict /utf-8)
else()
    target_compile_options(dungeon_deferred PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_libraries(dungeon_deferred PRIVATE m)
endif()

find_package(Python3 COMPONENTS Interpreter)

if(Python3_Interpreter_FOUND)
    enable_testing()
    add_test(
        NAME deterministic_acceptance
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/acceptance.py
                $<TARGET_FILE:dungeon_deferred>
    )
    set_tests_properties(deterministic_acceptance PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

add_custom_target(run_acceptance
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/acceptance.py
            $<TARGET_FILE:dungeon_deferred>
    DEPENDS dungeon_deferred
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL
)
