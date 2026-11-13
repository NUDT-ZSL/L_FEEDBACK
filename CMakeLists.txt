cmake_minimum_required(VERSION 3.20)

project(skinbake VERSION 1.0.0 LANGUAGES CXX)

option(SKINBAKE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(SKINBAKE_BUILD_TESTS "Build the unit and integration tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_library(skinbake_core
    src/curve.cpp
    src/error.cpp
    src/json.cpp
    src/scene.cpp
    src/snapshot.cpp
    src/state_machine.cpp
    src/bake.cpp
    src/scenarios.cpp
    src/verify.cpp
    src/fs.cpp
)

target_include_directories(skinbake_core PUBLIC include)
target_link_libraries(skinbake_core PUBLIC Threads::Threads)

add_executable(skinbake src/main.cpp)
target_link_libraries(skinbake PRIVATE skinbake_core)

if(MSVC)
    target_compile_options(skinbake_core PRIVATE /W4 /fp:strict /permissive-)
    target_compile_options(skinbake PRIVATE /W4 /fp:strict /permissive-)
    if(SKINBAKE_WARNINGS_AS_ERRORS)
        target_compile_options(skinbake_core PRIVATE /WX)
        target_compile_options(skinbake PRIVATE /WX)
    endif()
else()
    target_compile_options(skinbake_core PRIVATE
        -Wall -Wextra -Wpedantic -ffp-contract=off -fno-fast-math)
    target_compile_options(skinbake PRIVATE
        -Wall -Wextra -Wpedantic -ffp-contract=off -fno-fast-math)
    if(SKINBAKE_WARNINGS_AS_ERRORS)
        target_compile_options(skinbake_core PRIVATE -Werror)
        target_compile_options(skinbake PRIVATE -Werror)
    endif()
endif()

include(CTest)

if(SKINBAKE_BUILD_TESTS AND BUILD_TESTING)
    add_executable(skinbake_unit_tests tests/unit_tests.cpp)
    target_link_libraries(skinbake_unit_tests PRIVATE skinbake_core)
    if(MSVC)
        target_compile_options(skinbake_unit_tests PRIVATE /W4 /fp:strict)
    else()
        target_compile_options(skinbake_unit_tests PRIVATE
            -Wall -Wextra -Wpedantic -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME skinbake_unit COMMAND skinbake_unit_tests)

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME skinbake_integration
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_tests.py
                    --baker $<TARGET_FILE:skinbake>
                    --work ${CMAKE_CURRENT_BINARY_DIR}/integration-work
        )
    endif()
endif()

