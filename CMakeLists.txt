cmake_minimum_required(VERSION 3.25)

project(offline_character_rasterizer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SOFTRASTER_BUILD_TESTS "Build deterministic unit and integration tests" ON)

add_library(softraster_core
    src/error.cpp
    src/json.cpp
    src/math.cpp
    src/scene.cpp
    src/renderer.cpp
    src/sha256.cpp
    src/sample_scene.cpp
)

target_include_directories(softraster_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

if(MSVC)
    target_compile_options(softraster_core PRIVATE /W4 /permissive- /utf-8)
    target_compile_definitions(softraster_core PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    target_compile_options(softraster_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

add_executable(softraster src/main.cpp)
target_link_libraries(softraster PRIVATE softraster_core)

if(MSVC)
    target_compile_options(softraster PRIVATE /W4 /permissive- /utf-8)
    target_compile_definitions(softraster PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    target_compile_options(softraster PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

if(SOFTRASTER_BUILD_TESTS)
    enable_testing()
    add_executable(softraster_unit_tests tests/unit_tests.cpp)
    target_link_libraries(softraster_unit_tests PRIVATE softraster_core)
    if(MSVC)
        target_compile_options(softraster_unit_tests PRIVATE /W4 /permissive- /utf-8)
        target_compile_definitions(softraster_unit_tests PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
    else()
        target_compile_options(softraster_unit_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()
    add_test(NAME softraster_unit COMMAND softraster_unit_tests)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME softraster_integration
            COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration.py"
                    "$<TARGET_FILE:softraster>"
                    "${CMAKE_CURRENT_BINARY_DIR}/integration_work"
        )
    endif()
endif()

install(TARGETS softraster RUNTIME DESTINATION bin)
