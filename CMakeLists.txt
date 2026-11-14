cmake_minimum_required(VERSION 3.20)

project(volumetric_cloud_cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_library(cloudcore STATIC
    src/json.cpp
    src/math.cpp
    src/crc32.cpp
    src/log.cpp
    src/thread_pool.cpp
    src/density.cpp
    src/renderer.cpp
    src/snapshot.cpp
    src/config.cpp
    src/image.cpp
)

target_include_directories(cloudcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(cloudcore PUBLIC Threads::Threads)

add_executable(cloud_render src/main.cpp)
target_link_libraries(cloud_render PRIVATE cloudcore)

add_executable(cloud_tests tests/cloud_tests.cpp)
target_link_libraries(cloud_tests PRIVATE cloudcore)

if (MSVC)
    target_compile_options(cloudcore PRIVATE /W4 /permissive- /fp:strict)
    target_compile_options(cloud_render PRIVATE /W4 /permissive- /fp:strict)
    target_compile_options(cloud_tests PRIVATE /W4 /permissive- /fp:strict)
else()
    target_compile_options(cloudcore PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math)
    target_compile_options(cloud_render PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math)
    target_compile_options(cloud_tests PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math)
endif()

enable_testing()
add_test(NAME cloud_core_tests COMMAND cloud_tests "${CMAKE_CURRENT_BINARY_DIR}/test-output")
set_tests_properties(cloud_core_tests PROPERTIES TIMEOUT 600)

find_package(Python3 QUIET COMPONENTS Interpreter)
if (Python3_Interpreter_FOUND)
    add_custom_target(validate
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate.py
                --binary $<TARGET_FILE:cloud_render>
                --workspace ${CMAKE_CURRENT_SOURCE_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS cloud_render cloud_tests
        USES_TERMINAL
    )
else()
    add_custom_target(validate
        COMMAND ${CMAKE_COMMAND} -E echo "Python 3 not found; run cloud_tests and the sample configs directly"
        DEPENDS cloud_render cloud_tests
    )
endif()

