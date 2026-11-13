cmake_minimum_required(VERSION 3.15)

project(rigskin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_library(rigskin_core
    src/math.cpp
    src/sha256.cpp
    src/model.cpp
    src/skinning.cpp
    src/mesh_io.cpp
    src/render.cpp
    src/snapshot.cpp
    src/pipeline.cpp
)

target_include_directories(rigskin_core PUBLIC src)
target_link_libraries(rigskin_core PUBLIC Threads::Threads)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10)
    target_link_libraries(rigskin_core PUBLIC stdc++fs)
endif()

if(MSVC)
    target_compile_options(rigskin_core PRIVATE /W4 /permissive- /fp:precise)
else()
    target_compile_options(
        rigskin_core PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
endif()

add_executable(rigskin src/main.cpp)
target_link_libraries(rigskin PRIVATE rigskin_core)

enable_testing()
add_executable(rigskin_unit_tests tests/unit_tests.cpp)
target_link_libraries(rigskin_unit_tests PRIVATE rigskin_core)
add_test(NAME rigskin_unit_tests COMMAND rigskin_unit_tests)

