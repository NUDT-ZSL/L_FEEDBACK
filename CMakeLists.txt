cmake_minimum_required(VERSION 3.20)

project(OfflineHdrRenderer VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Wno-dangling-reference" RENDER_HAS_NO_DANGLING_REFERENCE)

option(RENDER_BUILD_TESTS "Build parser and integration tests" ON)
option(RENDER_FETCH_GLFW "Fetch GLFW when a package is not available" ON)

find_package(glfw3 3.3 CONFIG QUIET)
if(NOT glfw3_FOUND)
    if(NOT RENDER_FETCH_GLFW)
        message(FATAL_ERROR "GLFW 3.3 was not found and RENDER_FETCH_GLFW is OFF")
    endif()
    include(FetchContent)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        glfw
        URL https://github.com/glfw/glfw/archive/refs/tags/3.3.9.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(glfw)
endif()

add_library(renderer_glad STATIC third_party/glad/src/gl.c)
target_include_directories(renderer_glad PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/glad/include"
)

add_library(renderer_stb INTERFACE)
target_include_directories(renderer_stb INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb"
)

add_executable(render
    src/main.cpp
    src/renderer.cpp
    src/scene.cpp
    src/logger.cpp
    src/stb_impl.cpp
)
target_include_directories(render PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)
target_link_libraries(render PRIVATE renderer_glad renderer_stb glfw ${CMAKE_DL_LIBS})

if(MSVC)
    target_compile_options(render PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(render PRIVATE -Wall -Wextra -Wpedantic)
    if(RENDER_HAS_NO_DANGLING_REFERENCE)
        target_compile_options(render PRIVATE -Wno-dangling-reference)
    endif()
endif()

if(RENDER_BUILD_TESTS)
    enable_testing()

    add_executable(scene_tests
        tests/test_scene.cpp
        src/scene.cpp
    )
    target_include_directories(scene_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_libraries(scene_tests PRIVATE renderer_stb)
    if(MSVC)
        target_compile_options(scene_tests PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(scene_tests PRIVATE -Wall -Wextra -Wpedantic)
        if(RENDER_HAS_NO_DANGLING_REFERENCE)
            target_compile_options(scene_tests PRIVATE -Wno-dangling-reference)
        endif()
    endif()

    add_test(NAME scene_validation COMMAND scene_tests)
    add_test(
        NAME render_integration
        COMMAND "${CMAKE_COMMAND}"
            "-DRENDER_EXE=$<TARGET_FILE:render>"
            "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/integration"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_test.cmake"
    )
    set_tests_properties(render_integration PROPERTIES TIMEOUT 300)
endif()
