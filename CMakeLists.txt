cmake_minimum_required(VERSION 3.24)

project(character_layered_renderer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CHARACTER_RENDER_OPENGL "Build the OpenGL 4.3 deferred renderer" ${WIN32})
option(CHARACTER_RENDER_TESTS "Add CTest integration tests" ON)

find_package(nlohmann_json CONFIG QUIET)
find_package(Stb QUIET)

add_executable(character_renderer
    src/main.cpp
    src/common.cpp
    src/sample_generator.cpp
    src/material.cpp
    src/scene.cpp
    src/snapshot.cpp
    src/cpu_renderer.cpp
    src/stb_impl.cpp
)

target_include_directories(character_renderer PRIVATE src)

if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(character_renderer PRIVATE nlohmann_json::nlohmann_json)
else()
    target_include_directories(character_renderer SYSTEM PRIVATE third_party)
    message(STATUS "nlohmann-json not found; using pinned third_party header")
endif()

if(Stb_FOUND OR TARGET Stb::Stb)
    target_include_directories(character_renderer SYSTEM PRIVATE ${Stb_INCLUDE_DIR})
    message(STATUS "Using vcpkg stb headers from ${Stb_INCLUDE_DIR}")
else()
    target_include_directories(character_renderer SYSTEM PRIVATE third_party/stb)
    message(STATUS "stb not found; using pinned third_party headers")
endif()

if(CHARACTER_RENDER_OPENGL)
    if(NOT WIN32)
        message(FATAL_ERROR "CHARACTER_RENDER_OPENGL is intended for the Windows target")
    endif()
    find_package(glfw3 CONFIG REQUIRED)
    find_package(glm CONFIG REQUIRED)
    target_sources(character_renderer PRIVATE
        src/opengl_renderer.cpp
        src/gl_loader.cpp
    )
    target_compile_definitions(character_renderer PRIVATE CHARACTER_RENDER_HAVE_OPENGL=1)
    target_link_libraries(character_renderer PRIVATE glfw glm::glm opengl32)
endif()

if(MSVC)
    target_compile_options(character_renderer PRIVATE /W4 /permissive- /utf-8)
    set_source_files_properties(src/stb_impl.cpp PROPERTIES COMPILE_FLAGS "/w")
else()
    target_compile_options(character_renderer PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
    set_source_files_properties(src/stb_impl.cpp PROPERTIES COMPILE_FLAGS "-w")
endif()

if(CHARACTER_RENDER_TESTS)
    enable_testing()
    add_test(
        NAME character_renderer_integration
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:character_renderer>
            -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/integration_data
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_integration.cmake
    )
    set_tests_properties(character_renderer_integration PROPERTIES TIMEOUT 300)
endif()

