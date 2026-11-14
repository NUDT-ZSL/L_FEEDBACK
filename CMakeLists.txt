cmake_minimum_required(VERSION 3.20)

project(lightcluster_deferred VERSION 1.0.0 LANGUAGES CXX)

option(LIGHTCLUSTER_BUILD_RENDERER "Build the Windows OpenGL 4.5 renderer" ON)
option(LIGHTCLUSTER_BUILD_TESTS "Build deterministic core tests" ON)

find_package(Threads REQUIRED)

add_library(lightcluster_core
    src/core/animation.cpp
    src/core/buffer_sync.cpp
    src/core/cli.cpp
    src/core/deterministic_engine.cpp
    src/core/json_log.cpp
    src/core/scene.cpp
    src/core/types.cpp
)
add_library(lightcluster::core ALIAS lightcluster_core)

target_include_directories(lightcluster_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(lightcluster_core PUBLIC cxx_std_17)
target_link_libraries(lightcluster_core PUBLIC Threads::Threads)

if(MSVC)
    target_compile_options(lightcluster_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(lightcluster_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(lightcluster_sim src/sim_main.cpp)
target_link_libraries(lightcluster_sim PRIVATE lightcluster::core)

if(LIGHTCLUSTER_BUILD_RENDERER)
    find_package(glfw3 CONFIG QUIET)
    find_package(glm CONFIG QUIET)
    find_package(GLEW QUIET)
    find_package(OpenGL QUIET)

    if(TARGET glfw AND TARGET glm::glm AND TARGET GLEW::GLEW AND
       TARGET OpenGL::GL)
        add_executable(lightcluster
            src/app/main.cpp
            src/app/renderer.cpp
        )
        target_link_libraries(lightcluster
            PRIVATE
                lightcluster::core
                glfw
                glm::glm
                GLEW::GLEW
                OpenGL::GL
        )
        target_compile_definitions(lightcluster PRIVATE LIGHTCLUSTER_HAS_OPENGL=1)
        target_compile_definitions(lightcluster PRIVATE GLEW_NO_GLU=1)
        target_compile_definitions(lightcluster PRIVATE GLFW_INCLUDE_NONE=1)
        message(STATUS "OpenGL renderer enabled: lightcluster")
    else()
        message(WARNING
            "GLFW/GLM/GLEW were not found; building core and simulator only. "
            "Configure with the vcpkg toolchain to build the OpenGL renderer.")
    endif()
endif()

if(LIGHTCLUSTER_BUILD_TESTS)
    enable_testing()
    add_executable(lightcluster_core_tests tests/core_tests.cpp)
    target_link_libraries(lightcluster_core_tests PRIVATE lightcluster::core)
    add_test(NAME core_tests COMMAND lightcluster_core_tests)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME deterministic_cli_validation
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate_cli.py
                $<TARGET_FILE:lightcluster_sim>
        )
    endif()
endif()
