cmake_minimum_required(VERSION 3.20)

project(material_verify VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MATEVAL_ENABLE_GLFW "Use GLFW when a glfw3 CMake package is available" ON)
option(MATEVAL_BUILD_TESTS "Register the local acceptance test with CTest" ON)

add_library(mateval_glad STATIC third_party/glad/src/gl.c)
target_include_directories(mateval_glad PUBLIC third_party/glad/include)
set_target_properties(mateval_glad PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(material-verify
    src/main.cpp
    src/cli.cpp
    src/material.cpp
    src/context.cpp
    src/renderer.cpp
    src/validation.cpp
    src/util.cpp
)

target_include_directories(material-verify PRIVATE
    src
    third_party/nlohmann
    third_party
    third_party/glad/include
)

target_link_libraries(material-verify PRIVATE mateval_glad)

if(MSVC)
    target_compile_options(material-verify PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(material-verify PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(WIN32)
    target_link_libraries(material-verify PRIVATE opengl32)
elseif(UNIX)
    target_link_libraries(material-verify PRIVATE ${CMAKE_DL_LIBS})
endif()

if(MATEVAL_ENABLE_GLFW)
    find_package(glfw3 CONFIG QUIET)
    if(TARGET glfw)
        target_link_libraries(material-verify PRIVATE glfw)
        target_compile_definitions(material-verify PRIVATE MATEVAL_HAVE_GLFW=1)
        message(STATUS "material-verify: GLFW context enabled")
    elseif(WIN32)
        message(FATAL_ERROR
            "glfw3 was not found. Configure CMake with the vcpkg toolchain and "
            "install the glfw3 package.")
    else()
        message(STATUS "material-verify: glfw3 not found; EGL surfaceless fallback is used on Unix")
    endif()
endif()

install(TARGETS material-verify RUNTIME DESTINATION bin)

if(MATEVAL_BUILD_TESTS)
    enable_testing()
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME material_verify_acceptance
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/acceptance.py
                    $<TARGET_FILE:material-verify>
                    ${CMAKE_CURRENT_BINARY_DIR}/acceptance
        )
    endif()
endif()
