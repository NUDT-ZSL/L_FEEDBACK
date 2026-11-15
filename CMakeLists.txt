cmake_minimum_required(VERSION 3.20)

project(SubdivisionPreview VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SUBDIV_BUILD_TESTS "Build deterministic CPU tests" ON)
option(SUBDIV_FETCH_GLFW "Fetch GLFW when no installed package is found" ON)

find_package(glfw3 3.3 CONFIG QUIET)

if(NOT TARGET glfw AND NOT TARGET glfw3 AND NOT TARGET glfw3::glfw)
    if(SUBDIV_FETCH_GLFW)
        include(FetchContent)
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            glfw
            URL https://github.com/glfw/glfw/archive/refs/tags/3.4.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(glfw)
    else()
        message(FATAL_ERROR
            "GLFW 3.3+ was not found. Install it through vcpkg, provide "
            "CMAKE_PREFIX_PATH, or configure with -DSUBDIV_FETCH_GLFW=ON.")
    endif()
endif()

if(TARGET glfw)
    set(SUBDIV_GLFW_TARGET glfw)
elseif(TARGET glfw3::glfw)
    set(SUBDIV_GLFW_TARGET glfw3::glfw)
else()
    set(SUBDIV_GLFW_TARGET glfw3)
endif()

add_library(subdiv_core
    src/mesh.cpp
    src/subdivision.cpp
    src/json.cpp
    src/scene.cpp
)
target_include_directories(subdiv_core PUBLIC src)
target_compile_features(subdiv_core PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(subdiv_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(subdiv_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(subdivision_preview
    src/main.cpp
    src/renderer.cpp
    src/gl_api.cpp
)
target_compile_definitions(subdivision_preview PRIVATE GLFW_INCLUDE_NONE)
target_link_libraries(subdivision_preview PRIVATE subdiv_core ${SUBDIV_GLFW_TARGET})
if(WIN32)
    target_link_libraries(subdivision_preview PRIVATE psapi)
    target_compile_definitions(subdivision_preview PRIVATE NOMINMAX)
endif()

if(MSVC)
    target_compile_options(subdivision_preview PRIVATE /W4 /permissive-)
else()
    target_compile_options(subdivision_preview PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(MSVC)
    target_compile_definitions(subdivision_preview PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

if(SUBDIV_BUILD_TESTS)
    enable_testing()
    add_executable(subdiv_tests tests/core_tests.cpp)
    target_link_libraries(subdiv_tests PRIVATE subdiv_core)
    if(MSVC)
        target_compile_options(subdiv_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(subdiv_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME subdiv_core_tests COMMAND subdiv_tests)
endif()

