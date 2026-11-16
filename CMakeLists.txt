cmake_minimum_required(VERSION 3.25)

project(SkelPreview
    VERSION 1.0.0
    DESCRIPTION "Deterministic skeletal animation preview and offline renderer"
    LANGUAGES CXX)

option(SKELPREVIEW_BUILD_TESTS "Build the automated test executable" ON)
option(SKELPREVIEW_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# A local dependency prefix makes the Linux build usable without sudo. It is
# populated by scripts/bootstrap_deps_linux.sh and is ignored by git.
list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.deps/usr/usr")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.deps/usr/usr/lib/x86_64-linux-gnu")
    list(PREPEND CMAKE_LIBRARY_PATH
        "${CMAKE_CURRENT_SOURCE_DIR}/.deps/usr/usr/lib/x86_64-linux-gnu")
endif()

set(OpenGL_GL_PREFERENCE LEGACY)
find_package(OpenGL REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

find_package(glfw3 3.3 CONFIG QUIET)
if(NOT TARGET glfw)
    find_path(GLFW_INCLUDE_DIR GLFW/glfw3.h REQUIRED)
    find_library(GLFW_LIBRARY NAMES glfw glfw3 REQUIRED)
    add_library(glfw UNKNOWN IMPORTED)
    set_target_properties(glfw PROPERTIES
        IMPORTED_LOCATION "${GLFW_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GLFW_INCLUDE_DIR}")
endif()

find_path(STB_INCLUDE_DIR
    NAMES stb_image_write.h
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/.deps/usr/usr/include"
    PATH_SUFFIXES stb
    REQUIRED)
add_library(stb_headers INTERFACE)
target_include_directories(stb_headers INTERFACE "${STB_INCLUDE_DIR}")

add_library(skelpreview_core STATIC
    src/asset.cpp
    src/asset_reader.cpp
    src/error.cpp
    src/image_io.cpp
    src/offline_renderer.cpp
    src/report.cpp
    src/scene.cpp
    src/sample_generator.cpp)
target_include_directories(skelpreview_core PUBLIC src)
target_link_libraries(skelpreview_core
    PUBLIC
        glm::glm
        nlohmann_json::nlohmann_json
        stb_headers)

add_library(skelpreview_preview STATIC
    src/gl_api.cpp
    src/preview.cpp)
target_link_libraries(skelpreview_preview
    PUBLIC
        skelpreview_core
        glfw
        OpenGL::GL)

add_executable(skelpreview
    src/cli.cpp
    src/main.cpp)
target_link_libraries(skelpreview PRIVATE skelpreview_preview)
if(UNIX AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.deps/usr/usr/lib/x86_64-linux-gnu")
    target_link_directories(skelpreview PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/.deps/usr/usr/lib/x86_64-linux-gnu")
    target_link_options(skelpreview PRIVATE
        "-Wl,-rpath-link,${CMAKE_CURRENT_SOURCE_DIR}/.deps/usr/usr/lib/x86_64-linux-gnu"
        "-Wl,-rpath-link,/usr/lib/x86_64-linux-gnu")
endif()

add_executable(asset_generator tools/asset_gen.cpp)
target_link_libraries(asset_generator PRIVATE skelpreview_core)

if(MSVC)
    target_compile_options(skelpreview_core PRIVATE /W4 /permissive-)
    target_compile_options(skelpreview_preview PRIVATE /W4 /permissive-)
    target_compile_options(skelpreview PRIVATE /W4 /permissive-)
    target_compile_definitions(skelpreview_core PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    target_compile_options(skelpreview_core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(skelpreview_preview PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(skelpreview PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(SKELPREVIEW_WARNINGS_AS_ERRORS)
    if(MSVC)
        target_compile_options(skelpreview_core PRIVATE /WX)
    else()
        target_compile_options(skelpreview_core PRIVATE -Werror)
    endif()
endif()

if(SKELPREVIEW_BUILD_TESTS)
    enable_testing()
    add_library(skelpreview_testlib STATIC
        tests/test_runner.cpp)
    target_link_libraries(skelpreview_testlib PUBLIC skelpreview_core)
    target_include_directories(skelpreview_testlib PUBLIC tests)
    target_link_libraries(skelpreview PRIVATE skelpreview_testlib)
    target_compile_definitions(skelpreview PRIVATE SKELPREVIEW_HAS_TESTS=1)

    add_executable(skelpreview_tests tests/test_main.cpp)
    target_link_libraries(skelpreview_tests PRIVATE skelpreview_testlib)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME acceptance
            COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/acceptance.py"
                "$<TARGET_FILE:asset_generator>"
                "$<TARGET_FILE:skelpreview>"
                "${CMAKE_CURRENT_BINARY_DIR}/acceptance")
    endif()
    add_test(NAME unit_and_numeric
        COMMAND skelpreview_tests "${CMAKE_CURRENT_BINARY_DIR}/unit_assets")
    add_test(NAME cli_selftest
        COMMAND skelpreview selftest "${CMAKE_CURRENT_BINARY_DIR}/cli_selftest")
endif()

install(TARGETS skelpreview asset_generator RUNTIME DESTINATION bin)

