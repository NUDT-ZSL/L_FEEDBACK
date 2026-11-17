cmake_minimum_required(VERSION 3.20)

project(CharView LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CHARVIEW_BUILD_TESTS "Build the deterministic software-renderer tests" ON)

add_library(charview_core
    src/cv_math.cpp
    src/assets.cpp
    src/visibility.cpp
    src/raster.cpp
    src/renderer.cpp
    src/image_io.cpp
    src/options.cpp
)
target_include_directories(charview_core PUBLIC src)

if(MSVC)
    # Determinism requires precise floating point. Never enable /fp:fast.
    target_compile_options(charview_core PUBLIC /W4 /permissive- /fp:precise /Zc:preprocessor)
    target_compile_definitions(charview_core PUBLIC _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    target_compile_options(charview_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(charview_headless src/headless_main.cpp)
target_link_libraries(charview_headless PRIVATE charview_core)

if(WIN32)
    # A console-subsystem executable keeps the required culling/statistics log visible.
    add_executable(charview src/win32_main.cpp)
    target_link_libraries(charview PRIVATE charview_core gdi32 user32)
    target_compile_definitions(charview PRIVATE UNICODE _UNICODE _WIN32_WINNT=0x0601
                                                     WINVER=0x0601)
    if(MSVC)
        target_compile_options(charview PRIVATE /W4 /permissive- /fp:precise)
    endif()
endif()

if(CHARVIEW_BUILD_TESTS)
    enable_testing()
    add_executable(charview_tests tests/test_main.cpp)
    target_link_libraries(charview_tests PRIVATE charview_core)
    add_test(NAME charview_deterministic_checks COMMAND charview_tests)
    set_tests_properties(charview_deterministic_checks PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()

