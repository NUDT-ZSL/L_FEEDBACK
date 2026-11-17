cmake_minimum_required(VERSION 3.20)

project(OfflineSoftRasterizer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    include(CheckIncludeFileCXX)
    check_include_file_cxx("DirectXMath.h" RASTER_FOUND_DIRECTXMATH)
    if(NOT RASTER_FOUND_DIRECTXMATH)
        message(FATAL_ERROR
            "DirectXMath.h was not found. Install the Visual Studio C++ desktop "
            "workload and a Windows 10/11 SDK.")
    endif()
endif()

add_library(raster_core
    src/bmp.cpp
    src/json.cpp
    src/renderer.cpp
    src/scene.cpp
)

target_include_directories(raster_core PUBLIC include)

if(MSVC)
    target_compile_options(raster_core PRIVATE /W4 /fp:precise)
    target_compile_definitions(raster_core PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(raster_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(soft_raster src/main.cpp)
target_link_libraries(soft_raster PRIVATE raster_core)

if(MSVC)
    target_compile_options(soft_raster PRIVATE /W4 /fp:precise)
    target_compile_definitions(soft_raster PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(soft_raster PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(generate_textures tools/generate_textures.cpp)
target_link_libraries(generate_textures PRIVATE raster_core)

set(SAMPLE_BUILD_DIR "${CMAKE_BINARY_DIR}/samples")
file(MAKE_DIRECTORY "${SAMPLE_BUILD_DIR}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/samples/acceptance.json")
configure_file(samples/acceptance.json "${SAMPLE_BUILD_DIR}/acceptance.json" COPYONLY)

add_custom_command(
    OUTPUT
        "${SAMPLE_BUILD_DIR}/textures/base.bmp"
        "${SAMPLE_BUILD_DIR}/textures/detail.bmp"
    COMMAND generate_textures "${SAMPLE_BUILD_DIR}/textures"
    DEPENDS generate_textures
    COMMENT "Generating deterministic sample textures"
    VERBATIM
)
add_custom_target(sample_assets ALL
    DEPENDS
        "${SAMPLE_BUILD_DIR}/textures/base.bmp"
        "${SAMPLE_BUILD_DIR}/textures/detail.bmp"
)

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_FOUND)
        add_test(
            NAME acceptance
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/acceptance.py"
                --renderer "$<TARGET_FILE:soft_raster>"
                --texture-gen "$<TARGET_FILE:generate_textures>"
                --sample "${CMAKE_SOURCE_DIR}/samples/acceptance.json"
        )
        add_custom_target(check
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/acceptance.py"
                --renderer "$<TARGET_FILE:soft_raster>"
                --texture-gen "$<TARGET_FILE:generate_textures>"
                --sample "${CMAKE_SOURCE_DIR}/samples/acceptance.json"
            DEPENDS soft_raster generate_textures sample_assets
            USES_TERMINAL
            VERBATIM
        )
    else()
        message(WARNING "Python 3 not found; integration acceptance test is disabled.")
    endif()
endif()

