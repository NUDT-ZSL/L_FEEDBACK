cmake_minimum_required(VERSION 3.20)

project(OfflineAssetPreview VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PREVIEW_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

add_library(preview_core STATIC
    src/bmp.cpp
    src/config.cpp
    src/image.cpp
    src/json.cpp
    src/model.cpp
    src/renderer.cpp
    src/stb_impl.cpp
)

target_include_directories(preview_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${PREVIEW_THIRD_PARTY_DIR}"
        "${PREVIEW_THIRD_PARTY_DIR}/glm"
)

if(MSVC)
    target_compile_options(preview_core PRIVATE /W4 /permissive- /utf-8 /EHsc)
else()
    target_compile_options(preview_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(asset_preview src/main.cpp)
target_link_libraries(asset_preview PRIVATE preview_core)
set_target_properties(asset_preview PROPERTIES
    WIN32_EXECUTABLE FALSE
    OUTPUT_NAME asset_preview
)

if(MSVC)
    target_compile_options(asset_preview PRIVATE /W4 /permissive- /utf-8 /EHsc)
endif()

add_executable(make_sample tools/sample_generator.cpp)
target_include_directories(make_sample PRIVATE
    "${PREVIEW_THIRD_PARTY_DIR}"
    "${PREVIEW_THIRD_PARTY_DIR}/glm"
)

if(MSVC)
    target_compile_options(make_sample PRIVATE /W4 /permissive- /utf-8 /EHsc)
else()
    target_compile_options(make_sample PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME renderer_integration
            COMMAND
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/validate.py"
                --exe "$<TARGET_FILE:asset_preview>"
                --generator "$<TARGET_FILE:make_sample>"
                --source-dir "${CMAKE_CURRENT_SOURCE_DIR}"
                --work-dir "${CMAKE_CURRENT_BINARY_DIR}/integration_work"
        )
    endif()
endif()
