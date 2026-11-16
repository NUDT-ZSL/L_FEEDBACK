cmake_minimum_required(VERSION 3.20)

project(DeterministicCsm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(deterministic_csm
    src/main.cpp
    src/Options.cpp
    src/Scene.cpp
    src/Renderer.cpp
    src/WicPng.cpp
    src/Stats.cpp
)

target_compile_definitions(deterministic_csm PRIVATE
    UNICODE
    _UNICODE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
)

if(MSVC)
    target_compile_options(deterministic_csm PRIVATE /W4 /permissive- /EHsc /utf-8)
endif()

target_link_libraries(deterministic_csm PRIVATE
    d3d11
    dxgi
    d3dcompiler
    windowscodecs
    ole32
    shlwapi
)

add_custom_command(TARGET deterministic_csm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:deterministic_csm>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/scene.hlsl"
            "$<TARGET_FILE_DIR:deterministic_csm>/shaders/scene.hlsl"
    VERBATIM
)

