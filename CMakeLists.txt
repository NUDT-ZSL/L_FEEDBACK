cmake_minimum_required(VERSION 3.21)

project(TerrainLOD VERSION 1.0.0 LANGUAGES CXX)

option(TERRAIN_BUILD_GL "Build the GLFW/OpenGL 4.6 desktop renderer" ${WIN32})
option(TERRAIN_BUILD_TESTS "Build deterministic core tests" ON)
option(TERRAIN_BUILD_EGL_SMOKE "Build the optional Linux EGL surfaceless GL smoke test" OFF)

if(TERRAIN_BUILD_TESTS OR TERRAIN_BUILD_EGL_SMOKE)
    enable_testing()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(terrain_core
    src/camera.cpp
    src/camera_path.cpp
    src/cache.cpp
    src/cli.cpp
    src/lod.cpp
    src/ppm.cpp
    src/render_data.cpp
    src/seams.cpp
    src/software_renderer.cpp
    src/terrain.cpp
    src/version.cpp
)

target_include_directories(terrain_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(terrain_core PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(terrain_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(terrain_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(terrain_lod src/main.cpp src/app.cpp)
target_link_libraries(terrain_lod PRIVATE terrain_core)

if(TERRAIN_BUILD_GL)
    find_package(glfw3 3.4 CONFIG REQUIRED)
    find_package(glm 1.0.1 CONFIG QUIET)
    if(NOT glm_FOUND)
        find_package(glm CONFIG REQUIRED)
    endif()

    target_sources(terrain_lod PRIVATE
        src/gl/gl_api.cpp
        src/gl/gl_renderer.cpp
        src/gl_app.cpp
    )
    target_compile_definitions(terrain_lod PRIVATE TERRAIN_HAS_GL=1)
    target_link_libraries(terrain_lod PRIVATE glfw glm::glm)
    if(WIN32)
        target_link_libraries(terrain_lod PRIVATE opengl32)
    endif()

    if(TERRAIN_BUILD_EGL_SMOKE AND UNIX)
        add_executable(gl_egl_smoke
            tests/gl_egl_smoke.cpp
            src/gl/gl_api.cpp
            src/gl/gl_renderer.cpp
        )
        target_include_directories(gl_egl_smoke PRIVATE src)
        target_link_libraries(gl_egl_smoke PRIVATE terrain_core glfw ${CMAKE_DL_LIBS})
        if(MSVC)
            target_compile_options(gl_egl_smoke PRIVATE /W4 /permissive-)
        else()
            target_compile_options(gl_egl_smoke PRIVATE -Wall -Wextra -Wpedantic)
        endif()
        add_test(NAME gl_egl_smoke COMMAND gl_egl_smoke ${CMAKE_CURRENT_BINARY_DIR}/egl_smoke.ppm)
        set_tests_properties(gl_egl_smoke PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

if(MSVC)
    target_compile_options(terrain_lod PRIVATE /W4 /permissive-)
else()
    target_compile_options(terrain_lod PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(TERRAIN_BUILD_TESTS)
    add_executable(terrain_tests tests/terrain_tests.cpp)
    target_link_libraries(terrain_tests PRIVATE terrain_core)
    if(MSVC)
        target_compile_options(terrain_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(terrain_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME terrain_core_tests COMMAND terrain_tests)
endif()

install(TARGETS terrain_lod RUNTIME DESTINATION bin)
install(FILES assets/camera_path.txt DESTINATION share/terrain-lod)

