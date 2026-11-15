cmake_minimum_required(VERSION 3.20)

project(offscreen_renderer VERSION 1.0.0 LANGUAGES C CXX)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0144 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SCENE_USE_VENDORED_DEPS "Use the pinned GLFW/GLEW sources in third_party" ON)
option(SCENE_FORCE_OSMESA "Use GLFW's OSMesa backend on Unix" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(SCENE_OSMESA_CONTEXT OFF)

if(UNIX AND NOT APPLE)
    message(STATUS "OSMesa search root: ${OSMESA_ROOT}")
    find_path(SCENE_OSMESA_INCLUDE_DIR
        NAMES GL/osmesa.h
        HINTS
            "${OSMESA_ROOT}/include"
            "$ENV{OSMESA_ROOT}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/osmesa/include"
        DOC "OSMesa headers for headless OpenGL"
    )
    find_library(SCENE_OSMESA_LIBRARY
        NAMES OSMesa libOSMesa
        HINTS
            "${OSMESA_ROOT}/lib"
            "$ENV{OSMESA_ROOT}/lib"
            "${OSMESA_ROOT}/lib/x86_64-linux-gnu"
            "$ENV{OSMESA_ROOT}/lib/x86_64-linux-gnu"
        DOC "OSMesa library for headless OpenGL"
    )
    message(STATUS "OSMesa include: ${SCENE_OSMESA_INCLUDE_DIR}")
    message(STATUS "OSMesa library: ${SCENE_OSMESA_LIBRARY}")

    if(SCENE_FORCE_OSMESA AND (NOT SCENE_OSMESA_INCLUDE_DIR OR NOT SCENE_OSMESA_LIBRARY))
        message(FATAL_ERROR "SCENE_FORCE_OSMESA was set, but OSMesa was not found")
    endif()

    if(SCENE_OSMESA_INCLUDE_DIR AND SCENE_OSMESA_LIBRARY)
        set(SCENE_OSMESA_CONTEXT ON)
        set(OSMESA_INCLUDE_DIR "${SCENE_OSMESA_INCLUDE_DIR}" CACHE PATH "" FORCE)
        set(OSMESA_LIBRARIES "${SCENE_OSMESA_LIBRARY}" CACHE FILEPATH "" FORCE)
        get_filename_component(SCENE_OSMESA_LIBRARY_DIR
            "${SCENE_OSMESA_LIBRARY}" DIRECTORY)
    endif()
endif()

find_package(Threads REQUIRED)

set(SCENE_GLFW_TARGET "")
set(SCENE_GLEW_TARGET "")

if(SCENE_USE_VENDORED_DEPS)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    set(GLFW_USE_OSMESA ${SCENE_OSMESA_CONTEXT} CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/glfw EXCLUDE_FROM_ALL)
    set(SCENE_GLFW_TARGET glfw)

    add_library(glew_static STATIC third_party/glew/src/glew.c)
    add_library(GLEW::GLEW ALIAS glew_static)
    target_include_directories(glew_static
        PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/glew/include"
        PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/glvnd/include"
    )
    target_compile_definitions(glew_static PUBLIC GLEW_STATIC GLEW_NO_GLU)
    if(SCENE_OSMESA_CONTEXT)
        target_compile_definitions(glew_static PRIVATE GLEW_OSMESA)
        target_compile_definitions(glew_static PRIVATE
            "GLEW_INCLUDE=\"${CMAKE_CURRENT_SOURCE_DIR}/third_party/glew/glew_osmesa_compat.h\"")
        target_include_directories(glew_static PRIVATE "${SCENE_OSMESA_INCLUDE_DIR}")
        find_library(SCENE_LLVM_LIBRARY
            NAMES LLVM-20 LLVM
            HINTS "${SCENE_OSMESA_LIBRARY_DIR}"
        )
        find_library(SCENE_DRM_LIBRARY
            NAMES drm.so.2 libdrm.so.2 drm
            HINTS /usr/lib/x86_64-linux-gnu /usr/lib
        )
        set(SCENE_OSMESA_EXTRA_LIBRARIES "")
        if(SCENE_LLVM_LIBRARY)
            list(APPEND SCENE_OSMESA_EXTRA_LIBRARIES "${SCENE_LLVM_LIBRARY}")
        endif()
        if(SCENE_DRM_LIBRARY)
            list(APPEND SCENE_OSMESA_EXTRA_LIBRARIES "${SCENE_DRM_LIBRARY}")
        endif()
        target_link_libraries(glew_static PUBLIC
            "${SCENE_OSMESA_LIBRARY}"
            ${SCENE_OSMESA_EXTRA_LIBRARIES}
        )
    elseif(WIN32)
        target_link_libraries(glew_static PUBLIC opengl32)
    else()
        find_package(X11 REQUIRED)
        target_include_directories(glew_static PRIVATE ${X11_INCLUDE_DIR})
        target_link_libraries(glew_static PUBLIC ${X11_LIBRARIES} GL)
    endif()
    set(SCENE_GLEW_TARGET GLEW::GLEW)
else()
    find_package(glfw3 3.3 CONFIG REQUIRED)
    find_package(GLEW 2.2 REQUIRED)
    set(SCENE_GLFW_TARGET glfw)
    set(SCENE_GLEW_TARGET GLEW::GLEW)
endif()

add_executable(offscreen_renderer
    src/main.cpp
    src/context.cpp
    src/gl.cpp
    src/pipeline.cpp
    src/scene.cpp
    src/renderer.cpp
    src/post_process.cpp
    src/error.cpp
)

target_include_directories(offscreen_renderer
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_definitions(offscreen_renderer
    PRIVATE
        GLFW_INCLUDE_NONE
        SCENE_DEFAULT_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

if(SCENE_OSMESA_CONTEXT)
    target_compile_definitions(offscreen_renderer PRIVATE SCENE_OSMESA_CONTEXT=1)
    target_link_options(offscreen_renderer PRIVATE
        "-Wl,--disable-new-dtags"
        "-Wl,--allow-shlib-undefined"
    )
    set_property(TARGET offscreen_renderer APPEND PROPERTY
        BUILD_RPATH "${SCENE_OSMESA_LIBRARY_DIR}"
    )
endif()

target_link_libraries(offscreen_renderer
    PRIVATE
        ${SCENE_GLFW_TARGET}
        ${SCENE_GLEW_TARGET}
        Threads::Threads
)

if(MSVC)
    target_compile_options(offscreen_renderer PRIVATE /W4 /permissive-)
else()
    target_compile_options(offscreen_renderer PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_custom_command(TARGET offscreen_renderer POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
        "$<TARGET_FILE_DIR:offscreen_renderer>/shaders"
    COMMENT "Copying GLSL shaders next to the renderer"
)

include(CTest)
if(BUILD_TESTING)
    add_test(
        NAME renderer_acceptance
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/acceptance.sh"
                "$<TARGET_FILE:offscreen_renderer>"
                "${CMAKE_CURRENT_SOURCE_DIR}"
                "${CMAKE_CURRENT_BINARY_DIR}/acceptance"
    )
    set_tests_properties(renderer_acceptance PROPERTIES
        TIMEOUT 180
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}:$ENV{LD_LIBRARY_PATH}"
    )
endif()

