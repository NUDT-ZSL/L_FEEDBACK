cmake_minimum_required(VERSION 3.20)

project(SkinnedCsmRenderer VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SR_BUILD_TESTS "Build the CPU unit tests" ON)
option(SR_WARNINGS_AS_ERRORS "Treat project warnings as errors" OFF)

set(SR_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

add_library(glad STATIC "${SR_THIRD_PARTY_DIR}/glad/src/gl.c")
target_include_directories(glad PUBLIC "${SR_THIRD_PARTY_DIR}/glad/include")

add_library(sr_core STATIC
    src/animation.cpp
    src/cascade.cpp
    src/cli.cpp
    src/image_io.cpp
    src/scene.cpp
)
target_include_directories(sr_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${SR_THIRD_PARTY_DIR}"
        "${SR_THIRD_PARTY_DIR}/stb"
)

if(MSVC)
    set(SR_WARNING_FLAGS /W4 /permissive-)
    target_compile_definitions(sr_core PUBLIC _CRT_SECURE_NO_WARNINGS NOMINMAX)
    if(SR_WARNINGS_AS_ERRORS)
        list(APPEND SR_WARNING_FLAGS /WX)
    endif()
else()
    set(SR_WARNING_FLAGS -Wall -Wextra -Wpedantic)
    if(SR_WARNINGS_AS_ERRORS)
        list(APPEND SR_WARNING_FLAGS -Werror)
    endif()
endif()

target_compile_options(sr_core PRIVATE ${SR_WARNING_FLAGS})

if(SR_BUILD_TESTS)
    enable_testing()
    add_executable(sr_core_tests tests/core_tests.cpp)
    target_link_libraries(sr_core_tests PRIVATE sr_core)
    target_compile_options(sr_core_tests PRIVATE ${SR_WARNING_FLAGS})
    add_test(NAME sr_core_tests COMMAND sr_core_tests)
endif()

set(SR_HAVE_RENDER_EXECUTABLE FALSE)

if(WIN32)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

    if(EXISTS "${SR_THIRD_PARTY_DIR}/glfw/CMakeLists.txt")
        add_subdirectory("${SR_THIRD_PARTY_DIR}/glfw" EXCLUDE_FROM_ALL)
        set(SR_GLFW_TARGET glfw)
    else()
        find_package(glfw3 3.3 CONFIG REQUIRED)
        set(SR_GLFW_TARGET glfw)
    endif()

    add_executable(skinned_csm_renderer
        src/main.cpp
        src/gl_context.cpp
        src/renderer.cpp
    )
    target_link_libraries(skinned_csm_renderer PRIVATE sr_core glad ${SR_GLFW_TARGET} opengl32)
    target_compile_definitions(skinned_csm_renderer PRIVATE SR_PLATFORM_WINDOWS=1)
    set(SR_HAVE_RENDER_EXECUTABLE TRUE)
elseif(UNIX)
    set(OSMESA_ROOT "" CACHE PATH "Prefix containing include/GL/osmesa.h and libOSMesa")
    find_path(SR_OSMESA_INCLUDE_DIR GL/osmesa.h
        HINTS
            "${OSMESA_ROOT}/include"
            "${OSMESA_ROOT}/usr/include"
    )
    find_library(SR_OSMESA_LIBRARY NAMES OSMesa
        HINTS
            "${OSMESA_ROOT}/lib"
            "${OSMESA_ROOT}/usr/lib"
            "${OSMESA_ROOT}/lib/x86_64-linux-gnu"
            "${OSMESA_ROOT}/usr/lib/x86_64-linux-gnu"
    )
    find_library(SR_OSMESA_LLVM_LIBRARY NAMES LLVM-20 LLVM
        HINTS
            "${OSMESA_ROOT}/lib"
            "${OSMESA_ROOT}/usr/lib"
            "${OSMESA_ROOT}/lib/x86_64-linux-gnu"
            "${OSMESA_ROOT}/usr/lib/x86_64-linux-gnu"
    )
    find_file(SR_Z_LIBRARY NAMES libz.so.1 libz.so)

    if(SR_OSMESA_INCLUDE_DIR AND SR_OSMESA_LIBRARY)
        add_executable(skinned_csm_renderer
            src/main.cpp
            src/gl_context.cpp
            src/renderer.cpp
        )
        target_include_directories(skinned_csm_renderer PRIVATE "${SR_OSMESA_INCLUDE_DIR}")
        target_include_directories(skinned_csm_renderer PRIVATE
            "${SR_THIRD_PARTY_DIR}/glvnd")
        target_link_libraries(skinned_csm_renderer PRIVATE
            sr_core glad "${SR_OSMESA_LIBRARY}")
        if(SR_Z_LIBRARY)
            target_link_libraries(skinned_csm_renderer PRIVATE "${SR_Z_LIBRARY}")
        endif()
        if(SR_OSMESA_LLVM_LIBRARY)
            target_link_libraries(skinned_csm_renderer PRIVATE "${SR_OSMESA_LLVM_LIBRARY}")
        endif()
        find_file(SR_DRM_LIBRARY NAMES libdrm.so.2 libdrm.so)
        if(SR_DRM_LIBRARY)
            target_link_libraries(skinned_csm_renderer PRIVATE "${SR_DRM_LIBRARY}")
        endif()
        target_compile_definitions(skinned_csm_renderer PRIVATE SR_PLATFORM_LINUX_OSMESA=1)
        get_filename_component(SR_OSMESA_LIBRARY_DIR "${SR_OSMESA_LIBRARY}" DIRECTORY)
        set(SR_MULTIARCH_LIB_DIR "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
        target_link_options(skinned_csm_renderer PRIVATE
            "-Wl,--disable-new-dtags"
            "-Wl,-rpath-link,${SR_OSMESA_LIBRARY_DIR}"
        )
        if(EXISTS "${SR_MULTIARCH_LIB_DIR}")
            target_link_options(skinned_csm_renderer PRIVATE
                "-Wl,-rpath-link,${SR_MULTIARCH_LIB_DIR}")
        endif()
        set_target_properties(skinned_csm_renderer PROPERTIES
            BUILD_RPATH "${SR_OSMESA_LIBRARY_DIR}"
            INSTALL_RPATH "${SR_OSMESA_LIBRARY_DIR}"
        )
        set(SR_HAVE_RENDER_EXECUTABLE TRUE)
    else()
        message(WARNING
            "OSMesa was not found. The CPU tests will build, but the headless "
            "Linux renderer will be skipped. Set -DOSMESA_ROOT to an OSMesa prefix.")
    endif()
else()
    message(WARNING "The GPU renderer currently supports Windows (GLFW) and Linux (OSMesa).")
endif()

if(SR_HAVE_RENDER_EXECUTABLE)
    target_compile_options(skinned_csm_renderer PRIVATE ${SR_WARNING_FLAGS})
    set_target_properties(skinned_csm_renderer PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

if(TARGET sr_core_tests)
    set_target_properties(sr_core_tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()
