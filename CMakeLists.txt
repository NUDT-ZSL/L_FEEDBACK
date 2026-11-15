cmake_minimum_required(VERSION 3.16)

project(pbr_offline_renderer VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(pbr_renderer
    src/main.cpp
    src/options.cpp
    src/scene.cpp
    src/environment.cpp
    src/image.cpp
    src/stb_impl.cpp
    src/stats.cpp
    src/validation.cpp
    src/gl_backend.cpp
    src/renderer.cpp
    third_party/glad/gl.c
)

target_include_directories(pbr_renderer PRIVATE
    src
    third_party
)

if(MSVC)
    target_compile_options(pbr_renderer PRIVATE /W4 /permissive-)
else()
    target_compile_options(pbr_renderer PRIVATE -Wall -Wextra -Wpedantic)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(pbr_renderer PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-dangling-reference>
        )
    endif()
endif()

if(WIN32)
    find_package(OpenGL REQUIRED)
    if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/glfw/CMakeLists.txt")
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
        add_subdirectory(third_party/glfw)
        target_link_libraries(pbr_renderer PRIVATE glfw OpenGL::GL)
    else()
        find_package(glfw3 3.3 QUIET)
    endif()
    if(TARGET glfw AND NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/glfw/CMakeLists.txt")
        target_link_libraries(pbr_renderer PRIVATE glfw OpenGL::GL)
    elseif(NOT TARGET glfw)
        find_path(GLFW_INCLUDE_DIR NAMES GLFW/glfw3.h)
        find_library(GLFW_LIBRARY NAMES glfw glfw3)
        if(NOT GLFW_INCLUDE_DIR OR NOT GLFW_LIBRARY)
            message(FATAL_ERROR
                "GLFW 3.3+ was not found. Install glfw3 with vcpkg and pass "
                "-DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake, "
                "or set GLFW_INCLUDE_DIR and GLFW_LIBRARY.")
        endif()
        target_include_directories(pbr_renderer PRIVATE ${GLFW_INCLUDE_DIR})
        target_link_libraries(pbr_renderer PRIVATE ${GLFW_LIBRARY} OpenGL::GL)
    endif()
else()
    find_package(Threads REQUIRED)
    find_library(EGL_LIBRARY NAMES EGL libEGL.so.1)
    find_library(GL_DISPATCH_LIBRARY NAMES GLdispatch libGLdispatch.so.0)
    find_library(HOST_GL_LIBRARY NAMES GL OpenGL libGL.so.1)
    if(NOT EGL_LIBRARY OR NOT GL_DISPATCH_LIBRARY OR NOT HOST_GL_LIBRARY)
        message(FATAL_ERROR "Linux headless backend requires EGL, GLdispatch, and OpenGL runtime libraries")
    endif()
    target_link_libraries(pbr_renderer PRIVATE
        ${EGL_LIBRARY}
        ${GL_DISPATCH_LIBRARY}
        ${HOST_GL_LIBRARY}
        ${CMAKE_DL_LIBS}
        Threads::Threads
    )
endif()

if(WIN32)
    add_custom_command(TARGET pbr_renderer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_SOURCE_DIR}/assets"
                "$<TARGET_FILE_DIR:pbr_renderer>/assets"
        COMMENT "Copying offline scene assets next to the renderer"
    )
endif()

install(TARGETS pbr_renderer RUNTIME DESTINATION bin)
install(DIRECTORY assets/ DESTINATION share/pbr_offline_renderer/assets)
