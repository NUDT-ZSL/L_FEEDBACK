cmake_minimum_required(VERSION 3.20)

project(IndoorDeferredRenderer LANGUAGES C CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(INDOOR_BUILD_APP "Build the GLFW interactive renderer" ON)
option(INDOOR_BUILD_GPU_TESTS "Build the EGL/OpenGL headless verification executable" ON)
option(INDOOR_BUILD_CORE_TESTS "Build deterministic CPU tests" ON)

set(GBUF_ATTACHMENT_WORLD_POSITION 0)
set(GBUF_ATTACHMENT_VIEW_NORMAL 1)
set(GBUF_ATTACHMENT_ALBEDO 2)
set(GBUF_ATTACHMENT_SURFACE 3)
set(GBUF_ATTACHMENT_COUNT 4)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/render/GBufferLayout.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/GBufferLayout.hpp
    @ONLY
)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gbuffer_layout.glsl.in
    ${CMAKE_CURRENT_BINARY_DIR}/shaders/gbuffer_layout.glsl
    @ONLY
)
file(GLOB INDOOR_SHADER_FILES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.frag
)
add_custom_target(indoor_shaders ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${INDOOR_SHADER_FILES} ${CMAKE_CURRENT_BINARY_DIR}/shaders
    DEPENDS ${INDOOR_SHADER_FILES}
    COMMENT "Copying GLSL shaders"
)

find_package(Threads REQUIRED)

find_path(GLM_INCLUDE_DIR glm/glm.hpp REQUIRED)
add_library(glm_headers INTERFACE)
target_include_directories(glm_headers INTERFACE ${GLM_INCLUDE_DIR})

add_library(glad STATIC third_party/glad/src/gl.c)
target_include_directories(glad PUBLIC third_party/glad/include)

add_library(indoor_core STATIC
    src/core/DeterministicScene.cpp
    src/core/ShadowSystems.cpp
    src/core/ResourceManager.cpp
    src/render/ProceduralMaterials.cpp
)
target_include_directories(indoor_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_link_libraries(indoor_core PUBLIC glm_headers Threads::Threads)

add_library(indoor_renderer STATIC
    src/render/GLUtil.cpp
    src/render/ShaderProgram.cpp
    src/render/DeferredRenderer.cpp
)
target_include_directories(indoor_renderer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_link_libraries(indoor_renderer PUBLIC indoor_core glad)
add_dependencies(indoor_renderer indoor_shaders)
target_compile_definitions(indoor_renderer PUBLIC
    INDOOR_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders"
)

if(INDOOR_BUILD_APP)
    find_package(glfw3 3.3 CONFIG REQUIRED)
    add_executable(indoor_deferred
        src/main.cpp
        src/app/Application.cpp
    )
    target_include_directories(indoor_deferred PRIVATE src)
    target_link_libraries(indoor_deferred PRIVATE indoor_renderer glfw)
    if(WIN32)
        target_compile_definitions(indoor_deferred PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    endif()
endif()

if(INDOOR_BUILD_CORE_TESTS)
    add_executable(indoor_core_tests tests/core_tests.cpp)
    target_link_libraries(indoor_core_tests PRIVATE indoor_core)
    add_test(NAME indoor_core_tests COMMAND indoor_core_tests)
endif()

if(INDOOR_BUILD_GPU_TESTS)
    find_path(EGL_INCLUDE_DIR EGL/egl.h)
    find_library(EGL_LIBRARY NAMES EGL.so.1 EGL)
    find_library(GLDISPATCH_LIBRARY NAMES GLdispatch.so.0 GLdispatch)
    if(EGL_INCLUDE_DIR AND EGL_LIBRARY)
        add_library(EGL::EGL UNKNOWN IMPORTED)
        set_target_properties(EGL::EGL PROPERTIES
            IMPORTED_LOCATION "${EGL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${EGL_INCLUDE_DIR}"
        )
        add_executable(indoor_gpu_tests tests/gpu_tests.cpp)
        target_link_libraries(indoor_gpu_tests PRIVATE indoor_renderer EGL::EGL)
        if(GLDISPATCH_LIBRARY)
            target_link_libraries(indoor_gpu_tests PRIVATE ${GLDISPATCH_LIBRARY})
        endif()
        target_compile_definitions(indoor_gpu_tests PRIVATE
            INDOOR_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders"
        )
        add_test(NAME indoor_gpu_tests COMMAND indoor_gpu_tests --artifacts
                 ${CMAKE_CURRENT_SOURCE_DIR}/artifacts)
    else()
        message(STATUS "EGL not found; headless GPU tests are disabled")
    endif()
endif()
