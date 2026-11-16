cmake_minimum_required(VERSION 3.20)

project(offline_render_verify VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wno-dangling-reference)
  endif()
endif()

set(OFFLINE_RENDER_CORE_SOURCES
  src/cli.cpp
  src/json.cpp
  src/output.cpp
  src/recovery.cpp
  src/scene.cpp
)

add_library(offline_render_core STATIC ${OFFLINE_RENDER_CORE_SOURCES})
target_include_directories(offline_render_core PUBLIC src)

add_executable(offline_render_verify
  src/main.cpp
  src/context.cpp
  src/gl_api.cpp
  src/renderer.cpp
)
target_link_libraries(offline_render_verify PRIVATE offline_render_core)

if(WIN32)
  find_package(glfw3 3.4 CONFIG QUIET)
  if(NOT TARGET glfw AND NOT TARGET glfw3 AND NOT TARGET glfw3::glfw)
    set(OFFLINE_RENDER_GLFW_SOURCE_DIR "" CACHE PATH
      "Local GLFW 3.4 source tree used as an offline fallback")
    if(OFFLINE_RENDER_GLFW_SOURCE_DIR AND
       EXISTS "${OFFLINE_RENDER_GLFW_SOURCE_DIR}/CMakeLists.txt")
      add_subdirectory("${OFFLINE_RENDER_GLFW_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/glfw")
    else()
      message(FATAL_ERROR
        "GLFW 3.4 was not found. Install it with vcpkg, provide a CMake package "
        "prefix, or set OFFLINE_RENDER_GLFW_SOURCE_DIR to a local source tree.")
    endif()
  endif()
  if(TARGET glfw)
    set(OFFLINE_RENDER_GLFW_TARGET glfw)
  elseif(TARGET glfw3)
    set(OFFLINE_RENDER_GLFW_TARGET glfw3)
  elseif(TARGET glfw3::glfw)
    set(OFFLINE_RENDER_GLFW_TARGET glfw3::glfw)
  else()
    message(FATAL_ERROR "GLFW was found but no usable CMake target was exported")
  endif()
  target_link_libraries(offline_render_verify PRIVATE
    ${OFFLINE_RENDER_GLFW_TARGET} opengl32)
  target_compile_definitions(offline_render_verify PRIVATE OFFLINE_RENDER_WINDOWS=1)
else()
  target_link_libraries(offline_render_verify PRIVATE ${CMAKE_DL_LIBS})
endif()

add_executable(scene_generator tools/scene_gen.cpp)

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_FOUND)
    add_test(
      NAME offline_render_integration
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_tests.py"
        --renderer "$<TARGET_FILE:offline_render_verify>"
        --generator "$<TARGET_FILE:scene_generator>"
    )
    set_tests_properties(offline_render_integration PROPERTIES
      TIMEOUT 180
      ENVIRONMENT "LIBGL_ALWAYS_SOFTWARE=1"
    )
  endif()
endif()
