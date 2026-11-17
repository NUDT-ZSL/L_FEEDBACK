cmake_minimum_required(VERSION 3.20)

project(TerrainSoftRenderer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(TERRAIN_STATIC_RUNTIME "Use the static MSVC runtime" OFF)

add_library(terrain_core
  src/data.cpp
  src/image_io.cpp
  src/json_util.cpp
  src/renderer.cpp
)

target_include_directories(terrain_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_compile_definitions(terrain_core PRIVATE
  NOMINMAX
  _CRT_SECURE_NO_WARNINGS
)

add_executable(terrain_renderer src/main.cpp)
target_link_libraries(terrain_renderer PRIVATE terrain_core)
target_compile_definitions(terrain_renderer PRIVATE
  NOMINMAX
  _CRT_SECURE_NO_WARNINGS
)

if(MSVC)
  target_compile_options(terrain_core PRIVATE /W4 /permissive- /utf-8)
  target_compile_options(terrain_renderer PRIVATE /W4 /permissive- /utf-8)
  if(TERRAIN_STATIC_RUNTIME)
    set_property(TARGET terrain_core terrain_renderer PROPERTY
      MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()
else()
  target_compile_options(terrain_core PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(terrain_renderer PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(terrain_renderer PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS terrain_renderer RUNTIME DESTINATION bin)
install(DIRECTORY third_party/ DESTINATION third_party)

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(
      NAME terrain_acceptance
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_acceptance.py
              --exe $<TARGET_FILE:terrain_renderer>
              --work-dir ${CMAKE_BINARY_DIR}/acceptance_test
    )
  endif()
endif()

