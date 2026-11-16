cmake_minimum_required(VERSION 3.20)

project(LayeredCharacterRenderer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LCR_BUILD_TESTS "Build deterministic core tests" ON)
if(WIN32)
  option(LCR_BUILD_APP "Build the OpenGL desktop demo" ON)
else()
  option(LCR_BUILD_APP "Build the OpenGL desktop demo (requires GLFW/GLAD/GLM)" OFF)
endif()

add_library(lcr_core STATIC
  src/core/Animation.cpp
  src/core/CapeSorter.cpp
  src/core/Cli.cpp
  src/core/Diagnostics.cpp
  src/core/Equipment.cpp
  src/core/Error.cpp
  src/core/Mesh.cpp
  src/core/Report.cpp
  src/core/Serializer.cpp
  src/core/Simulation.cpp
  src/core/Skeleton.cpp
)
target_include_directories(lcr_core PUBLIC src)
target_compile_features(lcr_core PUBLIC cxx_std_17)
if(MSVC)
  target_compile_options(lcr_core PRIVATE /W4 /permissive-)
  target_compile_definitions(lcr_core PUBLIC _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  target_compile_options(lcr_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(character_headless src/headless_main.cpp)
target_link_libraries(character_headless PRIVATE lcr_core)

if(LCR_BUILD_TESTS)
  enable_testing()
  add_executable(character_tests tests/core_tests.cpp)
  target_link_libraries(character_tests PRIVATE lcr_core)
  add_test(NAME core_determinism COMMAND character_tests)

  add_test(NAME fault_texture_exit
    COMMAND ${CMAKE_COMMAND}
      -DEXE=$<TARGET_FILE:character_headless>
      -DFAULT=texture
      -DEXPECTED_EXIT=20
      -DEXPECTED_CODE=E_TEXTURE_GENERATION
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/assert_fault.cmake)
  add_test(NAME fault_shader_exit
    COMMAND ${CMAKE_COMMAND}
      -DEXE=$<TARGET_FILE:character_headless>
      -DFAULT=shader
      -DEXPECTED_EXIT=30
      -DEXPECTED_CODE=E_SHADER_COMPILE
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/assert_fault.cmake)
  add_test(NAME fault_bone_exit
    COMMAND ${CMAKE_COMMAND}
      -DEXE=$<TARGET_FILE:character_headless>
      -DFAULT=bone
      -DEXPECTED_EXIT=40
      -DEXPECTED_CODE=E_BONE_INDEX
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/assert_fault.cmake)
endif()

option(LCR_BUILD_OFFSCREEN "Build the OSMesa offscreen render validator" OFF)
if(LCR_BUILD_OFFSCREEN)
  find_package(glad CONFIG REQUIRED)
  find_package(glm CONFIG REQUIRED)
  find_path(OSMESA_INCLUDE_DIR GL/osmesa.h)
  find_library(OSMESA_LIBRARY NAMES OSMesa)
  if(NOT OSMESA_INCLUDE_DIR OR NOT OSMESA_LIBRARY)
    message(FATAL_ERROR "LCR_BUILD_OFFSCREEN requires OSMesa headers and library")
  endif()
  add_executable(character_offscreen
    tools/offscreen_render.cpp
    src/render/GLRenderer.cpp
    src/render/ShaderProgram.cpp
  )
  target_include_directories(character_offscreen PRIVATE src ${OSMESA_INCLUDE_DIR})
  target_link_libraries(character_offscreen PRIVATE
    lcr_core
    glad::glad
    glm::glm
    ${OSMESA_LIBRARY}
    ${CMAKE_DL_LIBS}
  )
  if(MSVC)
    target_compile_options(character_offscreen PRIVATE /W4 /permissive-)
    target_compile_definitions(character_offscreen PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
  else()
    target_compile_options(character_offscreen PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()

if(LCR_BUILD_APP)
  find_package(OpenGL REQUIRED)
  find_package(glfw3 CONFIG REQUIRED)
  find_package(glad CONFIG REQUIRED)
  find_package(glm CONFIG REQUIRED)

  add_executable(character_demo
    src/window_main.cpp
    src/render/GLRenderer.cpp
    src/render/ShaderProgram.cpp
  )
  target_link_libraries(character_demo PRIVATE
    lcr_core
    glfw
    glad::glad
    glm::glm
    OpenGL::GL
  )
  if(MSVC)
    target_compile_options(character_demo PRIVATE /W4 /permissive-)
    target_compile_definitions(character_demo PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
  else()
    target_compile_options(character_demo PRIVATE -Wall -Wextra -Wpedantic)
  endif()

  add_custom_command(TARGET character_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:character_demo>/reports"
    COMMENT "Create reports directory for character_demo"
  )
endif()

install(TARGETS character_headless RUNTIME DESTINATION bin)
if(LCR_BUILD_APP)
  install(TARGETS character_demo RUNTIME DESTINATION bin)
endif()

