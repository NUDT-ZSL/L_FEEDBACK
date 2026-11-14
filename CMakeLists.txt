cmake_minimum_required(VERSION 3.25)

project(lf_layered_character VERSION 1.0.0 LANGUAGES CXX)

option(LF_BUILD_APP "Build the interactive OpenGL application" ON)
option(LF_BUILD_TESTS "Build deterministic core tests" ON)
option(LF_FETCH_DEPS "Fetch pinned dependencies when vcpkg packages are absent" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}/bin")
endforeach()

find_package(glm 0.9.9.8 CONFIG QUIET)
find_package(nlohmann_json 3.11.2 CONFIG QUIET)

if(LF_FETCH_DEPS)
  include(FetchContent)

  if(NOT glm_FOUND)
    FetchContent_Declare(
      glm
      URL https://github.com/g-truc/glm/archive/refs/tags/0.9.9.8.tar.gz
      URL_HASH SHA256=7d508ab72cb5d43227a3711420f06ff99b0a0cb63ee2f93631b162bfe1fe9592
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(GLM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLM_BUILD_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glm)
  endif()

  if(NOT nlohmann_json_FOUND)
    FetchContent_Declare(
      nlohmann_json
      URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
      URL_HASH SHA256=8c4b26bf4b422252e13f332bc5e388ec0ab5c3443d24399acb675e68278d341f
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(nlohmann_json)
  endif()
endif()

if(NOT TARGET glm::glm)
  message(FATAL_ERROR "GLM 0.9.9.8 was not found. Install it with vcpkg or enable LF_FETCH_DEPS.")
endif()
if(NOT TARGET nlohmann_json::nlohmann_json)
  message(FATAL_ERROR "nlohmann-json 3.11.2 was not found. Install it with vcpkg or enable LF_FETCH_DEPS.")
endif()

add_library(lf_core STATIC
  src/core/brdf.cpp
  src/core/config.cpp
  src/core/config_store.cpp
  src/core/error_code.cpp
  src/core/log.cpp
  src/core/mesh.cpp
  src/core/samples.cpp
  src/core/snapshot.cpp
)
target_include_directories(lf_core PUBLIC include)
target_link_libraries(lf_core PUBLIC glm::glm nlohmann_json::nlohmann_json)
target_compile_definitions(lf_core PUBLIC LF_VERSION="${PROJECT_VERSION}")

if(MSVC)
  target_compile_options(lf_core PRIVATE /W4 /permissive- /utf-8)
else()
  target_compile_options(lf_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

if(LF_BUILD_TESTS)
  enable_testing()
  add_executable(lf_core_tests tests/test_main.cpp)
  target_link_libraries(lf_core_tests PRIVATE lf_core)
  if(MSVC)
    target_compile_options(lf_core_tests PRIVATE /W4 /permissive- /utf-8)
  else()
    target_compile_options(lf_core_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  endif()
  add_test(NAME lf_core_tests COMMAND lf_core_tests)
endif()

if(LF_BUILD_APP)
  find_package(glfw3 3.3.8 CONFIG QUIET)
  if(NOT glfw3_FOUND AND LF_FETCH_DEPS)
    include(FetchContent)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    if(WIN32)
      set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
      set(GLFW_BUILD_X11 OFF CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
      glfw
      URL https://github.com/glfw/glfw/archive/refs/tags/3.3.8.tar.gz
      URL_HASH SHA256=f30f42e05f11e5fc62483e513b0488d5bceeab7d9c5da0ffe2252ad81816c713
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(glfw)
  endif()

  if(TARGET glfw)
    set(LF_GLFW_TARGET glfw)
  elseif(TARGET glfw3)
    set(LF_GLFW_TARGET glfw3)
  else()
    message(FATAL_ERROR "GLFW 3.3.8 was not found. Install it with vcpkg or enable LF_FETCH_DEPS.")
  endif()

  add_executable(lf_character_engine
    src/gl/gl_api.cpp
    src/gl/renderer.cpp
    src/main.cpp
  )
  target_include_directories(lf_character_engine PRIVATE include)
  target_link_libraries(lf_character_engine PRIVATE lf_core ${LF_GLFW_TARGET})
  if(WIN32)
    target_link_libraries(lf_character_engine PRIVATE opengl32)
  endif()
  if(MSVC)
    target_compile_options(lf_character_engine PRIVATE /W4 /permissive- /utf-8)
  else()
    target_compile_options(lf_character_engine PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    find_package(Threads REQUIRED)
    target_link_libraries(lf_character_engine PRIVATE Threads::Threads)
  endif()

  add_custom_command(
    TARGET lf_character_engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/assets/shaders"
            "$<TARGET_FILE_DIR:lf_character_engine>/shaders"
    COMMENT "Copying GLSL shaders next to lf_character_engine"
  )
endif()

