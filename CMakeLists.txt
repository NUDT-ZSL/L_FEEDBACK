cmake_minimum_required(VERSION 3.20)

project(lf_baker VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(lf_baker
    src/atomic_io.cpp
    src/cache.cpp
    src/image.cpp
    src/json.cpp
    src/main.cpp
    src/render.cpp
    src/scene.cpp
    src/sha256.cpp
)

target_include_directories(lf_baker PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)

if(MSVC)
    target_compile_options(lf_baker PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(lf_baker PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(MSVC)
    set_property(TARGET lf_baker PROPERTY LINK_FLAGS "/SUBSYSTEM:CONSOLE")
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(lf_baker_unit_tests
        tests/unit_tests.cpp
        src/json.cpp
        src/sha256.cpp
    )
    target_include_directories(lf_baker_unit_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    if(MSVC)
        target_compile_options(lf_baker_unit_tests PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(lf_baker_unit_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME lf_baker_unit COMMAND lf_baker_unit_tests)

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME lf_baker_integration
            COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_tests.py"
                    "${CMAKE_CURRENT_SOURCE_DIR}"
                    "$<TARGET_FILE:lf_baker>"
                    "${CMAKE_CURRENT_BINARY_DIR}/test-work"
        )
        set_tests_properties(lf_baker_integration PROPERTIES TIMEOUT 180)
    endif()
endif()
