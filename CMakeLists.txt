cmake_minimum_required(VERSION 3.20)

project(voice-replay VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(voice-replay
    src/main.cpp
    src/error.cpp
    src/json.cpp
    src/sha256.cpp
    src/wav.cpp
    src/sample_gen.cpp
    src/exporter.cpp
)

target_include_directories(voice-replay PRIVATE src)

if(MSVC)
    target_compile_options(voice-replay PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(voice-replay PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(voice-replay PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

enable_testing()
if(WIN32)
    find_program(POWERSHELL_EXECUTABLE NAMES pwsh powershell)
    if(POWERSHELL_EXECUTABLE)
        add_test(NAME acceptance
            COMMAND "${POWERSHELL_EXECUTABLE}" -NoProfile -ExecutionPolicy Bypass
                    -File "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_acceptance.ps1"
                    "$<TARGET_FILE:voice-replay>"
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )
    endif()
else()
    add_test(NAME acceptance
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_acceptance.sh"
                "$<TARGET_FILE:voice-replay>"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()


