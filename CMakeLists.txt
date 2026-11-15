cmake_minimum_required(VERSION 3.20)

project(automix VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(AUTOMIX_BUILD_TESTS "Build the DSP and integration tests" ON)

set(_sndfile_hints "$ENV{LIBSNDFILE_DIR}")
find_path(SNDFILE_INCLUDE_DIR
    NAMES sndfile.h
    HINTS ${_sndfile_hints}
    PATH_SUFFIXES include)
find_library(SNDFILE_LIBRARY
    NAMES sndfile libsndfile-1 libsndfile
    HINTS ${_sndfile_hints}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)

if(NOT SNDFILE_INCLUDE_DIR OR NOT SNDFILE_LIBRARY)
    message(FATAL_ERROR
        "libsndfile was not found. Set LIBSNDFILE_DIR to the installation prefix "
        "(it must contain include/sndfile.h and lib/libsndfile).")
endif()

message(STATUS "libsndfile header: ${SNDFILE_INCLUDE_DIR}")
message(STATUS "libsndfile library: ${SNDFILE_LIBRARY}")

add_library(automix_core STATIC
    src/config.cpp
    src/drc.cpp
    src/engine.cpp
    src/json.cpp
    src/limiter.cpp
    src/logger.cpp
    src/loudness.cpp
    src/resampler.cpp
    src/stereo.cpp
    src/wav_io.cpp
)

target_include_directories(automix_core
    PUBLIC include
    PRIVATE ${SNDFILE_INCLUDE_DIR})
target_compile_definitions(automix_core PRIVATE AUTOMIX_VERSION="${PROJECT_VERSION}")
target_link_libraries(automix_core PUBLIC ${SNDFILE_LIBRARY})

if(MSVC)
    target_compile_options(automix_core PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(automix_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(automix src/main.cpp)
target_link_libraries(automix PRIVATE automix_core)
if(MINGW)
    target_link_options(automix PRIVATE -municode)
endif()
if(UNIX AND NOT APPLE)
    get_filename_component(_sndfile_library_dir "${SNDFILE_LIBRARY}" DIRECTORY)
    target_link_options(automix PRIVATE
        "-Wl,-rpath-link,${_sndfile_library_dir}")
endif()

function(automix_copy_runtime_dlls target)
    if(WIN32)
        foreach(_dir "$ENV{LIBSNDFILE_DIR}/bin" "$ENV{LIBSNDFILE_DIR}" "$ENV{FFTW_DIR}")
            if(EXISTS "${_dir}")
                file(GLOB _dlls "${_dir}/*.dll")
                if(_dlls)
                    add_custom_command(TARGET ${target} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_dlls}
                                "$<TARGET_FILE_DIR:${target}>"
                        VERBATIM)
                endif()
            endif()
        endforeach()
    endif()
endfunction()

automix_copy_runtime_dlls(automix)

if(AUTOMIX_BUILD_TESTS)
    enable_testing()
    find_package(Python3 COMPONENTS Interpreter)

    set(_fftw_hints "$ENV{FFTW_DIR}")
    find_path(FFTW_INCLUDE_DIR
        NAMES fftw3.h
        HINTS ${_fftw_hints}
        PATH_SUFFIXES include)
    find_library(FFTW3F_LIBRARY
        NAMES fftw3f libfftw3f-3 libfftw3f
        HINTS ${_fftw_hints}
        PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)

    if(NOT FFTW_INCLUDE_DIR OR NOT FFTW3F_LIBRARY)
        message(FATAL_ERROR
            "AUTOMIX_BUILD_TESTS is ON but FFTW 3 single precision was not found. "
            "Set FFTW_DIR or configure with -DAUTOMIX_BUILD_TESTS=OFF.")
    endif()

    add_executable(automix_tests
        tests/test_main.cpp
        tests/test_dsp.cpp
    )
    target_include_directories(automix_tests PRIVATE tests ${FFTW_INCLUDE_DIR})
    target_link_libraries(automix_tests PRIVATE automix_core ${FFTW3F_LIBRARY})
    if(UNIX AND NOT APPLE)
        get_filename_component(_fftw_library_dir "${FFTW3F_LIBRARY}" DIRECTORY)
        target_link_options(automix_tests PRIVATE
            "-Wl,-rpath-link,${_sndfile_library_dir}"
            "-Wl,-rpath-link,${_fftw_library_dir}")
    endif()
    automix_copy_runtime_dlls(automix_tests)

    add_test(NAME unit_and_dsp COMMAND automix_tests)

    if(Python3_Interpreter_FOUND)
        add_test(
            NAME sample_acceptance
            COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/scripts/acceptance_test.py"
                --exe "$<TARGET_FILE:automix>"
                --work-dir "${CMAKE_CURRENT_BINARY_DIR}/acceptance")
    endif()
endif()

install(TARGETS automix RUNTIME DESTINATION bin)
