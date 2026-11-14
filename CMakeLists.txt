cmake_minimum_required(VERSION 3.20)

project(SpatialAudioMixer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SPATIAL_AUDIO_BUILD_TESTS "Build the deterministic test suite" ON)
option(SPATIAL_AUDIO_BUILD_REALTIME "Build the PortAudio real-time executable" ON)
set(SPATIAL_AUDIO_LOCAL_DEPS "${CMAKE_CURRENT_SOURCE_DIR}/.deps/install")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# --- libsndfile -----------------------------------------------------------
# Preferred: an installed CMake package. The fallback intentionally supports
# the common Windows prebuilt layout: LIBSNDFILE_DIR/include + LIBSNDFILE_DIR/lib.
find_package(SndFile CONFIG QUIET
    HINTS
        "$ENV{LIBSNDFILE_DIR}"
        "${SPATIAL_AUDIO_LOCAL_DEPS}"
        "$ENV{LIBSNDFILE_DIR}/lib/cmake/SndFile"
        "${SPATIAL_AUDIO_LOCAL_DEPS}/lib/cmake/SndFile"
        "$ENV{LIBSNDFILE_DIR}/cmake"
)

if(NOT TARGET SndFile::sndfile)
    find_path(SNDFILE_INCLUDE_DIR
        NAMES sndfile.h
        HINTS "$ENV{LIBSNDFILE_DIR}" "${SPATIAL_AUDIO_LOCAL_DEPS}"
        PATH_SUFFIXES include
    )
    find_library(SNDFILE_LIBRARY
        NAMES sndfile libsndfile libsndfile-1 sndfile-static
        HINTS "$ENV{LIBSNDFILE_DIR}" "${SPATIAL_AUDIO_LOCAL_DEPS}"
        PATH_SUFFIXES lib lib/x64 lib/x86 lib/static
    )
    if(NOT SNDFILE_INCLUDE_DIR OR NOT SNDFILE_LIBRARY)
        message(FATAL_ERROR
            "libsndfile 1.2.0+ was not found. Set LIBSNDFILE_DIR to a prefix "
            "containing include/sndfile.h and lib/<sndfile library>.")
    endif()
    add_library(SndFile::sndfile UNKNOWN IMPORTED)
    set_target_properties(SndFile::sndfile PROPERTIES
        IMPORTED_LOCATION "${SNDFILE_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SNDFILE_INCLUDE_DIR}"
    )
endif()

add_library(spatial_audio_core STATIC
    src/audio_engine.cpp
    src/wav_io.cpp
)
target_include_directories(spatial_audio_core
    PUBLIC include
    PRIVATE src
)
target_link_libraries(spatial_audio_core PUBLIC SndFile::sndfile)
target_compile_features(spatial_audio_core PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(spatial_audio_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(spatial_audio_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- PortAudio ------------------------------------------------------------
if(SPATIAL_AUDIO_BUILD_REALTIME)
    find_package(PortAudio CONFIG QUIET
        HINTS
            "$ENV{PORTAUDIO_DIR}"
            "${SPATIAL_AUDIO_LOCAL_DEPS}"
            "$ENV{PORTAUDIO_DIR}/lib/cmake/portaudio"
            "${SPATIAL_AUDIO_LOCAL_DEPS}/lib/cmake/portaudio"
            "$ENV{PORTAUDIO_DIR}/cmake"
    )
    find_package(portaudio CONFIG QUIET
        HINTS
            "$ENV{PORTAUDIO_DIR}"
            "${SPATIAL_AUDIO_LOCAL_DEPS}"
            "$ENV{PORTAUDIO_DIR}/lib/cmake/portaudio"
            "${SPATIAL_AUDIO_LOCAL_DEPS}/lib/cmake/portaudio"
            "$ENV{PORTAUDIO_DIR}/cmake"
    )

    if(TARGET PortAudio::PortAudio)
        set(SPATIAL_PORTAUDIO_TARGET PortAudio::PortAudio)
    elseif(TARGET portaudio)
        set(SPATIAL_PORTAUDIO_TARGET portaudio)
    elseif(TARGET portaudio_static)
        set(SPATIAL_PORTAUDIO_TARGET portaudio_static)
    else()
        find_path(PORTAUDIO_INCLUDE_DIR
            NAMES portaudio.h
            HINTS "$ENV{PORTAUDIO_DIR}" "${SPATIAL_AUDIO_LOCAL_DEPS}"
            PATH_SUFFIXES include
        )
        find_library(PORTAUDIO_LIBRARY
            NAMES portaudio portaudio_x64 portaudio_x86 libportaudio
            HINTS "$ENV{PORTAUDIO_DIR}" "${SPATIAL_AUDIO_LOCAL_DEPS}"
            PATH_SUFFIXES lib lib/x64 lib/x86 lib/static
        )
        if(NOT PORTAUDIO_INCLUDE_DIR OR NOT PORTAUDIO_LIBRARY)
            message(FATAL_ERROR
                "PortAudio 19.7.0+ was not found. Set PORTAUDIO_DIR to a prefix "
                "containing include/portaudio.h and lib/<portaudio library>, "
                "or configure with -DSPATIAL_AUDIO_BUILD_REALTIME=OFF.")
        endif()
        add_library(PortAudio::PortAudio UNKNOWN IMPORTED)
        set_target_properties(PortAudio::PortAudio PROPERTIES
            IMPORTED_LOCATION "${PORTAUDIO_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${PORTAUDIO_INCLUDE_DIR}"
        )
        if(WIN32)
            set_property(TARGET PortAudio::PortAudio PROPERTY
                INTERFACE_LINK_LIBRARIES "winmm;ole32")
        elseif(UNIX)
            find_package(Threads REQUIRED)
            set_property(TARGET PortAudio::PortAudio PROPERTY
                INTERFACE_LINK_LIBRARIES "Threads::Threads;${CMAKE_DL_LIBS}")
        endif()
        set(SPATIAL_PORTAUDIO_TARGET PortAudio::PortAudio)
    endif()
endif()

add_executable(spatial_audio_mixer
    src/main.cpp
    src/portaudio_output.cpp
    src/scenario.cpp
)
target_include_directories(spatial_audio_mixer PRIVATE src)
target_link_libraries(spatial_audio_mixer PRIVATE spatial_audio_core)
if(SPATIAL_AUDIO_BUILD_REALTIME)
    target_link_libraries(spatial_audio_mixer PRIVATE ${SPATIAL_PORTAUDIO_TARGET})
    target_compile_definitions(spatial_audio_mixer PRIVATE SPATIAL_AUDIO_HAS_PORTAUDIO=1)
else()
    target_compile_definitions(spatial_audio_mixer PRIVATE SPATIAL_AUDIO_HAS_PORTAUDIO=0)
endif()

if(MSVC)
    target_compile_options(spatial_audio_mixer PRIVATE /W4 /permissive-)
else()
    target_compile_options(spatial_audio_mixer PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(SPATIAL_AUDIO_BUILD_TESTS)
    add_executable(spatial_audio_tests tests/audio_engine_tests.cpp)
    target_include_directories(spatial_audio_tests PRIVATE src)
    target_link_libraries(spatial_audio_tests PRIVATE spatial_audio_core)
    if(MSVC)
        target_compile_options(spatial_audio_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(spatial_audio_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME audio_engine_tests COMMAND spatial_audio_tests)
    add_test(
        NAME deterministic_mix
        COMMAND spatial_audio_mixer --deterministic --duration 4
                --out "${CMAKE_CURRENT_BINARY_DIR}/deterministic_mix.wav"
                --sample-dir "${CMAKE_CURRENT_BINARY_DIR}/deterministic_samples"
    )
endif()

