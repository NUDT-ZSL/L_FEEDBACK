cmake_minimum_required(VERSION 3.20)
project(lipsync_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# FFmpeg 依赖：优先 find_package（vcpkg/系统安装），其次 FFmpeg_ROOT 手动路径。
# ---------------------------------------------------------------------------
find_package(FFmpeg QUIET COMPONENTS avformat avcodec avutil swresample)

if(NOT FFmpeg_FOUND)
    if(NOT DEFINED FFmpeg_ROOT AND DEFINED ENV{FFMPEG_ROOT})
        set(FFmpeg_ROOT "$ENV{FFMPEG_ROOT}")
    endif()

    if(FFmpeg_ROOT)
        if(WIN32)
            set(_ffmpeg_inc_dir "${FFmpeg_ROOT}/include")
            if(EXISTS "${FFmpeg_ROOT}/lib")
                set(_ffmpeg_lib_dir "${FFmpeg_ROOT}/lib")
            else()
                set(_ffmpeg_lib_dir "${FFmpeg_ROOT}/bin")
            endif()
        else()
            find_path(_ffmpeg_inc_dir libavformat/avformat.h
                      PATHS "${FFmpeg_ROOT}/include"
                      PATH_SUFFIXES x86_64-linux-gnu)
            find_library(_avformat NAMES avformat PATHS
                         "${FFmpeg_ROOT}/lib" "${FFmpeg_ROOT}/lib/x86_64-linux-gnu")
            get_filename_component(_ffmpeg_lib_dir "${_avformat}" DIRECTORY)
        endif()

        # Debian/Ubuntu 的 FFmpeg 共享库以 -z defs 构建，手动指定前缀时
        # 其传递依赖（va/X11/编解码第三方库）不一定全部可见；允许共享库
        # 内部保留未解析符号，运行时由 SONAME 正常加载。Windows 不受影响。
        set(_ffmpeg_link_options "")
        if(UNIX AND NOT APPLE)
            set(_ffmpeg_link_options "-Wl,--unresolved-symbols=ignore-in-shared-libs")
        endif()

        add_library(FFmpeg::avutil INTERFACE IMPORTED)
        target_include_directories(FFmpeg::avutil INTERFACE "${_ffmpeg_inc_dir}")
        target_link_directories(FFmpeg::avutil INTERFACE "${_ffmpeg_lib_dir}")
        target_link_libraries(FFmpeg::avutil INTERFACE avutil)
        if(_ffmpeg_link_options)
            target_link_options(FFmpeg::avutil INTERFACE "${_ffmpeg_link_options}")
        endif()

        add_library(FFmpeg::swresample INTERFACE IMPORTED)
        target_include_directories(FFmpeg::swresample INTERFACE "${_ffmpeg_inc_dir}")
        target_link_directories(FFmpeg::swresample INTERFACE "${_ffmpeg_lib_dir}")
        target_link_libraries(FFmpeg::swresample INTERFACE swresample avutil)

        add_library(FFmpeg::avcodec INTERFACE IMPORTED)
        target_include_directories(FFmpeg::avcodec INTERFACE "${_ffmpeg_inc_dir}")
        target_link_directories(FFmpeg::avcodec INTERFACE "${_ffmpeg_lib_dir}")
        target_link_libraries(FFmpeg::avcodec INTERFACE avcodec avutil)

        add_library(FFmpeg::avformat INTERFACE IMPORTED)
        target_include_directories(FFmpeg::avformat INTERFACE "${_ffmpeg_inc_dir}")
        target_link_directories(FFmpeg::avformat INTERFACE "${_ffmpeg_lib_dir}")
        target_link_libraries(FFmpeg::avformat INTERFACE avformat avcodec avutil)
    else()
        message(FATAL_ERROR
                "FFmpeg development files not found. Install FFmpeg 5.1+ dev package "
                "or pass -DFFmpeg_ROOT=<path> (set FFMPEG_ROOT on Windows).")
    endif()
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# 可链接库 lipsync_core
# ---------------------------------------------------------------------------
add_library(lipsync_core STATIC
    src/lipsync/json.cpp
    src/lipsync/config.cpp
    src/lipsync/viseme_mapper.cpp
    src/lipsync/phoneme_parser.cpp
    src/lipsync/audio_processor.cpp
    src/lipsync/lip_sync.cpp
    src/lipsync/pipeline.cpp
)
target_include_directories(lipsync_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(lipsync_core PUBLIC
    FFmpeg::avformat
    FFmpeg::avcodec
    FFmpeg::avutil
    FFmpeg::swresample
    Threads::Threads
)

# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
add_executable(lipsync_cli app/main.cpp)
set_target_properties(lipsync_cli PROPERTIES OUTPUT_NAME lipsync)
target_link_libraries(lipsync_cli PRIVATE lipsync_core)

# ---------------------------------------------------------------------------
# 确定性样本生成器（可选 Python；存在时注册 CMake 目标，否则用批处理脚本）
# ---------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_custom_target(generate_samples
        COMMAND ${CMAKE_COMMAND} -E echo
                "Generating deterministic samples via tools/generate_samples.py"
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_samples.py"
                --output "${CMAKE_BINARY_DIR}/samples"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_samples.py"
        VERBATIM
        USES_TERMINAL
    )

    add_custom_target(run_demo
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/output"
        COMMAND lipsync_cli
                --input "${CMAKE_BINARY_DIR}/samples"
                --output "${CMAKE_BINARY_DIR}/output"
                --config "${CMAKE_CURRENT_SOURCE_DIR}/config/default.json"
        DEPENDS lipsync_cli generate_samples
        VERBATIM
        USES_TERMINAL
    )
endif()

enable_testing()
add_subdirectory(tests)
