if(NOT DEFINED RENDER_EXE)
    message(FATAL_ERROR "RENDER_EXE is required")
endif()
if(NOT DEFINED WORK_DIR)
    message(FATAL_ERROR "WORK_DIR is required")
endif()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")
set(SAMPLE_DIR "${WORK_DIR}/samples")

execute_process(
    COMMAND "${RENDER_EXE}" --generate-samples "${SAMPLE_DIR}"
    RESULT_VARIABLE generation_result
    OUTPUT_VARIABLE generation_output
    ERROR_VARIABLE generation_error
    TIMEOUT 30
)
if(NOT generation_result EQUAL 0)
    message(FATAL_ERROR "sample generation failed: ${generation_error}")
endif()

set(SCENE "${SAMPLE_DIR}/scene.json")
set(FAST_ONE "${WORK_DIR}/fast_one.png")
set(FAST_TWO "${WORK_DIR}/fast_two.png")
set(HIGH "${WORK_DIR}/high.png")
set(FAST_LOG "${WORK_DIR}/fast.log")
set(HIGH_LOG "${WORK_DIR}/high.log")

execute_process(
    COMMAND "${RENDER_EXE}" --scene "${SCENE}" --mode fast
            --log "${FAST_LOG}" --output "${FAST_ONE}"
    RESULT_VARIABLE fast_one_result
    OUTPUT_VARIABLE fast_one_output
    ERROR_VARIABLE fast_one_error
    TIMEOUT 240
)
if(NOT fast_one_result EQUAL 0)
    message(FATAL_ERROR "fast render 1 failed: ${fast_one_error}")
endif()

execute_process(
    COMMAND "${RENDER_EXE}" --scene "${SCENE}" --mode fast
            --log "${WORK_DIR}/fast_two.log" --output "${FAST_TWO}"
    RESULT_VARIABLE fast_two_result
    OUTPUT_VARIABLE fast_two_output
    ERROR_VARIABLE fast_two_error
    TIMEOUT 240
)
if(NOT fast_two_result EQUAL 0)
    message(FATAL_ERROR "fast render 2 failed: ${fast_two_error}")
endif()

file(SHA256 "${FAST_ONE}" fast_one_hash)
file(SHA256 "${FAST_TWO}" fast_two_hash)
if(NOT fast_one_hash STREQUAL fast_two_hash)
    message(FATAL_ERROR "determinism failure: ${fast_one_hash} != ${fast_two_hash}")
endif()

execute_process(
    COMMAND "${RENDER_EXE}" --scene "${SCENE}" --mode high
            --log "${HIGH_LOG}" --output "${HIGH}"
    RESULT_VARIABLE high_result
    OUTPUT_VARIABLE high_output
    ERROR_VARIABLE high_error
    TIMEOUT 240
)
if(NOT high_result EQUAL 0)
    message(FATAL_ERROR "high render failed: ${high_error}")
endif()

file(SHA256 "${HIGH}" high_hash)
if(high_hash STREQUAL fast_one_hash)
    message(FATAL_ERROR "fast and high mode produced identical pixels")
endif()

execute_process(
    COMMAND "${RENDER_EXE}" --scene "${SCENE}" --mode high
            --log "${WORK_DIR}/high_two.log" --output "${WORK_DIR}/high_two.png"
    RESULT_VARIABLE high_two_result
    OUTPUT_VARIABLE high_two_output
    ERROR_VARIABLE high_two_error
    TIMEOUT 240
)
if(NOT high_two_result EQUAL 0)
    message(FATAL_ERROR "high render 2 failed: ${high_two_error}")
endif()
file(SHA256 "${WORK_DIR}/high_two.png" high_two_hash)
if(NOT high_hash STREQUAL high_two_hash)
    message(FATAL_ERROR "high-mode determinism failure: ${high_hash} != ${high_two_hash}")
endif()

file(READ "${FAST_ONE}" fast_png_hex HEX LIMIT 24)
file(READ "${HIGH}" high_png_hex HEX LIMIT 24)
if(NOT fast_png_hex MATCHES "^89504e470d0a1a0a")
    message(FATAL_ERROR "fast output is not a PNG")
endif()
if(NOT high_png_hex MATCHES "^89504e470d0a1a0a")
    message(FATAL_ERROR "high output is not a PNG")
endif()
if(NOT fast_png_hex MATCHES "0000028000000168")
    message(FATAL_ERROR "fast output dimensions are not 640x360")
endif()
if(NOT high_png_hex MATCHES "0000028000000168")
    message(FATAL_ERROR "high output dimensions are not 640x360")
endif()

file(READ "${FAST_LOG}" fast_log)
file(READ "${HIGH_LOG}" high_log)
file(READ "${WORK_DIR}/fast_two.log" fast_two_log)
file(READ "${WORK_DIR}/high_two.log" high_two_log)

foreach(LOG_NAME fast_log high_log fast_two_log high_two_log)
    if(NOT ${LOG_NAME} MATCHES "timestamp=20")
        message(FATAL_ERROR "${LOG_NAME} is missing an ISO timestamp")
    endif()
    if(NOT ${LOG_NAME} MATCHES "event=gpu_timings")
        message(FATAL_ERROR "${LOG_NAME} is missing gpu_timings")
    endif()
    if(NOT ${LOG_NAME} MATCHES "geometry_gpu_ms=")
        message(FATAL_ERROR "${LOG_NAME} is missing geometry_gpu_ms")
    endif()
    if(NOT ${LOG_NAME} MATCHES "lighting_gpu_ms=")
        message(FATAL_ERROR "${LOG_NAME} is missing lighting_gpu_ms")
    endif()
    if(NOT ${LOG_NAME} MATCHES "event=light_atlas")
        message(FATAL_ERROR "${LOG_NAME} is missing per-light atlas slots")
    endif()
    if(NOT ${LOG_NAME} MATCHES "event=shadow_stats")
        message(FATAL_ERROR "${LOG_NAME} is missing shadow stats")
    endif()
    if(NOT ${LOG_NAME} MATCHES "before_mean=")
        message(FATAL_ERROR "${LOG_NAME} is missing before_mean")
    endif()
    if(NOT ${LOG_NAME} MATCHES "after_mean=")
        message(FATAL_ERROR "${LOG_NAME} is missing after_mean")
    endif()
    if(NOT ${LOG_NAME} MATCHES "non_background_percent=")
        message(FATAL_ERROR "${LOG_NAME} is missing non_background_percent")
    endif()
    if(NOT ${LOG_NAME} MATCHES "event=resource_balance")
        message(FATAL_ERROR "${LOG_NAME} is missing resource balance")
    endif()
    if(${LOG_NAME} MATCHES "level=ERROR")
        message(FATAL_ERROR "${LOG_NAME} contains an error entry")
    endif()
endforeach()

if(NOT fast_log MATCHES "shadow_samples=1")
    message(FATAL_ERROR "fast mode did not log shadow_samples=1")
endif()
if(NOT fast_log MATCHES "atlas_tile_resolution=128")
    message(FATAL_ERROR "fast mode did not reduce atlas resolution to 128")
endif()
if(NOT fast_log MATCHES "gaussian_filter=0")
    message(FATAL_ERROR "fast mode did not skip Gaussian filtering")
endif()
if(NOT high_log MATCHES "shadow_samples=25")
    message(FATAL_ERROR "high mode did not increase shadow samples to 25")
endif()
if(NOT high_log MATCHES "atlas_tile_resolution=256")
    message(FATAL_ERROR "high mode did not use 256-texel tiles")
endif()
if(NOT high_log MATCHES "gaussian_filter=1")
    message(FATAL_ERROR "high mode did not enable Gaussian filtering")
endif()
if(NOT high_log MATCHES "filter_iterations=2")
    message(FATAL_ERROR "high mode did not use two post-process iterations")
endif()
if(NOT high_log MATCHES "face_count=6")
    message(FATAL_ERROR "point light atlas cube-face allocation was not logged")
endif()
if(NOT fast_log MATCHES "non_background_percent=([0-9]+\\.[0-9]+)")
    message(FATAL_ERROR "cannot parse non_background_percent")
endif()
set(non_background_percent "${CMAKE_MATCH_1}")
if(non_background_percent LESS 1.0)
    message(FATAL_ERROR "rendered image has too few foreground pixels: ${non_background_percent}")
endif()

execute_process(
    COMMAND "${RENDER_EXE}" --scene "${SCENE}" --validate
            --log "${WORK_DIR}/validate.log"
    RESULT_VARIABLE validate_result
    OUTPUT_VARIABLE validate_output
    ERROR_VARIABLE validate_error
    TIMEOUT 30
)
if(NOT validate_result EQUAL 0)
    message(FATAL_ERROR "valid scene validation failed: ${validate_error}")
endif()

execute_process(
    COMMAND "${RENDER_EXE}" --scene "${SAMPLE_DIR}/invalid_missing_camera.json"
            --mode fast --log "${WORK_DIR}/invalid.log"
    RESULT_VARIABLE invalid_result
    OUTPUT_VARIABLE invalid_output
    ERROR_VARIABLE invalid_error
    TIMEOUT 30
)
if(NOT invalid_result EQUAL 1)
    message(FATAL_ERROR "invalid scene did not exit with code 1: ${invalid_result}")
endif()
if(NOT invalid_error MATCHES "E_INVALID_SCENE")
    message(FATAL_ERROR "invalid scene stderr lacks E_INVALID_SCENE: ${invalid_error}")
endif()

execute_process(
    COMMAND "${RENDER_EXE}" --scene "${SAMPLE_DIR}/invalid_atlas_capacity.json"
            --mode high --log "${WORK_DIR}/invalid_atlas.log"
    RESULT_VARIABLE atlas_result
    OUTPUT_VARIABLE atlas_output
    ERROR_VARIABLE atlas_error
    TIMEOUT 30
)
if(NOT atlas_result EQUAL 1)
    message(FATAL_ERROR "invalid atlas did not exit with code 1: ${atlas_result}")
endif()
if(NOT atlas_error MATCHES "E_INVALID_SCENE")
    message(FATAL_ERROR "invalid atlas stderr lacks E_INVALID_SCENE: ${atlas_error}")
endif()

execute_process(
    COMMAND "${RENDER_EXE}" --scene "${SCENE}" --mode fast
            --log "${WORK_DIR}/self_test.log" --self-test-resources
    RESULT_VARIABLE self_test_result
    OUTPUT_VARIABLE self_test_output
    ERROR_VARIABLE self_test_error
    TIMEOUT 60
)
if(NOT self_test_result EQUAL 0)
    message(FATAL_ERROR "resource self-test failed: ${self_test_error}")
endif()
file(READ "${WORK_DIR}/self_test.log" self_test_log)
if(NOT self_test_log MATCHES "event=resource_self_test")
    message(FATAL_ERROR "resource self-test event is missing")
endif()
if(NOT self_test_log MATCHES "resource_balance=0")
    message(FATAL_ERROR "resource self-test did not report a zero balance")
endif()

message(STATUS "integration=passed deterministic=yes fast_sha256=${fast_one_hash} high_sha256=${high_hash}")
