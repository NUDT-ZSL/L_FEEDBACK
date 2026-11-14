add_executable(lipsync_unit_tests
    unit_tests.cpp
)
target_link_libraries(lipsync_unit_tests PRIVATE lipsync_core)
add_test(NAME lipsync_unit_tests COMMAND lipsync_unit_tests)

add_executable(lipsync_integration_test
    integration_test.cpp
)
target_link_libraries(lipsync_integration_test PRIVATE lipsync_core)
add_test(NAME lipsync_integration_test COMMAND lipsync_integration_test)

add_executable(lipsync_concurrency_test
    concurrency_test.cpp
)
target_link_libraries(lipsync_concurrency_test PRIVATE lipsync_core)
add_test(NAME lipsync_concurrency_test COMMAND lipsync_concurrency_test)
