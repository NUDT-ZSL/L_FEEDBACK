if(NOT DEFINED EXE OR NOT DEFINED FAULT OR NOT DEFINED EXPECTED_EXIT OR NOT DEFINED EXPECTED_CODE)
  message(FATAL_ERROR "Missing fault-test arguments")
endif()

execute_process(
  COMMAND "${EXE}" --headless --fault "${FAULT}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout_text
  ERROR_VARIABLE stderr_text
)

if(NOT "${result}" STREQUAL "${EXPECTED_EXIT}")
  message(FATAL_ERROR "Expected exit ${EXPECTED_EXIT}, got ${result}\nstdout:\n${stdout_text}\nstderr:\n${stderr_text}")
endif()
string(FIND "${stderr_text}" "\"code\":\"${EXPECTED_CODE}\"" code_pos)
if(code_pos EQUAL -1)
  message(FATAL_ERROR "stderr does not contain ${EXPECTED_CODE}\nstderr:\n${stderr_text}")
endif()
if(NOT "${stderr_text}" MATCHES "\"stage\":\"")
  message(FATAL_ERROR "stderr is not structured JSON\nstderr:\n${stderr_text}")
endif()

