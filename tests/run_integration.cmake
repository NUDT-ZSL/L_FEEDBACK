if(NOT DEFINED TEST_EXE)
    message(FATAL_ERROR "TEST_EXE is required")
endif()
if(NOT DEFINED TEST_ROOT)
    message(FATAL_ERROR "TEST_ROOT is required")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}")

set(DATA_DIR "${TEST_ROOT}/data")
set(OUTPUT_DIR "${TEST_ROOT}/output")

function(run_case name expect_success)
    execute_process(
        COMMAND ${ARGN}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
        TIMEOUT 180
    )
    set("${name}_RESULT" "${result}" PARENT_SCOPE)
    set("${name}_OUTPUT" "${output}" PARENT_SCOPE)
    set("${name}_ERROR" "${error}" PARENT_SCOPE)
    if(expect_success)
        if(NOT result EQUAL 0)
            message(FATAL_ERROR
                "${name} failed with ${result}\nstdout:\n${output}\nstderr:\n${error}")
        endif()
    else()
        if(result EQUAL 0)
            message(FATAL_ERROR "${name} unexpectedly succeeded\nstdout:\n${output}")
        endif()
        if(NOT result EQUAL 1)
            message(FATAL_ERROR
                "${name} used process exit code ${result}; business errors must map to 1")
        endif()
    endif()
endfunction()

function(require_text name haystack needle)
    string(FIND "${haystack}" "${needle}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "${name} did not contain '${needle}'\nactual:\n${haystack}")
    endif()
endfunction()

# First render generates all deterministic samples and the initial snapshot.
run_case(render TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --backend cpu --render)
require_text(render "${render_OUTPUT}" "rendered backend=cpu")
if(NOT EXISTS "${OUTPUT_DIR}/render.png")
    message(FATAL_ERROR "render.png was not created")
endif()
if(NOT EXISTS "${OUTPUT_DIR}/reference_cpu.png")
    message(FATAL_ERROR "reference_cpu.png was not created")
endif()

# Two independent verifies validate in-process and cross-process determinism.
run_case(verify_first TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --backend cpu --verify)
require_text(verify_first "${verify_first_OUTPUT}" "verify=ok")
run_case(verify_second TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --backend cpu --verify)
require_text(verify_second "${verify_second_OUTPUT}" "verify=ok")

run_case(snapshot_list_initial TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --list-snapshots)
require_text(snapshot_list_initial "${snapshot_list_initial_OUTPUT}" "\"version\": 1")

run_case(create_two TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --create-snapshot)
require_text(create_two "${create_two_OUTPUT}" "snapshot=2")
run_case(update_light TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --update-light key intensity 40)
run_case(create_three TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --create-snapshot)
require_text(create_three "${create_three_OUTPUT}" "snapshot=3")

file(SHA256 "${DATA_DIR}/snapshots/snapshot_000002.json" snapshot_two_before)
file(SHA256 "${DATA_DIR}/snapshots/snapshot_000003.json" snapshot_three_before)

run_case(diff_two_three TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --diff 2 3)
require_text(diff_two_three "${diff_two_three_OUTPUT}" "\"id\": \"key\"")
require_text(diff_two_three "${diff_two_three_OUTPUT}" "\"intensity\"")
require_text(diff_two_three "${diff_two_three_OUTPUT}" "\"base\": 28.0")
require_text(diff_two_three "${diff_two_three_OUTPUT}" "\"target\": 40.0")

run_case(rollback_two TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --rollback 2)
require_text(rollback_two "${rollback_two_OUTPUT}" "rolled_back=2")
file(SHA256 "${DATA_DIR}/snapshots/snapshot_000002.json" snapshot_two_after)
file(SHA256 "${DATA_DIR}/snapshots/snapshot_000003.json" snapshot_three_after)
if(NOT snapshot_two_before STREQUAL snapshot_two_after)
    message(FATAL_ERROR "rollback modified snapshot 2")
endif()
if(NOT snapshot_three_before STREQUAL snapshot_three_after)
    message(FATAL_ERROR "rollback/diff modified snapshot 3")
endif()

run_case(merge_two_three TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --merge 2 3)
require_text(merge_two_three "${merge_two_three_OUTPUT}" "merged=2->3")
run_case(lights_after_merge TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --list-lights)
require_text(lights_after_merge "${lights_after_merge_OUTPUT}" "\"intensity\": 40.0")

# Conflict must be reported and must not alter current state.
run_case(rollback_two_again TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --rollback 2)
run_case(update_conflicting TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --update-light key intensity 35)
file(SHA256 "${DATA_DIR}/state/current_lights.json" state_before_conflict)
run_case(merge_conflict FALSE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --merge 2 3)
require_text(merge_conflict "${merge_conflict_ERROR}" "E_SNAPSHOT_CONFLICT")
file(SHA256 "${DATA_DIR}/state/current_lights.json" state_after_conflict)
if(NOT state_before_conflict STREQUAL state_after_conflict)
    message(FATAL_ERROR "merge conflict modified current scene state")
endif()

run_case(missing_snapshot FALSE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --rollback 999)
require_text(missing_snapshot "${missing_snapshot_ERROR}" "E_SNAPSHOT_NOT_FOUND")

# Add/remove diff coverage.
run_case(add_light TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --add-light extra 1.0,2.0,3.0 0.2,0.4,0.8 12.0 6.0)
run_case(create_four TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --create-snapshot)
require_text(create_four "${create_four_OUTPUT}" "snapshot=4")
run_case(diff_added TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --diff 3 4)
require_text(diff_added "${diff_added_OUTPUT}" "\"id\": \"extra\"")
run_case(remove_light TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --remove-light extra)
run_case(create_five TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --create-snapshot)
require_text(create_five "${create_five_OUTPUT}" "snapshot=5")
run_case(diff_removed TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --diff 4 5)
require_text(diff_removed "${diff_removed_OUTPUT}" "\"extra\"")

# Parser errors are reported but valid materials continue to load.
file(WRITE "${DATA_DIR}/materials/zz_bad_blend.json" [=[
{
  "name": "broken_blend",
  "layers": [{
    "name": "bad",
    "baseColor": "../textures/m0_l0_skin_base_basecolor.png",
    "metallic": "../textures/m0_l0_skin_base_metallic.png",
    "roughness": "../textures/m0_l0_skin_base_roughness.png",
    "normal": "../textures/m0_l0_skin_base_normal.png",
    "blendMode": "screen",
    "opacity": 1.0
  }]
}
]=])
file(WRITE "${DATA_DIR}/materials/zz_missing_texture.json" [=[
{
  "name": "broken_missing",
  "layers": [{
    "name": "bad",
    "baseColor": "../textures/absent.png",
    "metallic": "../textures/m0_l0_skin_base_metallic.png",
    "roughness": "../textures/m0_l0_skin_base_roughness.png",
    "normal": "../textures/m0_l0_skin_base_normal.png",
    "blendMode": "override",
    "opacity": 1.0
  }]
}
]=])
run_case(parse_errors_continue TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --list-materials)
require_text(parse_errors_continue "${parse_errors_continue_ERROR}" "E_MAT_PARSE")
require_text(parse_errors_continue "${parse_errors_continue_ERROR}" "illegal blendMode")
require_text(parse_errors_continue "${parse_errors_continue_ERROR}" "does not exist")
require_text(parse_errors_continue "${parse_errors_continue_OUTPUT}" "\"name\": \"skin\"")
file(REMOVE
    "${DATA_DIR}/materials/zz_bad_blend.json"
    "${DATA_DIR}/materials/zz_missing_texture.json")

# Texture decode failures are distinct from material parse failures.
set(CORRUPT_DATA "${TEST_ROOT}/corrupt_data")
file(MAKE_DIRECTORY "${CORRUPT_DATA}")
file(COPY "${DATA_DIR}/" DESTINATION "${CORRUPT_DATA}")
file(WRITE "${CORRUPT_DATA}/textures/m0_l0_skin_base_basecolor.png" "not a PNG")
run_case(texture_load_error FALSE
    "${TEST_EXE}" --data-dir "${CORRUPT_DATA}" --output-dir "${TEST_ROOT}/corrupt_out"
    --render)
require_text(texture_load_error "${texture_load_error_ERROR}" "E_TEX_LOAD")

# Permission mapping is enforceable on POSIX. Windows ACL testing is left to the
# packaged validation because chmod does not emulate ACL denial there.
if(UNIX)
    set(DENIED_DATA "${TEST_ROOT}/denied_data")
    file(MAKE_DIRECTORY "${DENIED_DATA}")
    file(COPY "${DATA_DIR}/" DESTINATION "${DENIED_DATA}")
    file(CHMOD "${DENIED_DATA}/snapshots"
        PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    run_case(io_denied FALSE
        "${TEST_EXE}" --data-dir "${DENIED_DATA}"
        --output-dir "${TEST_ROOT}/denied_out" --create-snapshot)
    require_text(io_denied "${io_denied_ERROR}" "E_IO_DENIED")
    file(CHMOD "${DENIED_DATA}/snapshots"
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)

    set(DENIED_TEXTURE_DATA "${TEST_ROOT}/denied_texture_data")
    file(MAKE_DIRECTORY "${DENIED_TEXTURE_DATA}")
    file(COPY "${DATA_DIR}/" DESTINATION "${DENIED_TEXTURE_DATA}")
    execute_process(
        COMMAND chmod 0200 "${DENIED_TEXTURE_DATA}/textures/m0_l0_skin_base_basecolor.png"
        RESULT_VARIABLE texture_chmod_result)
    if(NOT texture_chmod_result EQUAL 0)
        message(FATAL_ERROR "failed to make texture unreadable")
    endif()
    run_case(texture_io_denied FALSE
        "${TEST_EXE}" --data-dir "${DENIED_TEXTURE_DATA}"
        --output-dir "${TEST_ROOT}/denied_texture_out" --render)
    require_text(texture_io_denied "${texture_io_denied_ERROR}" "E_IO_DENIED")
    execute_process(
        COMMAND chmod 0644 "${DENIED_TEXTURE_DATA}/textures/m0_l0_skin_base_basecolor.png"
        RESULT_VARIABLE texture_chmod_restore_result)
    if(NOT texture_chmod_restore_result EQUAL 0)
        message(FATAL_ERROR "failed to restore texture permissions")
    endif()
endif()

# An orphan snapshot with no commit-log marker is ignored. The next version skips
# the reserved filename but the orphan is never reported as recovered/committed.
file(MAKE_DIRECTORY "${TEST_ROOT}/orphan_tmp")
file(COPY "${DATA_DIR}/snapshots/snapshot_000005.json"
    DESTINATION "${TEST_ROOT}/orphan_tmp")
file(RENAME "${TEST_ROOT}/orphan_tmp/snapshot_000005.json"
    "${DATA_DIR}/snapshots/snapshot_000099.json")
run_case(orphan_ignored TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --list-snapshots)
string(FIND "${orphan_ignored_OUTPUT}" "\"version\": 99" orphan_position)
if(NOT orphan_position EQUAL -1)
    message(FATAL_ERROR "uncommitted orphan snapshot was restored")
endif()
require_text(orphan_ignored "${orphan_ignored_OUTPUT}" "\"version\": 5")

# Losing the current-state file falls back to the highest commit marker.
file(REMOVE "${DATA_DIR}/state/current_lights.json")
run_case(recover_state TRUE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --list-lights)
require_text(recover_state "${recover_state_OUTPUT}" "\"id\": \"key\"")

# The saved reference was rendered with the default lights. Current lights differ,
# so verify must reject the reference comparison even though fresh renders match.
run_case(nondeterministic_reference FALSE
    "${TEST_EXE}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"
    --backend cpu --verify)
require_text(nondeterministic_reference
    "${nondeterministic_reference_ERROR}" "E_RENDER_NONDETERMINISTIC")

message(STATUS "integration tests passed")

