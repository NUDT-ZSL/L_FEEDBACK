#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${root_dir}/build}"
output_dir="${2:-${root_dir}/artifacts/verify-linux}"
app="${build_dir}/skeletal_material_demo"
cpu_tests="${build_dir}/cpu_tests"

if [[ ! -x "${app}" || ! -x "${cpu_tests}" ]]; then
    echo "Build executables not found in ${build_dir}." >&2
    echo "Configure and build first; see README.md." >&2
    exit 2
fi
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo "A desktop display is required for OpenGL integration tests." >&2
    exit 2
fi

mkdir -p "${output_dir}"

echo "[1/8] CPU scene/UBO contract"
"${cpu_tests}"

echo "[2/8] GPU self-test and screenshot"
"${app}" --self-test --screenshot "${output_dir}/self-test.ppm" \
    >"${output_dir}/self-test.stdout" 2>"${output_dir}/self-test.stderr"
grep -q "\[SELFTEST\] PASS" "${output_dir}/self-test.stdout"
grep -q "Visible character pixels" "${output_dir}/self-test.stdout"
grep -q "Shader mode: CustomLighting" "${output_dir}/self-test.stdout"

echo "[3/8] Shader compile failure fallback"
"${app}" --self-test --inject-shader-compile-failure \
    --screenshot "${output_dir}/compile-fallback.ppm" \
    >"${output_dir}/compile-fallback.stdout" \
    2>"${output_dir}/compile-fallback.stderr"
grep -q "E2001/SHADER_COMPILE_FALLBACK" \
    "${output_dir}/compile-fallback.stderr"
grep -q "Shader mode: SafeFallback" "${output_dir}/compile-fallback.stdout"
grep -q "\[SELFTEST\] PASS" "${output_dir}/compile-fallback.stdout"

echo "[4/8] Shader link failure fallback"
"${app}" --self-test --inject-shader-link-failure \
    >"${output_dir}/link-fallback.stdout" \
    2>"${output_dir}/link-fallback.stderr"
grep -q "E2002/SHADER_LINK_FALLBACK" "${output_dir}/link-fallback.stderr"
grep -q "Shader mode: SafeFallback" "${output_dir}/link-fallback.stdout"
grep -q "\[SELFTEST\] PASS" "${output_dir}/link-fallback.stdout"

check_expected_failure() {
    local name="$1"
    local expected_log="$2"
    shift 2

    set +e
    "${app}" "$@" >"${output_dir}/${name}.stdout" \
        2>"${output_dir}/${name}.stderr"
    local exit_code=$?
    set -e

    if [[ ${exit_code} -eq 0 ]]; then
        echo "Expected non-zero exit for ${name}" >&2
        exit 1
    fi
    grep -q "${expected_log}" "${output_dir}/${name}.stderr"
    echo "  ${name}: non-zero exit (${exit_code}), ${expected_log} recorded"
}

echo "[5/8] Context creation failure path"
check_expected_failure "context-failure" "E1003/OPENGL_45_UNAVAILABLE" \
    --simulate-context-failure

echo "[6/8] Procedural resource failure path"
check_expected_failure "resource-failure" "E3001/SCENE_RESOURCE_INVALID" \
    --simulate-resource-failure

echo "[7/8] Three-consecutive-frame GL error path"
check_expected_failure "gl-error" "E1010/PERSISTENT_GL_ERROR" \
    --self-test --inject-gl-error

echo "[8/8] VSync-off 200 FPS cap"
"${app}" --benchmark-frames 120 --vsync-off \
    >"${output_dir}/fps-cap.stdout" 2>"${output_dir}/fps-cap.stderr"
grep -q "vsync=OFF" "${output_dir}/fps-cap.stdout"
measured_fps="$(
    sed -n 's/.*fps=\([0-9.]*\).*/\1/p' "${output_dir}/fps-cap.stdout"
)"
awk -v fps="${measured_fps}" 'BEGIN { exit !(fps > 0.0 && fps <= 205.0) }'

echo "All Linux verification checks passed. Outputs: ${output_dir}"
