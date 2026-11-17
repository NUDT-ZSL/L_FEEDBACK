#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build/linux}"
if [[ -z "${OSMESA_ROOT:-}" ]]; then
    if [[ -d /home/liang/deps/osmesa-prefix/usr ]]; then
        OSMESA_ROOT=/home/liang/deps/osmesa-prefix/usr
    else
        OSMESA_ROOT=/usr
    fi
fi
OUT_DIR="${OUT_DIR:-${ROOT}/artifacts/validation}"

failures=0

fail() {
    echo "FAIL: $*" >&2
    failures=$((failures + 1))
}

expect_exit() {
    local expected="$1"
    local name="$2"
    shift 2
    local stdout_file="${OUT_DIR}/${name}.stdout.txt"
    local stderr_file="${OUT_DIR}/${name}.stderr.txt"
    "$@" >"${stdout_file}" 2>"${stderr_file}"
    local actual=$?
    if [[ ${actual} -ne ${expected} ]]; then
        fail "${name}: expected exit ${expected}, got ${actual}"
        return
    fi
    if [[ ! -s ${stderr_file} ]]; then
        fail "${name}: expected a readable stderr diagnostic"
        return
    fi
    echo "PASS: ${name} exited ${actual}: $(head -n 1 "${stderr_file}")"
}

echo "Configuring and building..."
cmake -S "${ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DOSMESA_ROOT="${OSMESA_ROOT}" \
    -DSR_WARNINGS_AS_ERRORS=ON >/dev/null || exit 1
cmake --build "${BUILD_DIR}" -j2 >/dev/null || exit 1

mkdir -p "${OUT_DIR}"

echo "Running CPU tests..."
"${BUILD_DIR}/bin/sr_core_tests" || fail "CPU unit tests"

COMMON_ARGS=(
    --start-frame 0
    --frames 4
    --width 640
    --height 360
    --shadow-size 512
    --cascades 2
    --regenerate-frame 2
    --regenerate-variant 1
    --context osmesa
)

rm -rf "${OUT_DIR}/run_a" "${OUT_DIR}/run_b"
echo "Rendering deterministic run A..."
"${BUILD_DIR}/bin/skinned_csm_renderer" --output "${OUT_DIR}/run_a" \
    "${COMMON_ARGS[@]}" >"${OUT_DIR}/run_a.stdout.txt" || fail "run A"
echo "Rendering deterministic run B..."
"${BUILD_DIR}/bin/skinned_csm_renderer" --output "${OUT_DIR}/run_b" \
    "${COMMON_ARGS[@]}" >"${OUT_DIR}/run_b.stdout.txt" || fail "run B"

echo "Comparing PNG bytes..."
for color_file in "${OUT_DIR}"/run_a/*.png; do
    name="$(basename "${color_file}")"
    if ! cmp -s "${color_file}" "${OUT_DIR}/run_b/${name}"; then
        fail "PNG mismatch: ${name}"
    fi
done

if ! file "${OUT_DIR}/run_a/frame_000000_depth.png" | grep -q "16-bit grayscale"; then
    fail "depth PNG is not 16-bit grayscale"
fi

echo "Checking failure paths..."
expect_exit 2 "usage_error" \
    "${BUILD_DIR}/bin/skinned_csm_renderer" --unknown-option
expect_exit 3 "output_io_error" \
    "${BUILD_DIR}/bin/skinned_csm_renderer" --output /dev/null/not-a-dir
expect_exit 5 "shader_error" \
    "${BUILD_DIR}/bin/skinned_csm_renderer" --output "${OUT_DIR}/err_shader" \
    --test-error shader --context osmesa
expect_exit 6 "framebuffer_error" \
    "${BUILD_DIR}/bin/skinned_csm_renderer" --output "${OUT_DIR}/err_fbo" \
    --test-error framebuffer --context osmesa
expect_exit 7 "cascade_error" \
    "${BUILD_DIR}/bin/skinned_csm_renderer" --output "${OUT_DIR}/err_cascade" \
    --cascades 1 --context osmesa

sha256sum "${OUT_DIR}"/run_a/*.png >"${OUT_DIR}/run_a.sha256"
{
    echo "Skinned CSM renderer validation"
    echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Renderer: ${BUILD_DIR}/bin/skinned_csm_renderer"
    echo "Parameters: --start-frame 0 --frames 4 --width 640 --height 360"
    echo "            --shadow-size 512 --cascades 2 --regenerate-frame 2"
    echo "            --regenerate-variant 1 --context osmesa"
    echo
    cat "${OUT_DIR}/run_a.sha256"
    echo
    if [[ ${failures} -eq 0 ]]; then
        echo "RESULT: PASS"
    else
        echo "RESULT: FAIL (${failures} failures)"
    fi
} >"${OUT_DIR}/validation_summary.txt"

cat "${OUT_DIR}/validation_summary.txt"
exit $(( failures == 0 ? 0 : 1 ))
