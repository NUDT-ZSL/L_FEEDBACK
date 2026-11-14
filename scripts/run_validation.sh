#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
CAPTURE_DIR="${ROOT_DIR}/artifacts/auto_test"
OVERLAY_DIR="${CAPTURE_DIR}/overlay"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j

EXE="${BUILD_DIR}/bin/pcss_shadow_renderer"
if [[ ! -x "${EXE}" ]]; then
    echo "Executable not found: ${EXE}" >&2
    exit 1
fi

mkdir -p "${CAPTURE_DIR}" "${OVERLAY_DIR}"
EXTRA_ARGS=()
if [[ "${ALLOW_GL45:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--allow-gl45)
fi

"${EXE}" --auto-test --width 960 --height 540 --frames 84 --no-vsync \
    --hide-overlay "${EXTRA_ARGS[@]}" --capture-dir "${CAPTURE_DIR}" \
    | tee "${CAPTURE_DIR}/stdout.log"

"${EXE}" --auto-test --width 960 --height 540 --frames 20 --no-vsync \
    "${EXTRA_ARGS[@]}" --capture-dir "${OVERLAY_DIR}" \
    | tee "${OVERLAY_DIR}/stdout.log"

grep -q "\[RESULT\]" "${CAPTURE_DIR}/stdout.log"
grep -q "pcss_light_size_delta_percent" "${CAPTURE_DIR}/stdout.log"
grep -q "SHADOW\] allocated resolution=4096" "${CAPTURE_DIR}/stdout.log"

PNG_COUNT="$(find "${CAPTURE_DIR}" -maxdepth 1 -name '*.png' | wc -l)"
if [[ "${PNG_COUNT}" -lt 10 ]]; then
    echo "Expected at least 10 auto-test PNGs, found ${PNG_COUNT}" >&2
    exit 1
fi

echo "Validation complete: ${PNG_COUNT} scene captures + overlay capture in ${OVERLAY_DIR}"

