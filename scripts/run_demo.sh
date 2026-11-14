#!/usr/bin/env bash
# Linux/macOS end-to-end demo. Pass an optional FFmpeg dev prefix:
#   ./scripts/run_demo.sh /path/to/ffmpeg-prefix
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build"
FFMPEG_ROOT_ARG="${1:-}"

CONFIGURE_ARGS=(-S "${ROOT}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release)
if [[ -n "${FFMPEG_ROOT_ARG}" ]]; then
  CONFIGURE_ARGS+=(-DFFmpeg_ROOT="${FFMPEG_ROOT_ARG}")
fi

cmake "${CONFIGURE_ARGS[@]}"
cmake --build "${BUILD}" --parallel

python3 "${ROOT}/tools/generate_samples.py" --output "${BUILD}/samples"

"${BUILD}/lipsync" \
  --input "${BUILD}/samples" \
  --output "${BUILD}/output" \
  --config "${ROOT}/config/default.json"

ctest --test-dir "${BUILD}" --output-on-failure
