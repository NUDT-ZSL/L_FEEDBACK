#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DSKINRENDER_USE_GLFW=OFF \
    -DSKINRENDER_BUILD_TESTS=ON
cmake --build build --parallel
if command -v ctest >/dev/null 2>&1; then
    ctest --test-dir build --output-on-failure
else
    python3 scripts/validate.py --exe build/skin_renderer
fi

