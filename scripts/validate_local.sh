#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build/headless"
REPORTS="$ROOT/build/headless/reports"

cmake -S "$ROOT" -B "$BUILD" \
  -DLCR_BUILD_APP=OFF \
  -DLCR_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j

if command -v ctest >/dev/null 2>&1; then
  ctest --test-dir "$BUILD" --output-on-failure
else
  "$BUILD/character_tests"
  for pair in "texture:20:E_TEXTURE_GENERATION" "shader:30:E_SHADER_COMPILE" "bone:40:E_BONE_INDEX"; do
    IFS=: read -r fault expected code <<<"$pair"
    set +e
    stderr="$("$BUILD/character_headless" --headless --fault "$fault" 2>&1 >/dev/null)"
    actual=$?
    set -e
    if [[ "$actual" != "$expected" || "$stderr" != *"$code"* ]]; then
      echo "fault test failed: $fault rc=$actual stderr=$stderr" >&2
      exit 1
    fi
    echo "fault $fault -> rc=$actual $code"
  done
fi

"$BUILD/character_headless" --headless --seed 1337 --logic-frames 360 \
  --render-fps 120 --max-render-frames 720 --switch-time 1.25 \
  --equipment iron \
  --report "$REPORTS/report-a.json" \
  --snapshot "$REPORTS/snapshot-a.json"

"$BUILD/character_headless" --headless --seed 1337 --logic-frames 360 \
  --render-fps 45 --max-render-frames 270 --switch-time 1.25 \
  --equipment iron \
  --report "$REPORTS/report-b.json" \
  --snapshot "$REPORTS/snapshot-b.json"

cmp "$REPORTS/snapshot-a.json" "$REPORTS/snapshot-b.json"
echo "Deterministic snapshots are byte-identical."
echo "Reports: $REPORTS"

