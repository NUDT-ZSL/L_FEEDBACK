#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${1:-$ROOT/build/validation}"
OUT="$ROOT/artifacts/validation"
mkdir -p "$ROOT/build/tmp" "$OUT"
export TMPDIR="$ROOT/build/tmp"

echo "== configure/build =="
cmake -S "$ROOT" -B "$BUILD" \
  -DTERRAIN_BUILD_GL=OFF \
  -DTERRAIN_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j2

BIN="$BUILD/terrain_lod"
TEST_BIN="$BUILD/terrain_tests"

echo "== core tests =="
"$TEST_BIN" | tee "$OUT/core_tests.txt"

echo "== deterministic replay =="
"$BIN" \
  --camera-path "$ROOT/assets/camera_path.txt" \
  --time-step 0.1 \
  --frames 161 \
  --debug-view seam \
  --width 960 \
  --height 540 \
  > "$OUT/replay_a.jsonl"

"$BIN" \
  --camera-path "$ROOT/assets/camera_path.txt" \
  --time-step 0.1 \
  --frames 161 \
  --debug-view seam \
  --width 960 \
  --height 540 \
  > "$OUT/replay_b.jsonl"

cmp "$OUT/replay_a.jsonl" "$OUT/replay_b.jsonl"
cp "$OUT/replay_a.jsonl" "$OUT/replay.jsonl"

jq -s -e '
  length == 161 and
  (map(.frame) == [range(0; 161)]) and
  all(.[];
    has("triangles") and has("drawCalls") and has("stitchVertices") and
    .drawCalls >= 1 and .drawCalls <= 6 and
    .totalDrawCalls == (.drawCalls + .seamDrawCalls) and
    .maxStitchResidual < 0.001) and
  ((map(.seamSegments) | add) > 0) and
  ((map(.cacheHits) | add) > 0)
' "$OUT/replay.jsonl" > /dev/null

echo "== screenshots =="
for view in normal wireframe lod seam; do
  "$BIN" \
    --camera-path "$ROOT/assets/camera_path.txt" \
    --time-step 0.1 \
    --frames 1 \
    --debug-view "$view" \
    --width 960 \
    --height 540 \
    --screenshot "$OUT/$view.ppm" \
    > "$OUT/$view.jsonl"
  test "$(head -c 2 "$OUT/$view.ppm")" = "P6"
  python3 "$ROOT/scripts/ppm_to_png.py" "$OUT/$view.ppm"
done

"$BIN" \
  --camera-path "$ROOT/assets/camera_path.txt" \
  --time-step 0.1 \
  --frames 61 \
  --debug-view seam \
  --width 960 \
  --height 540 \
  --screenshot "$OUT/replay_6s_seam.ppm" \
  > "$OUT/replay_6s_seam.jsonl"
python3 "$ROOT/scripts/ppm_to_png.py" "$OUT/replay_6s_seam.ppm"

echo "== error codes =="
run_failure() {
  local label="$1"
  shift
  local stdout_file="$OUT/${label}.stdout"
  local stderr_file="$OUT/${label}.stderr"
  set +e
  "$@" > "$stdout_file" 2> "$stderr_file"
  local code=$?
  set -e
  printf '%s\n' "$code" > "$OUT/${label}.exitcode"
}

run_failure invalid_seed "$BIN" --seed -1
run_failure invalid_amplitude "$BIN" --amplitude 0
run_failure invalid_frequency "$BIN" --frequency 65
run_failure missing_path "$BIN" --replay --camera-path "$OUT/does_not_exist.cam"
printf '0 256 200\n' > "$OUT/bad_path.cam"
run_failure malformed_path "$BIN" --replay --camera-path "$OUT/bad_path.cam"

test "$(cat "$OUT/invalid_seed.exitcode")" = "2"
test "$(cat "$OUT/invalid_amplitude.exitcode")" = "2"
test "$(cat "$OUT/invalid_frequency.exitcode")" = "2"
test "$(cat "$OUT/missing_path.exitcode")" = "3"
test "$(cat "$OUT/malformed_path.exitcode")" = "3"
grep -q "E_INVALID_PARAM" "$OUT/invalid_seed.stderr"
grep -q "E_INVALID_PARAM" "$OUT/invalid_amplitude.stderr"
grep -q "E_INVALID_PARAM" "$OUT/invalid_frequency.stderr"
grep -q "E_BAD_PATH" "$OUT/missing_path.stderr"
grep -q "E_BAD_PATH" "$OUT/malformed_path.stderr"

echo "== stats summary =="
jq -s '{
  frames: length,
  triangles_min: (map(.triangles) | min),
  triangles_max: (map(.triangles) | max),
  draw_calls_min: (map(.drawCalls) | min),
  draw_calls_max: (map(.drawCalls) | max),
  stitch_vertices_total: (map(.stitchVertices) | add),
  seam_segments_total: (map(.seamSegments) | add),
  cache_hits_total: (map(.cacheHits) | add),
  cache_misses_total: (map(.cacheMisses) | add),
  max_stitch_residual: (map(.maxStitchResidual) | max)
}' "$OUT/replay.jsonl" > "$OUT/stats_summary.json"

{
  echo "Terrain LOD validation"
  echo "core_tests: $(cat "$OUT/core_tests.txt")"
  echo "replay_lines: $(wc -l < "$OUT/replay.jsonl")"
  echo "replay_repeat_identical: yes"
  echo "invalid_seed_exit: $(cat "$OUT/invalid_seed.exitcode")"
  echo "invalid_amplitude_exit: $(cat "$OUT/invalid_amplitude.exitcode")"
  echo "invalid_frequency_exit: $(cat "$OUT/invalid_frequency.exitcode")"
  echo "missing_path_exit: $(cat "$OUT/missing_path.exitcode")"
  echo "malformed_path_exit: $(cat "$OUT/malformed_path.exitcode")"
  echo "screenshots: normal, wireframe, lod, seam, replay_6s_seam"
  echo "stats:"
  cat "$OUT/stats_summary.json"
  echo "sha256:"
  (
    cd "$OUT"
    sha256sum replay.jsonl *.ppm *.png
  )
} | tee "$OUT/summary.txt"

