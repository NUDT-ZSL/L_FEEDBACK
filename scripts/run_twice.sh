#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD=${1:-"$ROOT/build"}
WORK=${2:-"$ROOT/work/two-runs"}
BAKER="$BUILD/baker"
SCENE_GEN="$BUILD/scene_gen"

rm -rf "$WORK"
mkdir -p "$WORK"

"$SCENE_GEN" --out "$WORK/scene.bin" --seed 22
"$SCENE_GEN" --out "$WORK/scene_again.bin" --seed 22
cmp "$WORK/scene.bin" "$WORK/scene_again.bin"
echo "scene bytes: identical"

"$BAKER" --input "$WORK/scene.bin" --output "$WORK/out1.rgba" \
    --snapshot "$WORK/snapshot.bin" | tee "$WORK/run1.log"
"$BAKER" --input "$WORK/scene.bin" --output "$WORK/out2.rgba" \
    --snapshot "$WORK/snapshot.bin" | tee "$WORK/run2.log"

cmp "$WORK/out1.rgba" "$WORK/out2.rgba"
grep -E '^\[baker\] (degenerate_layers|layer_pixels)=' "$WORK/run1.log" \
    >"$WORK/stats1.txt"
grep -E '^\[baker\] (degenerate_layers|layer_pixels)=' "$WORK/run2.log" \
    >"$WORK/stats2.txt"
cmp "$WORK/stats1.txt" "$WORK/stats2.txt"

grep -q 'snapshot=hit resume_from=' "$WORK/run2.log"
echo "two-run output and statistics: identical"
echo "artifacts: $WORK"
