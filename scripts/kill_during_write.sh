#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD=${1:-"$ROOT/build"}
WORK=${2:-"$ROOT/work/kill-during-write"}
BAKER="$BUILD/baker"
SCENE_GEN="$BUILD/scene_gen"

rm -rf "$WORK"
mkdir -p "$WORK"

"$SCENE_GEN" --out "$WORK/scene.bin" --seed 22 >/dev/null
"$BAKER" --input "$WORK/scene.bin" --output "$WORK/baseline.rgba" \
    --snapshot "$WORK/baseline.snap" --sync-io >/dev/null

cp "$WORK/baseline.rgba" "$WORK/out.rgba"
cp "$WORK/baseline.snap" "$WORK/snapshot.bin"

BAKER_IO_DELAY_MS=500 "$BAKER" --input "$WORK/scene.bin" \
    --output "$WORK/out.rgba" --snapshot "$WORK/snapshot.bin" \
    >"$WORK/killed.log" 2>&1 &
pid=$!

for ((i = 0; i < 300; ++i)); do
    if compgen -G "$WORK/out.rgba.tmp.*" >/dev/null; then
        break
    fi
    sleep 0.02
done
if ! compgen -G "$WORK/out.rgba.tmp.*" >/dev/null; then
    echo "temporary output was not observed" >&2
    kill "$pid" 2>/dev/null || true
    exit 1
fi

kill -9 "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true
cmp "$WORK/out.rgba" "$WORK/baseline.rgba"
echo "after SIGKILL: previous output is intact"

"$BAKER" --input "$WORK/scene.bin" --output "$WORK/out.rgba" \
    --snapshot "$WORK/snapshot.bin" | tee "$WORK/resume.log"
cmp "$WORK/out.rgba" "$WORK/baseline.rgba"
grep -q 'snapshot=hit resume_from=' "$WORK/resume.log"
echo "rerun: output matches the sequential baseline"
echo "artifacts: $WORK"
