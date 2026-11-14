#!/usr/bin/env bash
set -euo pipefail

BAKER=${1:?baker executable}
SCENE_GEN=${2:?scene_gen executable}
WORK=${3:?work directory}
PYTHON=${PYTHON:-python3}

rm -rf "$WORK"
mkdir -p "$WORK"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

expect_exit() {
    local expected=$1
    shift
    set +e
    "$@" >"$WORK/last.stdout" 2>"$WORK/last.stderr"
    local actual=$?
    set -e
    [[ $actual -eq $expected ]] || fail "expected exit $expected, got $actual: $*"
}

"$SCENE_GEN" --out "$WORK/scene_a.bin" --width 160 --height 120 --layers 6 \
    --tex-size 64 --seed 22 >"$WORK/gen_a.log"
"$SCENE_GEN" --out "$WORK/scene_a_again.bin" --width 160 --height 120 --layers 6 \
    --tex-size 64 --seed 22 >"$WORK/gen_a2.log"
cmp "$WORK/scene_a.bin" "$WORK/scene_a_again.bin" ||
    fail "scene generator is not deterministic"

"$BAKER" --input "$WORK/scene_a.bin" --output "$WORK/out_a1.rgba" \
    --snapshot "$WORK/snap_a.bin" >"$WORK/run_a1.log"
grep -q 'snapshot=discard reason=missing' "$WORK/run_a1.log" ||
    fail "first run did not report a missing snapshot"
cp "$WORK/snap_a.bin" "$WORK/snapshot_a_complete.bin"

"$BAKER" --input "$WORK/scene_a.bin" --output "$WORK/out_a2.rgba" \
    --snapshot "$WORK/snap_a.bin" >"$WORK/run_a2.log"
grep -q 'snapshot=hit resume_from=6' "$WORK/run_a2.log" ||
    fail "second run did not hit the completed snapshot"
cmp "$WORK/out_a1.rgba" "$WORK/out_a2.rgba" ||
    fail "async runs produced different bytes"

grep -E '^\[baker\] (degenerate_layers|layer_pixels)=' "$WORK/run_a1.log" \
    >"$WORK/stats_a1.txt"
grep -E '^\[baker\] (degenerate_layers|layer_pixels)=' "$WORK/run_a2.log" \
    >"$WORK/stats_a2.txt"
cmp "$WORK/stats_a1.txt" "$WORK/stats_a2.txt" ||
    fail "statistics changed between runs"

"$BAKER" --input "$WORK/scene_a.bin" --output "$WORK/out_sync.rgba" \
    --snapshot "$WORK/snap_sync.bin" --sync-io >"$WORK/run_sync.log"
cmp "$WORK/out_a1.rgba" "$WORK/out_sync.rgba" ||
    fail "synchronous reference differs from async output"

BAKER_QUEUE_CAPACITY=1 BAKER_IO_DELAY_MS=2 \
    "$BAKER" --input "$WORK/scene_a.bin" --output "$WORK/out_backpressure.rgba" \
    --snapshot "$WORK/snap_backpressure.bin" >"$WORK/run_backpressure.log"
cmp "$WORK/out_a1.rgba" "$WORK/out_backpressure.rgba" ||
    fail "bounded-queue run differs from reference"

PYTHONPATH= "$PYTHON" - "$WORK/out_a1.rgba" <<'PY'
import struct
import sys

data = open(sys.argv[1], "rb").read()
magic, version, width, height = struct.unpack_from("<4sIII", data, 0)
assert magic == b"QRBA", magic
assert version == 1
assert (width, height) == (160, 120)
assert len(data) == 16 + width * height * 4
PY

"$SCENE_GEN" --out "$WORK/scene_b.bin" --width 160 --height 120 --layers 6 \
    --tex-size 64 --seed 2 >/dev/null
cp "$WORK/snapshot_a_complete.bin" "$WORK/snap_hash.bin"
"$BAKER" --input "$WORK/scene_b.bin" --output "$WORK/out_b.rgba" \
    --snapshot "$WORK/snap_hash.bin" >"$WORK/run_b.log"
grep -q 'snapshot=discard reason=hash_mismatch' "$WORK/run_b.log" ||
    fail "hash mismatch was not detected"

cp "$WORK/snapshot_a_complete.bin" "$WORK/snap_time.bin"
"$BAKER" --input "$WORK/scene_a.bin" --output "$WORK/out_time.rgba" \
    --snapshot "$WORK/snap_time.bin" --time 0.5 >"$WORK/run_time.log"
grep -q 'snapshot=discard reason=time_mismatch' "$WORK/run_time.log" ||
    fail "time mismatch was not detected"

"$SCENE_GEN" --out "$WORK/scene_c.bin" --width 80 --height 60 --layers 6 \
    --tex-size 32 --seed 3 >/dev/null
PYTHONPATH= "$PYTHON" - "$WORK/snapshot_a_complete.bin" "$WORK/scene_c.bin" \
    "$WORK/snap_resolution.bin" <<'PY'
import struct
import sys

snapshot = bytearray(open(sys.argv[1], "rb").read())
scene = open(sys.argv[2], "rb").read()
hash_value = 0xcbf29ce484222325
for byte in scene:
    hash_value = ((hash_value ^ byte) * 0x100000001b3) & 0xffffffffffffffff
struct.pack_into("<Q", snapshot, 8, hash_value)
open(sys.argv[3], "wb").write(snapshot)
PY
"$BAKER" --input "$WORK/scene_c.bin" --output "$WORK/out_res.rgba" \
    --snapshot "$WORK/snap_resolution.bin" >"$WORK/run_res.log"
grep -q 'snapshot=discard reason=resolution_mismatch' "$WORK/run_res.log" ||
    fail "resolution mismatch was not detected"

"$SCENE_GEN" --out "$WORK/scene_d.bin" --width 160 --height 120 --layers 5 \
    --tex-size 64 --seed 4 >/dev/null
PYTHONPATH= "$PYTHON" - "$WORK/snapshot_a_complete.bin" "$WORK/scene_d.bin" \
    "$WORK/snap_layers.bin" <<'PY'
import struct
import sys

snapshot = bytearray(open(sys.argv[1], "rb").read())
scene = open(sys.argv[2], "rb").read()
hash_value = 0xcbf29ce484222325
for byte in scene:
    hash_value = ((hash_value ^ byte) * 0x100000001b3) & 0xffffffffffffffff
struct.pack_into("<Q", snapshot, 8, hash_value)
open(sys.argv[3], "wb").write(snapshot)
PY
"$BAKER" --input "$WORK/scene_d.bin" --output "$WORK/out_layers.rgba" \
    --snapshot "$WORK/snap_layers.bin" >"$WORK/run_layers.log"
grep -q 'snapshot=discard reason=layer_mismatch' "$WORK/run_layers.log" ||
    fail "layer-count mismatch was not detected"

cp "$WORK/snapshot_a_complete.bin" "$WORK/snap_version.bin"
PYTHONPATH= "$PYTHON" - "$WORK/snap_version.bin" <<'PY'
import struct
import sys

path = sys.argv[1]
data = bytearray(open(path, "rb").read())
struct.pack_into("<I", data, 4, 99)
open(path, "wb").write(data)
PY
"$BAKER" --input "$WORK/scene_a.bin" --output "$WORK/out_version.rgba" \
    --snapshot "$WORK/snap_version.bin" >"$WORK/run_version.log"
grep -q 'snapshot=discard reason=version_mismatch' "$WORK/run_version.log" ||
    fail "snapshot version mismatch was not detected"

cp "$WORK/snapshot_a_complete.bin" "$WORK/snap_corrupt.bin"
truncate -s 64 "$WORK/snap_corrupt.bin"
"$BAKER" --input "$WORK/scene_a.bin" --output "$WORK/out_corrupt.rgba" \
    --snapshot "$WORK/snap_corrupt.bin" >"$WORK/run_corrupt.log"
grep -q 'snapshot=discard reason=corrupt' "$WORK/run_corrupt.log" ||
    fail "corrupt snapshot was not discarded"

printf 'sentinel-output-must-survive' >"$WORK/sentinel.rgba"
PYTHONPATH= "$PYTHON" - "$WORK/scene_a.bin" "$WORK" <<'PY'
import struct
import sys

scene = bytearray(open(sys.argv[1], "rb").read())
work = sys.argv[2]

bad_magic = bytearray(scene)
bad_magic[0] = ord("X")
open(work + "/bad_magic.bin", "wb").write(bad_magic)

bad_version = bytearray(scene)
struct.pack_into("<I", bad_version, 4, 99)
open(work + "/bad_version.bin", "wb").write(bad_version)

bad_value = bytearray(scene)
struct.pack_into("<I", bad_value, 8, 0)
open(work + "/bad_value.bin", "wb").write(bad_value)

open(work + "/truncated.bin", "wb").write(scene[:100])
PY

expect_exit 2 "$BAKER" --input "$WORK/bad_magic.bin" --output "$WORK/sentinel.rgba" \
    --snapshot "$WORK/error.snap"
grep -q '^ERR_MAGIC:' "$WORK/last.stderr" || fail "missing ERR_MAGIC"
expect_exit 3 "$BAKER" --input "$WORK/bad_version.bin" --output "$WORK/sentinel.rgba" \
    --snapshot "$WORK/error.snap"
grep -q '^ERR_VERSION:' "$WORK/last.stderr" || fail "missing ERR_VERSION"
expect_exit 5 "$BAKER" --input "$WORK/bad_value.bin" --output "$WORK/sentinel.rgba" \
    --snapshot "$WORK/error.snap"
grep -q '^ERR_VALUE:' "$WORK/last.stderr" || fail "missing ERR_VALUE"
expect_exit 4 "$BAKER" --input "$WORK/truncated.bin" --output "$WORK/sentinel.rgba" \
    --snapshot "$WORK/error.snap"
grep -q '^ERR_LENGTH:' "$WORK/last.stderr" || fail "missing ERR_LENGTH"

[[ "$(cat "$WORK/sentinel.rgba")" == "sentinel-output-must-survive" ]] ||
    fail "scene error modified the existing output"

echo "integration checks passed"
