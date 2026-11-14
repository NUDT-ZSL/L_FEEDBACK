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

wait_for_pattern() {
    local pattern=$1
    local limit=${2:-300}
    for ((i = 0; i < limit; ++i)); do
        if compgen -G "$pattern" >/dev/null; then
            return 0
        fi
        sleep 0.02
    done
    return 1
}

"$SCENE_GEN" --out "$WORK/scene.bin" --width 160 --height 120 --layers 6 \
    --tex-size 64 --seed 77 >/dev/null
"$BAKER" --input "$WORK/scene.bin" --output "$WORK/baseline.rgba" \
    --snapshot "$WORK/baseline.snap" --sync-io >/dev/null

mkdir -p "$WORK/kill_final" "$WORK/partial" "$WORK/sigterm"

cp "$WORK/baseline.rgba" "$WORK/kill_final/out.rgba"
cp "$WORK/baseline.snap" "$WORK/kill_final/snap.bin"
BAKER_IO_DELAY_MS=250 "$BAKER" --input "$WORK/scene.bin" \
    --output "$WORK/kill_final/out.rgba" --snapshot "$WORK/kill_final/snap.bin" \
    >"$WORK/kill_final/run.log" 2>&1 &
pid=$!
wait_for_pattern "$WORK/kill_final/out.rgba.tmp.*" ||
    fail "did not observe the final output temporary file"
kill -9 "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true
cmp "$WORK/kill_final/out.rgba" "$WORK/baseline.rgba" ||
    fail "SIGKILL during final write damaged the old output"

"$BAKER" --input "$WORK/scene.bin" --output "$WORK/kill_final/out.rgba" \
    --snapshot "$WORK/kill_final/snap.bin" >"$WORK/kill_final/resume.log"
cmp "$WORK/kill_final/out.rgba" "$WORK/baseline.rgba" ||
    fail "post-SIGKILL rerun produced different bytes"
grep -q 'snapshot=hit resume_from=' "$WORK/kill_final/resume.log" ||
    fail "post-SIGKILL rerun did not use the persisted snapshot"

cp "$WORK/baseline.rgba" "$WORK/partial/out.rgba"
BAKER_IO_DELAY_MS=300 "$BAKER" --input "$WORK/scene.bin" \
    --output "$WORK/partial/out.rgba" --snapshot "$WORK/partial/snap.bin" \
    >"$WORK/partial/run.log" 2>&1 &
pid=$!
completed=-1
for ((i = 0; i < 800; ++i)); do
    if [[ -f "$WORK/partial/snap.bin" ]]; then
        completed=$(
            PYTHONPATH= "$PYTHON" - "$WORK/partial/snap.bin" <<'PY'
import struct
import sys

data = open(sys.argv[1], "rb").read()
if len(data) >= 32 and data[:4] == b"QSNP":
    print(struct.unpack_from("<I", data, 28)[0])
else:
    print(-1)
PY
        )
        if [[ $completed -ge 1 && $completed -lt 6 ]]; then
            break
        fi
    fi
    sleep 0.02
done
[[ $completed -ge 1 && $completed -lt 6 ]] ||
    fail "did not observe a partial persisted snapshot"
kill -9 "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true
cmp "$WORK/partial/out.rgba" "$WORK/baseline.rgba" ||
    fail "SIGKILL during snapshot write damaged the old output"

completed=$(
    PYTHONPATH= "$PYTHON" - "$WORK/partial/snap.bin" <<'PY'
import struct
import sys

data = open(sys.argv[1], "rb").read()
print(struct.unpack_from("<I", data, 28)[0])
PY
)
"$BAKER" --input "$WORK/scene.bin" --output "$WORK/partial/out.rgba" \
    --snapshot "$WORK/partial/snap.bin" >"$WORK/partial/resume.log"
grep -q "snapshot=hit resume_from=$completed" "$WORK/partial/resume.log" ||
    fail "partial snapshot did not resume at layer $completed"
cmp "$WORK/partial/out.rgba" "$WORK/baseline.rgba" ||
    fail "partial-resume output differs from the sequential baseline"

cp "$WORK/baseline.rgba" "$WORK/sigterm/out.rgba"
cp "$WORK/baseline.snap" "$WORK/sigterm/snap.bin"
BAKER_IO_DELAY_MS=500 "$BAKER" --input "$WORK/scene.bin" \
    --output "$WORK/sigterm/out.rgba" --snapshot "$WORK/sigterm/snap.bin" \
    >"$WORK/sigterm/run.log" 2>&1 &
pid=$!
wait_for_pattern "$WORK/sigterm/out.rgba.tmp.*" ||
    fail "did not observe temporary output before SIGTERM"
kill -TERM "$pid"
set +e
wait "$pid"
status=$?
set -e
[[ $status -ne 0 ]] || fail "SIGTERM path unexpectedly returned success"
cmp "$WORK/sigterm/out.rgba" "$WORK/baseline.rgba" ||
    fail "SIGTERM path damaged the old output"
if compgen -G "$WORK/sigterm/out.rgba.tmp.*" >/dev/null; then
    fail "SIGTERM path left an output temporary file"
fi

echo "kill/recovery checks passed"
