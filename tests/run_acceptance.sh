#!/usr/bin/env bash
set -euo pipefail

EXE="${1:?usage: run_acceptance.sh <voice-replay executable>}"
EXE="$(realpath "$EXE")"
ROOT="$(mktemp -d "${TMPDIR:-/tmp}/voice-replay-acceptance.XXXXXX")"
trap 'rm -rf "$ROOT"' EXIT

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

pass() {
    echo "PASS: $*"
}

run_ok() {
    local label="$1"
    shift
    local log="$ROOT/${label// /_}.log"
    if ! "$@" >"$log.stdout" 2>"$log.stderr"; then
        cat "$log.stderr" >&2 || true
        fail "$label returned a non-zero exit code"
    fi
    LAST_STDERR="$log.stderr"
}

expect_error() {
    local expected_exit="$1"
    local expected_code="$2"
    local label="$3"
    shift 3
    local log="$ROOT/error_${label// /_}.log"
    set +e
    "$@" >"$log.stdout" 2>"$log.stderr"
    local actual=$?
    set -e
    [[ "$actual" -eq "$expected_exit" ]] ||
        fail "$label exit $actual, expected $expected_exit"
    grep -q "error=$expected_code" "$log.stderr" ||
        fail "$label did not emit $expected_code"
    LAST_STDERR="$log.stderr"
}

copy_input() {
    local target="$1"
    mkdir -p "$target"
    cp -R "$ROOT/in/." "$target/"
}

mkdir -p "$ROOT/in"
"$EXE" generate --output "$ROOT/in" >"$ROOT/generate.stdout" 2>"$ROOT/generate.stderr"

run_ok "full export" "$EXE" export --input "$ROOT/in" --output "$ROOT/full"
grep -q 'REUSE' "$LAST_STDERR" && fail "fresh export unexpectedly reused output"
[[ "$(grep -c 'voice-replay: SHA256 ' "$LAST_STDERR")" -eq 5 ]] ||
    fail "success log does not contain five WAV digest lines"
[[ "$(stat -c %s "$ROOT/full/master.wav")" -eq 1555244 ]] ||
    fail "unexpected stereo master WAV size"
[[ "$(stat -c %s "$ROOT/full/tracks/track_0001_caster_alpha.wav")" -eq 777644 ]] ||
    fail "unexpected mono stem WAV size"
grep -q '"total_frames": 388800' "$ROOT/full/manifest.json" ||
    fail "manifest does not record the match-length frame count"
grep -q '"alignment_offset_frames": 48000' "$ROOT/full/manifest.json" ||
    fail "expected caster_alpha event-anchor offset is missing"
grep -q '"used_frames": 0' "$ROOT/full/manifest.json" ||
    fail "zero-available track is not marked as having zero used frames"
grep -q '"clipped_samples": 31450' "$ROOT/full/manifest.json" ||
    fail "expected per-track clipping count is missing"
master_sha="$(sha256sum "$ROOT/full/master.wav" | cut -d' ' -f1)"
grep -q "$master_sha" "$ROOT/full/manifest.json" ||
    fail "master SHA-256 in manifest does not match the file"
pass "full export, partial final block, WAV shape, alignment, clipping and manifest digest"

run_ok "second full export" "$EXE" export --input "$ROOT/in" --output "$ROOT/full2"
while IFS= read -r rel; do
    cmp -s "$ROOT/full/$rel" "$ROOT/full2/$rel" ||
        fail "second full export differs at $rel"
done < <(cd "$ROOT/full" && find . -type f -printf '%P\n' | sort)
pass "same input produces byte-identical outputs"

set +e
"$EXE" export --input "$ROOT/in" --output "$ROOT/resumed" \
    --abort-after-blocks 17 >"$ROOT/abort.stdout" 2>"$ROOT/abort.stderr"
abort_code=$?
set -e
[[ "$abort_code" -eq 75 ]] || fail "test abort exit $abort_code, expected 75"
[[ -f "$ROOT/resumed/.voice-replay-state.json" ]] ||
    fail "abort did not leave a resume state"
[[ ! -f "$ROOT/resumed/manifest.json" ]] ||
    fail "abort left a completed manifest"
grep -q '"completed_blocks": 17' "$ROOT/resumed/.voice-replay-state.json" ||
    fail "state does not record the block boundary"
run_ok "resume export" "$EXE" export --input "$ROOT/in" --output "$ROOT/resumed" --resume
while IFS= read -r rel; do
    cmp -s "$ROOT/full/$rel" "$ROOT/resumed/$rel" ||
        fail "resumed output differs from uninterrupted output at $rel"
done < <(cd "$ROOT/full" && find . -type f -printf '%P\n' | sort)
[[ ! -f "$ROOT/resumed/.voice-replay-state.json" ]] ||
    fail "successful resume did not remove the temporary state"
pass "block-boundary abort and resume are byte-identical"

master_mtime="$(stat -c '%Y:%s' "$ROOT/full/master.wav")"
manifest_mtime="$(stat -c '%Y:%s' "$ROOT/full/manifest.json")"
sleep 1
run_ok "reuse export" "$EXE" export --input "$ROOT/in" --output "$ROOT/full"
grep -q 'REUSE existing byte-identical outputs' "$LAST_STDERR" ||
    fail "repeat export did not reuse byte-identical outputs"
[[ "$(stat -c '%Y:%s' "$ROOT/full/master.wav")" == "$master_mtime" ]] ||
    fail "repeat export rewrote master.wav"
[[ "$(stat -c '%Y:%s' "$ROOT/full/manifest.json")" == "$manifest_mtime" ]] ||
    fail "repeat export rewrote manifest.json"
pass "repeat export reuses files without timestamp changes"

copy_input "$ROOT/negative/in"
sed -i 's/"start_frame": 0/"start_frame": -1/' "$ROOT/negative/in/caster_alpha.json"
expect_error 4 E_META_INVALID "negative frame" \
    "$EXE" export --input "$ROOT/negative/in" --output "$ROOT/negative/out"
[[ ! -e "$ROOT/negative/out/master.wav" ]] ||
    fail "invalid metadata left a partial master"

copy_input "$ROOT/nonfinite/in"
sed -i 's/"gain_db": -6.0/"gain_db": Infinity/' "$ROOT/nonfinite/in/caster_alpha.json"
expect_error 4 E_META_INVALID "non-finite gain" \
    "$EXE" export --input "$ROOT/nonfinite/in" --output "$ROOT/nonfinite/out"

copy_input "$ROOT/nan/in"
sed -i 's/"gain_db": -6.0/"gain_db": NaN/' "$ROOT/nan/in/caster_alpha.json"
expect_error 4 E_META_INVALID "NaN gain" \
    "$EXE" export --input "$ROOT/nan/in" --output "$ROOT/nan/out"

copy_input "$ROOT/rate/in"
sed -i 's/"sample_rate": 48000/"sample_rate": 44100/' "$ROOT/rate/in/events.json"
expect_error 5 E_RATE_MISMATCH "sample-rate mismatch" \
    "$EXE" export --input "$ROOT/rate/in" --output "$ROOT/rate/out"

copy_input "$ROOT/event/in"
sed -i 's/"match_time_ms": 1000, "local_frame": 0/"match_time_ms": 8999, "local_frame": 0/' \
    "$ROOT/event/in/caster_alpha.json"
expect_error 4 E_META_INVALID "event outside match" \
    "$EXE" export --input "$ROOT/event/in" --output "$ROOT/event/out"

copy_input "$ROOT/duplicate/in"
sed -i 's/"commentary_id": "caster_alpha"/"commentary_id": "caster_beta"/' \
    "$ROOT/duplicate/in/caster_alpha.json"
expect_error 4 E_META_INVALID "duplicate commentary id" \
    "$EXE" export --input "$ROOT/duplicate/in" --output "$ROOT/duplicate/out"

expect_error 2 E_USAGE "missing arguments" "$EXE"
expect_error 7 E_STATE_MISSING "missing state" \
    "$EXE" export --input "$ROOT/in" --output "$ROOT/empty" --resume

mkdir -p "$ROOT/unwritable/in"
cp -R "$ROOT/in/." "$ROOT/unwritable/in/"
touch "$ROOT/unwritable/not_a_directory"
expect_error 6 E_OUTPUT_UNWRITABLE "unwritable output" \
    "$EXE" export --input "$ROOT/unwritable/in" \
    --output "$ROOT/unwritable/not_a_directory"
pass "business error codes and no partial output on validation failure"

copy_input "$ROOT/statechange/in"
set +e
"$EXE" export --input "$ROOT/statechange/in" --output "$ROOT/statechange/out" \
    --abort-after-blocks 11 >"$ROOT/statechange/abort.stdout" 2>"$ROOT/statechange/abort.stderr"
[[ $? -eq 75 ]] || fail "state-change setup abort failed"
set -e
sed -i 's/"gain_db": -6.0/"gain_db": -7.0/' "$ROOT/statechange/in/caster_alpha.json"
expect_error 8 E_STATE_CORRUPT "changed input with resume" \
    "$EXE" export --input "$ROOT/statechange/in" \
    --output "$ROOT/statechange/out" --resume
[[ -f "$ROOT/statechange/out/.voice-replay-state.json" ]] ||
    fail "state was removed after a corrupt-state error"
[[ ! -f "$ROOT/statechange/out/manifest.json" ]] ||
    fail "state mismatch silently completed the export"

copy_input "$ROOT/statehash/in"
set +e
"$EXE" export --input "$ROOT/statehash/in" --output "$ROOT/statehash/out" \
    --abort-after-blocks 11 >"$ROOT/statehash/abort.stdout" 2>"$ROOT/statehash/abort.stderr"
[[ $? -eq 75 ]] || fail "state-hash setup abort failed"
set -e
sed -i -E '0,/"prefix_sha256": "[0-9a-f]{64}"/s//"prefix_sha256": "0000000000000000000000000000000000000000000000000000000000000000"/' \
    "$ROOT/statehash/out/.voice-replay-state.json"
expect_error 8 E_STATE_CORRUPT "corrupt output prefix hash" \
    "$EXE" export --input "$ROOT/statehash/in" \
    --output "$ROOT/statehash/out" --resume
pass "input/state mismatch is rejected with E_STATE_CORRUPT"

echo "ALL ACCEPTANCE TESTS PASSED"

