#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${1:-$ROOT/build/rigskin}"
UNIT="${UNIT_TESTS:-$ROOT/build/rigskin_unit_tests}"
WORK="$(mktemp -d "$ROOT/build/integration.XXXXXXXX")"

if [[ ! -x "$BIN" ]]; then
    echo "FAIL: rigskin executable not found: $BIN" >&2
    exit 1
fi
if [[ ! -x "$UNIT" ]]; then
    echo "FAIL: unit test executable not found: $UNIT" >&2
    exit 1
fi

pass() {
    echo "PASS: $1"
}

fail() {
    echo "FAIL: $1" >&2
    echo "Integration work directory retained at $WORK" >&2
    exit 1
}

run() {
    local directory="$1"
    shift
    "$BIN" --snapshot-dir "$directory" "$@"
}

assert_summary() {
    local directory="$1"
    local expected_status="$2"
    local expected_reused="$3"
    local expected_recomputed="$4"
    local expected_codes="$5"
    grep -q "status=$expected_status" "$directory/summary.txt" ||
        fail "summary status in $directory"
    grep -q "frames_reused=$expected_reused" "$directory/summary.txt" ||
        fail "summary reused count in $directory"
    grep -q "frames_recomputed=$expected_recomputed" "$directory/summary.txt" ||
        fail "summary recomputed count in $directory"
    grep -q "error_codes=$expected_codes" "$directory/summary.txt" ||
        fail "summary error codes in $directory"
}

echo "work=$WORK"
echo "--- unit tests ---"
"$UNIT" || fail "unit tests"
pass "unit tests"

echo "--- serial versus parallel ---"
run "$WORK/serial" --threads 1 --frames 16 --width 640 --height 480 >/dev/null
run "$WORK/parallel" --threads 8 --frames 16 --width 640 --height 480 >/dev/null
diff -rq "$WORK/serial" "$WORK/parallel" >/dev/null ||
    fail "serial and parallel trees differ"
pass "serial/parallel byte-identical output trees"

echo "--- threads greater than frames ---"
run "$WORK/three-serial" --threads 1 --frames 3 --width 320 --height 240 >/dev/null
run "$WORK/three-wide" --threads 64 --frames 3 --width 320 --height 240 >/dev/null
diff -rq "$WORK/three-serial" "$WORK/three-wide" >/dev/null ||
    fail "oversubscribed thread output differs"
grep -q "frames_success=3" "$WORK/three-wide/summary.txt" ||
    fail "oversubscribed thread frame count"
pass "threads greater than frames produces no empty or duplicate frames"

echo "--- clean resume ---"
run "$WORK/parallel" --threads 3 --frames 16 --width 640 --height 480 >/dev/null
assert_summary "$WORK/parallel" complete 16 0 none
pass "clean resume reuses all frames"

echo "--- corruption recovery ---"
cp -a "$WORK/serial" "$WORK/corrupt"
printf '\000' >> "$WORK/corrupt/frames/frame_0005.mesh"
run "$WORK/corrupt" --threads 4 --frames 16 --width 640 --height 480 >/dev/null
assert_summary "$WORK/corrupt" complete 5 11 E2002
grep -q "E2002 frame=5" "$WORK/corrupt/errors/frame_recovery.txt" ||
    fail "corruption report"
diff -rq "$WORK/serial/frames" "$WORK/corrupt/frames" >/dev/null ||
    fail "corruption recovery frame outputs"
cmp -s "$WORK/serial/manifest.txt" "$WORK/corrupt/manifest.txt" ||
    fail "corruption recovery manifest"
pass "corrupt frame and subsequent frames recomputed"

echo "--- unsupported manifest version ---"
cp -a "$WORK/serial" "$WORK/badver"
sed -e 's/FMSNAP_MANIFEST 2/FMSNAP_MANIFEST 999/' \
    -e 's/manifest_version 2/manifest_version 999/' \
    "$WORK/badver/manifest.txt" > "$WORK/badver/manifest.txt.new"
mv "$WORK/badver/manifest.txt.new" "$WORK/badver/manifest.txt"
run "$WORK/badver" --threads 5 --frames 16 --width 640 --height 480 >/dev/null
assert_summary "$WORK/badver" complete 0 16 E1003
grep -q "detected_version=999" "$WORK/badver/errors/invalid_manifest_version.txt" ||
    fail "invalid version report"
[[ -d "$WORK/badver/quarantine/invalid_manifest_version/frames" ]] ||
    fail "old snapshot was not quarantined"
diff -rq "$WORK/serial/frames" "$WORK/badver/frames" >/dev/null ||
    fail "invalid version rebuild frames"
cmp -s "$WORK/serial/manifest.txt" "$WORK/badver/manifest.txt" ||
    fail "invalid version rebuild manifest"
pass "unsupported version quarantined and rebuilt"

echo "--- input hash change ---"
cp -a "$WORK/serial" "$WORK/hashchange"
run "$WORK/hashchange" --threads 4 --frames 16 --width 641 --height 480 >/dev/null
assert_summary "$WORK/hashchange" complete 0 16 E1004
grep -q "reason=input_hash_changed" "$WORK/hashchange/errors/input_hash_changed.txt" ||
    fail "input hash report"
[[ -d "$WORK/hashchange/quarantine/input_hash_changed/frames" ]] ||
    fail "old snapshot was not quarantined after input hash change"
pass "input hash change rebuilds the snapshot"

echo "--- forced termination and resume ---"
run "$WORK/kill" --threads 4 --frames 24 --width 640 --height 480 \
    --commit-delay-ms 200 >"$WORK/kill-partial.log" 2>&1 &
kill_pid=$!
seen=0
for _ in $(seq 1 300); do
    if [[ -f "$WORK/kill/manifest.txt" ]]; then
        seen="$(grep -c '^entry ' "$WORK/kill/manifest.txt" || true)"
    fi
    if [[ "$seen" -ge 3 ]]; then
        kill -9 "$kill_pid"
        break
    fi
    sleep 0.02
done
set +e
wait "$kill_pid"
killed_status=$?
set -e
[[ "$killed_status" -ne 0 ]] || fail "forced-kill process exited normally"
[[ ! -f "$WORK/kill/summary.txt" ]] ||
    fail "forced-kill process unexpectedly wrote a summary"
run "$WORK/kill" --threads 5 --frames 24 --width 640 --height 480 >/dev/null
grep -q "status=complete" "$WORK/kill/summary.txt" ||
    fail "kill resume status"
grep -q "error_codes=E2101" "$WORK/kill/summary.txt" ||
    fail "kill resume missing-entry code"
run "$WORK/kill-ref" --threads 1 --frames 24 --width 640 --height 480 >/dev/null
diff -rq "$WORK/kill-ref/frames" "$WORK/kill/frames" >/dev/null ||
    fail "kill resume frame outputs"
cmp -s "$WORK/kill-ref/manifest.txt" "$WORK/kill/manifest.txt" ||
    fail "kill resume manifest"
pass "SIGKILL recovery matches clean execution"

echo "--- SIGINT summary ---"
run "$WORK/sigint" --threads 4 --frames 24 --width 640 --height 480 \
    --commit-delay-ms 200 >"$WORK/sigint.log" 2>&1 &
sigint_pid=$!
seen=0
for _ in $(seq 1 300); do
    if [[ -f "$WORK/sigint/manifest.txt" ]]; then
        seen="$(grep -c '^entry ' "$WORK/sigint/manifest.txt" || true)"
    fi
    if [[ "$seen" -ge 2 ]]; then
        kill -INT "$sigint_pid"
        break
    fi
    sleep 0.02
done
set +e
wait "$sigint_pid"
sigint_status=$?
set -e
[[ "$sigint_status" -eq 130 ]] || fail "SIGINT exit code was $sigint_status"
grep -q "status=interrupted" "$WORK/sigint/summary.txt" ||
    fail "SIGINT summary status"
grep -q "error_codes=E9001" "$WORK/sigint/summary.txt" ||
    fail "SIGINT summary error code"
run "$WORK/sigint" --threads 3 --frames 24 --width 640 --height 480 >/dev/null
diff -rq "$WORK/kill-ref/frames" "$WORK/sigint/frames" >/dev/null ||
    fail "SIGINT resume frame outputs"
cmp -s "$WORK/kill-ref/manifest.txt" "$WORK/sigint/manifest.txt" ||
    fail "SIGINT resume manifest"
pass "SIGINT summary and resume"

echo "ALL INTEGRATION TESTS PASSED"

