#!/usr/bin/env bash
set -u

binary="${1:-./build/uid166}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
status=0

expect_exit() {
    local expected="$1"
    shift
    "$@" >/dev/null 2>"$work/stderr.txt"
    local actual=$?
    local expected_host=$((expected % 256))
    if [[ "$actual" -ne "$expected_host" ]] || ! grep -q "code=$expected " "$work/stderr.txt"; then
        echo "FAIL expected business code $expected (host exit $expected_host) got $actual: $*" >&2
        cat "$work/stderr.txt" >&2
        status=1
    else
        echo "PASS business-code=$expected host-exit=$actual $*"
    fi
}

"$binary" --benchmark --frames 1 --objects 64 --data-dir "$work/data" \
    --output "$work/out.json" >/dev/null

expect_exit 2001 "$binary" --benchmark --frames 1 --reuse-data \
    --data-dir "$work/missing" --output "$work/missing.json"

cp -a "$work/data" "$work/no_animation"
rm "$work/no_animation/uid166_character.skan"
expect_exit 2101 "$binary" --benchmark --frames 1 --reuse-data \
    --data-dir "$work/no_animation" --output "$work/noanim.json"

cp -a "$work/data" "$work/truncated"
truncate -s -12 "$work/truncated/uid166_character.skan"
expect_exit 2103 "$binary" --benchmark --frames 1 --reuse-data \
    --data-dir "$work/truncated" --output "$work/truncated.json"

cp -a "$work/data" "$work/bad_scene"
printf '{"version":1,' >"$work/bad_scene/uid166_scene.json"
expect_exit 2003 "$binary" --benchmark --frames 1 --reuse-data \
    --data-dir "$work/bad_scene" --output "$work/badscene.json"

exit "$status"
