#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/artifacts/offline}"
CONFIG="${CONFIG:-$ROOT/config/render.json}"
GPU_BACKEND="${GPU_BACKEND:-0}"

cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DDEFERRED_BUILD_TESTS=ON
cmake --build "$BUILD_DIR" -j"${JOBS:-2}"

"$BUILD_DIR/deferred_core_tests"

BACKEND="cpu"
if [[ "$GPU_BACKEND" == "1" ]]; then
    BACKEND="opengl"
fi

"$BUILD_DIR/deferred_fog" --config "$CONFIG" --offline --backend "$BACKEND" \
    --output "$OUTPUT_DIR"

if [[ "$GPU_BACKEND" == "1" ]]; then
    "$BUILD_DIR/deferred_fog" --config "$CONFIG" --offline --backend cpu \
        --output "$ROOT/artifacts/offline_cpu"
    "$BUILD_DIR/deferred_verify" "$OUTPUT_DIR" "$ROOT/artifacts/offline_cpu" \
        > "$ROOT/artifacts/verification_report.json"
else
    "$BUILD_DIR/deferred_verify" "$OUTPUT_DIR" > "$ROOT/artifacts/verification_report.json"
fi

expect_error() {
    local name="$1"
    local expected="$2"
    shift 2
    set +e
    "$@" >"$ROOT/artifacts/${name}.stdout" 2>"$ROOT/artifacts/${name}.stderr"
    local actual=$?
    set -e
    if [[ "$actual" != "$expected" ]]; then
        echo "FAIL: $name expected exit $expected, got $actual" >&2
        exit 1
    fi
    echo "$name exit=$actual stderr=$(cat "$ROOT/artifacts/${name}.stderr")"
}

expect_error invalid_steps 2 \
    "$BUILD_DIR/deferred_fog" --config "$ROOT/config/invalid_steps.json" \
    --offline --backend cpu
expect_error invalid_syntax 2 \
    "$BUILD_DIR/deferred_fog" --config "$ROOT/config/invalid_syntax.json" \
    --offline --backend cpu
expect_error invalid_type 2 \
    "$BUILD_DIR/deferred_fog" --config "$ROOT/config/invalid_type.json" \
    --offline --backend cpu
expect_error gbuffer_failure 3 \
    "$BUILD_DIR/deferred_fog" --config "$CONFIG" --offline --backend cpu \
    --inject-gbuffer-failure
expect_error shader_failure 4 \
    "$BUILD_DIR/deferred_fog" --config "$CONFIG" --offline --backend cpu \
    --inject-shader-failure

cat > "$ROOT/artifacts/error_codes.json" <<'JSON'
{
  "schema": "deferred-fog-error-codes/v1",
  "results": [
    {"case": "invalid_steps", "code": "E_CONFIG_PARSE", "exit": 2},
    {"case": "invalid_syntax", "code": "E_CONFIG_PARSE", "exit": 2},
    {"case": "invalid_type", "code": "E_CONFIG_PARSE", "exit": 2},
    {"case": "gbuffer_failure", "code": "E_GBUFFER_FAIL", "exit": 3},
    {"case": "shader_failure", "code": "E_SHADER_COMPILE", "exit": 4}
  ]
}
JSON

echo "validation complete: $ROOT/artifacts/verification_report.json"

