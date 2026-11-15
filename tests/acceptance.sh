#!/usr/bin/env bash
set -euo pipefail

RENDERER="$1"
SOURCE_DIR="$2"
WORK_DIR="$3"

mkdir -p "$WORK_DIR"
rm -f "$WORK_DIR"/*.png "$WORK_DIR"/*.json "$WORK_DIR"/*.log

SHADERS="$SOURCE_DIR/shaders"
PIPELINE="$SOURCE_DIR/configs/pipeline.json"
RECONFIG="$SOURCE_DIR/configs/pipeline_reconfigured.json"
COMMON=(--seed 1337 --width 320 --height 180 --warmup 2 --shaders "$SHADERS")

"$RENDERER" "${COMMON[@]}" --pipeline "$PIPELINE" \
    --output "$WORK_DIR/run_a.png" >"$WORK_DIR/run_a.log" 2>"$WORK_DIR/run_a.err"
"$RENDERER" "${COMMON[@]}" --pipeline "$PIPELINE" \
    --output "$WORK_DIR/run_b.png" >"$WORK_DIR/run_b.log" 2>"$WORK_DIR/run_b.err"

cmp "$WORK_DIR/run_a.png" "$WORK_DIR/run_b.png"
test ! -s "$WORK_DIR/run_a.err"
test ! -s "$WORK_DIR/run_b.err"

"$RENDERER" "${COMMON[@]}" --pipeline "$PIPELINE" \
    --output "$WORK_DIR/state_before.png" \
    --save-state "$WORK_DIR/state.json" \
    --state-roundtrip "$WORK_DIR/roundtrip.json" \
    >"$WORK_DIR/state.log" 2>"$WORK_DIR/state.err"

cmp "$WORK_DIR/run_a.png" "$WORK_DIR/state_before.png"
cmp "$WORK_DIR/run_a.png" "$WORK_DIR/state_before.png.restored.png"
test ! -s "$WORK_DIR/state.err"
python3 - "$WORK_DIR/state.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    state = json.load(stream)
assert state["version"] == 1, state
assert [stage["name"] for stage in state["stages"]] == [
    "geometry", "shadow", "deferred", "postprocess"
], state
assert [effect["type"] for effect in state["stages"][-1]["effects"]] == [
    "bloom", "tonemap", "fxaa", "color_grade"
], state
PY

"$RENDERER" "${COMMON[@]}" --pipeline "$WORK_DIR/state.json" \
    --output "$WORK_DIR/state_loaded.png" \
    >"$WORK_DIR/state_loaded.log" 2>"$WORK_DIR/state_loaded.err"
cmp "$WORK_DIR/run_a.png" "$WORK_DIR/state_loaded.png"
test ! -s "$WORK_DIR/state_loaded.err"

"$RENDERER" "${COMMON[@]}" --pipeline "$RECONFIG" \
    --output "$WORK_DIR/reconfigured_direct.png" \
    >"$WORK_DIR/reconfigured_direct.log" 2>"$WORK_DIR/reconfigured_direct.err"

"$RENDERER" "${COMMON[@]}" --pipeline "$PIPELINE" \
    --output "$WORK_DIR/reconfigured_base.png" \
    --reconfigure "$RECONFIG" \
    --reconfigure-output "$WORK_DIR/reconfigured_same_process.png" \
    --state-roundtrip "$WORK_DIR/reconfigured_restored.json" \
    >"$WORK_DIR/reconfigured.log" 2>"$WORK_DIR/reconfigured.err"

cmp "$WORK_DIR/reconfigured_direct.png" \
    "$WORK_DIR/reconfigured_same_process.png"
cmp "$WORK_DIR/reconfigured_same_process.png" \
    "$WORK_DIR/reconfigured_same_process.png.restored.png"
test ! -s "$WORK_DIR/reconfigured_direct.err"
test ! -s "$WORK_DIR/reconfigured.err"

cat >"$WORK_DIR/unknown_stage.json" <<'JSON'
{
  "version": 1,
  "stages": [
    {"name": "geometry", "enabled": true},
    {"name": "volumetric_magic", "enabled": true}
  ]
}
JSON

set +e
"$RENDERER" "${COMMON[@]}" --pipeline "$WORK_DIR/unknown_stage.json" \
    --output "$WORK_DIR/unknown.png" \
    >"$WORK_DIR/unknown.out" 2>"$WORK_DIR/unknown.err"
UNKNOWN_EXIT=$?
set -e
test "$UNKNOWN_EXIT" -ne 0
python3 - "$WORK_DIR/unknown.err" "$UNKNOWN_EXIT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    lines = stream.read().splitlines()
assert len(lines) == 1, lines
value = json.loads(lines[0])
assert value["type"] == "unknown_stage", value
assert value["exit_code"] == int(sys.argv[2]), value
assert "volumetric_magic" in value["message"], value
PY

cat >"$WORK_DIR/unsupported_version.json" <<'JSON'
{
  "version": 999,
  "stages": [
    {"name": "geometry", "enabled": true}
  ]
}
JSON

set +e
"$RENDERER" "${COMMON[@]}" --pipeline "$WORK_DIR/unsupported_version.json" \
    --output "$WORK_DIR/unsupported.png" \
    >"$WORK_DIR/unsupported.out" 2>"$WORK_DIR/unsupported.err"
UNSUPPORTED_EXIT=$?
set -e
test "$UNSUPPORTED_EXIT" -ne 0
python3 - "$WORK_DIR/unsupported.err" "$UNSUPPORTED_EXIT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    lines = stream.read().splitlines()
assert len(lines) == 1, lines
value = json.loads(lines[0])
assert value["type"] == "unsupported_version", value
assert value["exit_code"] == int(sys.argv[2]), value
assert "999" in value["message"], value
PY

mkdir -p "$WORK_DIR/badshaders"
cp "$SHADERS"/* "$WORK_DIR/badshaders/"
printf '#version 330 core\nvoid broken( {\n' >"$WORK_DIR/badshaders/geometry.frag"

set +e
"$RENDERER" "${COMMON[@]}" --pipeline "$PIPELINE" \
    --shaders "$WORK_DIR/badshaders" --output "$WORK_DIR/badshader.png" \
    >"$WORK_DIR/badshader.out" 2>"$WORK_DIR/badshader.err"
BAD_SHADER_EXIT=$?
set -e
test "$BAD_SHADER_EXIT" -ne 0
python3 - "$WORK_DIR/badshader.err" "$BAD_SHADER_EXIT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    lines = stream.read().splitlines()
assert len(lines) == 1, lines
value = json.loads(lines[0])
assert value["type"] == "shader_compile", value
assert value["exit_code"] == int(sys.argv[2]), value
assert "geometry.frag" in value["message"], value
assert "error:" in value["message"].lower(), value
PY

"$RENDERER" "${COMMON[@]}" --pipeline "$PIPELINE" --no-lights \
    --output "$WORK_DIR/no_lights.png" \
    >"$WORK_DIR/no_lights.log" 2>"$WORK_DIR/no_lights.err"
test ! -s "$WORK_DIR/no_lights.err"
python3 "$SOURCE_DIR/tests/check_png_nonblack.py" \
    "$WORK_DIR/no_lights.png" 0.02 20

echo "Acceptance checks passed."

