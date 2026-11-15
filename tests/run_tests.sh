#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
EXE="${1:-./build/pbr_renderer}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/pbr-renderer-test.XXXXXX")"
PASS_COUNT=0

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    echo "PASS: $*"
}

expect_exit() {
    local expected="$1"
    local name="$2"
    shift 2
    local log="$TMP/${name}.log"
    "$EXE" "$@" >"$log" 2>&1
    local actual=$?
    if [[ "$actual" -ne "$expected" ]]; then
        cat "$log" >&2
        fail "$name expected exit $expected, got $actual"
    fi
    pass "$name exit $actual"
}

python3 - "$TMP" <<'PY'
import copy
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
base = json.loads(pathlib.Path("assets/scene.json").read_text(encoding="utf-8"))

missing = copy.deepcopy(base)
del missing["camera"]["f_number"]
(out / "missing_field.json").write_text(json.dumps(missing), encoding="utf-8")

bad_type = copy.deepcopy(base)
bad_type["materials"][0]["roughness"] = "smooth"
(out / "bad_type.json").write_text(json.dumps(bad_type), encoding="utf-8")

bad_id = copy.deepcopy(base)
bad_id["spheres"][0]["material_id"] = 99
(out / "bad_material_id.json").write_text(json.dumps(bad_id), encoding="utf-8")
PY

expect_exit 1 invalid_samples --samples 0 --output "$TMP/unused"
expect_exit 1 invalid_tonemap --tonemap filmic --output "$TMP/unused"
expect_exit 2 missing_scene --scene "$TMP/does_not_exist.json" --output "$TMP/unused"
expect_exit 2 missing_field --scene "$TMP/missing_field.json" --output "$TMP/unused"
expect_exit 2 wrong_type --scene "$TMP/bad_type.json" --output "$TMP/unused"
expect_exit 2 material_id_out_of_range --scene "$TMP/bad_material_id.json" --output "$TMP/unused"
expect_exit 3 missing_reference --scene assets/scene.json --output "$TMP/missing_reference" \
    --mode validate --samples 2 --subframes 1 --reference "$TMP/no-reference.json"

touch "$TMP/blocked_parent"
expect_exit 3 output_io_failure --scene assets/scene.json --output "$TMP/blocked_parent/render"

expect_exit 0 deterministic_a --scene assets/scene.json --output "$TMP/deterministic_a" \
    --samples 8 --subframes 2 --exposure 0 --tonemap aces
expect_exit 0 deterministic_b --scene assets/scene.json --output "$TMP/deterministic_b" \
    --samples 8 --subframes 2 --exposure 0 --tonemap aces
cmp -s "$TMP/deterministic_a_ev0_aces.hdr" "$TMP/deterministic_b_ev0_aces.hdr" ||
    fail "repeated HDR renders are not byte-identical"
pass "repeated HDR renders are byte-identical"

mkdir -p "$TMP/output_directory"
expect_exit 0 output_directory --scene assets/scene.json --output "$TMP/output_directory" \
    --samples 2 --subframes 1 --exposure 0 --tonemap aces
test -f "$TMP/output_directory/render_ev0_aces.hdr" ||
    fail "directory output did not use render as the file prefix"
pass "directory --output uses render as the file prefix"

for exposure in -1 0 1; do
    for curve in reinhard aces uncharted2; do
        expect_exit 0 "color_${exposure}_${curve}" --scene assets/scene.json \
            --output "$TMP/color" --samples 2 --subframes 1 \
            --exposure "$exposure" --tonemap "$curve"
    done
done

python3 - "$TMP" <<'PY'
import pathlib
import struct
import sys

root = pathlib.Path(sys.argv[1])
png_files = sorted(root.glob("color_ev*_*.png"))
if len(png_files) != 9:
    raise SystemExit(f"expected 9 PNG files, found {len(png_files)}")
for path in png_files:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit(f"not a PNG: {path}")
    width, height = struct.unpack(">II", data[16:24])
    if (width, height) != (512, 512):
        raise SystemExit(f"wrong PNG dimensions in {path}: {width}x{height}")

hdr_files = sorted(root.glob("color_ev*_*.hdr"))
if len(hdr_files) != 9:
    raise SystemExit(f"expected 9 HDR files, found {len(hdr_files)}")
for path in hdr_files:
    text = path.read_text(encoding="latin-1")
    if "\n-Y 512 +X 512\n" not in text:
        raise SystemExit(f"wrong HDR resolution header in {path}")
    if "# exposure_ev=" not in text or "# tonemap=" not in text:
        raise SystemExit(f"missing HDR parameter comments in {path}")
PY
pass "all 3 exposures x 3 tone curves produce 512x512 HDR/PNG"

expect_exit 0 validation_over_threshold --scene assets/scene.json --output "$TMP/low" \
    --mode validate --samples 8 --subframes 2 --reference assets/reference_stats.json \
    --report "$TMP/low_report.txt"
grep -q "SUMMARY: FAIL" "$TMP/low_report.txt" ||
    fail "low-sample validation did not report threshold failure"
pass "out-of-threshold validation writes a FAIL report and exits 0"

expect_exit 0 validation_baseline --scene assets/scene.json --output "$TMP/baseline" \
    --mode validate --samples 256 --subframes 8 --reference assets/reference_stats.json \
    --report "$TMP/baseline_report.txt"
grep -q "SUMMARY: PASS" "$TMP/baseline_report.txt" ||
    fail "baseline validation did not pass"
pass "baseline validation passes"

echo "ALL TESTS PASSED ($PASS_COUNT checks); logs: $TMP"
