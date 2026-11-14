#!/usr/bin/env sh
set -eu

REPO="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
OUT="${1:-$REPO/artifacts/validation}"

cd "$REPO"
"$PYTHON" -m radio_pipeline run --config config/default.json --output-dir "$OUT"
"$PYTHON" -m radio_pipeline report --run-dir "$OUT" --output "$OUT/report.rebuilt.json"
"$PYTHON" -m radio_pipeline verify --run-dir "$OUT"
"$PYTHON" -m radio_pipeline run --config config/default.json --output-dir "$OUT/repeat"
"$PYTHON" scripts/check_binary_determinism.py "$OUT" "$OUT/repeat"
echo "Validation passed: run, report, verify, and binary determinism."

