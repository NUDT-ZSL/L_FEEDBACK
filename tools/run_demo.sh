#!/usr/bin/env bash
# Deterministic end-to-end demonstration of the audio packaging pipeline.
# Generates local WAV samples + manifest, then packs, interrupts and resumes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
DEMO="$ROOT/demo_run"

echo "== [1/4] generate deterministic samples and manifest =="
"$PY" -m audiopack.cli generate \
  --assets-dir "$DEMO/assets" \
  --manifest "$DEMO/manifest.json"

echo "== [2/4] interrupted run (stop after the 'en' block) =="
"$PY" -m audiopack.cli pack \
  --manifest "$DEMO/manifest.json" \
  --assets-dir "$DEMO/assets" \
  --work-dir "$DEMO/work" \
  --output-dir "$DEMO/out" \
  --max-workers 3 \
  --stop-after-lang en || true

echo "== [3/4] resume run (reuse en, recompute ja/zh) =="
"$PY" -m audiopack.cli pack \
  --manifest "$DEMO/manifest.json" \
  --assets-dir "$DEMO/assets" \
  --work-dir "$DEMO/work" \
  --output-dir "$DEMO/out" \
  --max-workers 3

echo "== [4/4] third run (every block reused) =="
"$PY" -m audiopack.cli pack \
  --manifest "$DEMO/manifest.json" \
  --assets-dir "$DEMO/assets" \
  --work-dir "$DEMO/work" \
  --output-dir "$DEMO/out3" \
  --max-workers 3 > "$DEMO/final_report.json"

echo
echo "Artifacts under: $DEMO"
echo "Final report:     $DEMO/final_report.json"
