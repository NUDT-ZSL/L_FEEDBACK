#!/usr/bin/env bash
# Deterministic end-to-end demonstration used to record validation results.
set -euo pipefail

PY="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$ROOT/scratch/demo"
IN="$WORK/in"
FULL="$WORK/full"
RESUME="$WORK/resume"

echo "== generate =="
"$PY" -m voiceforge generate "$IN"

echo "== one-shot synthesis =="
"$PY" -m voiceforge synthesize "$IN" "$FULL"

echo "== interrupted synthesis (checkpoint after 3 intervals) then resume =="
VOICEFORGE_INTERRUPT_AFTER=3 "$PY" -m voiceforge synthesize "$IN" "$RESUME"
"$PY" -m voiceforge synthesize "$IN" "$RESUME"

echo "== byte comparison =="
(
  cd "$WORK"
  sha256sum full/voice_track.wav resume/voice_track.wav
  sha256sum full/timeline.json resume/timeline.json
  sha256sum full/summary.json resume/summary.json
  sha256sum full/mix_float.f32 resume/mix_float.f32
) | tee "$WORK/verification.txt"

echo "== idempotent re-run summary =="
"$PY" -m voiceforge synthesize "$IN" "$FULL.repeat"
(cd "$WORK" && sha256sum full/summary.json full.repeat/summary.json) \
  | tee -a "$WORK/verification.txt"

