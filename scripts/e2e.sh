#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOTNET="${DOTNET:-dotnet}"
RID="${RID:-linux-x64}"
DURATION="${DURATION:-20}"
OUT="${OUT:-$(mktemp -d "${ROOT}/artifacts/e2e.XXXXXX")}"
PROJECT="$ROOT/src/ReplayForge/ReplayForge.csproj"
DLL="$ROOT/src/ReplayForge/bin/Release/net8.0/$RID/replayforge.dll"

echo "workspace: $OUT"
"$DOTNET" build "$PROJECT" -c Release -r "$RID" >/dev/null

"$DOTNET" "$DLL" generate --dir "$OUT" --duration "$DURATION" --seed 1337 --overwrite >"$OUT/generate.json"

set +e
setsid "$DOTNET" "$DLL" run --manifest "$OUT/manifest.json" \
  >"$OUT/crashed-run.json" 2>"$OUT/crashed-run.err" &
CRASH_PID=$!
set -e

COMPLETED=0
for _ in $(seq 1 3000); do
  if [[ -f "$OUT/work/resume.json" ]]; then
    COMPLETED="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["completedSegments"])' "$OUT/work/resume.json" 2>/dev/null || echo 0)"
    if [[ "$COMPLETED" -ge 3 ]]; then
      break
    fi
  fi
  if ! kill -0 "$CRASH_PID" 2>/dev/null; then
    break
  fi
  sleep 0.01
done

if [[ "$COMPLETED" -lt 3 ]]; then
  echo "render completed before an interruption could be injected" >&2
  wait "$CRASH_PID" || true
  exit 1
fi

kill -KILL -- "-$CRASH_PID"
wait "$CRASH_PID" 2>/dev/null || true
python3 - "$OUT" <<'PY'
import os, sys
root = sys.argv[1]
path = os.path.join(root, "output", "final.mp4")
with open(path, "ab") as stream:
    stream.write(b"SIMULATED-PARTIAL-GOP" * 256)
PY

"$DOTNET" "$DLL" run --manifest "$OUT/manifest.json" \
  >"$OUT/resume.json" 2>"$OUT/resume.err"

python3 - "$OUT" "$DURATION" <<'PY'
import json, os, sys
root, duration = sys.argv[1], int(sys.argv[2])
d = json.load(open(os.path.join(root, "resume.json")))
assert d["ok"] is True, d
assert d["recovery"]["mode"] == "forced-termination-recovery", d["recovery"]
assert d["recovery"]["previousCleanShutdown"] is False, d["recovery"]
assert d["recovery"]["rolledBackBytes"] > 0, d["recovery"]
assert d["validation"]["ok"] is True, d["validation"]
assert d["totalFrames"] == duration * 30, d["totalFrames"]
assert d["validation"]["keyframes"] == duration, d["validation"]
assert os.path.getsize(d["outputPath"]) == d["validation"]["fileLength"]
print("crash recovery:", json.dumps({
    "mode": d["recovery"]["mode"],
    "resumedFromSegment": d["recovery"]["resumedFromSegment"],
    "rolledBackBytes": d["recovery"]["rolledBackBytes"],
    "frames": d["totalFrames"],
    "duration": d["totalDurationSeconds"],
    "keyframes": d["validation"]["keyframes"],
    "fileLength": d["validation"]["fileLength"],
}, indent=2))
PY

"$DOTNET" "$DLL" run --manifest "$OUT/manifest.json" >"$OUT/normal-complete.json"
python3 - "$OUT" <<'PY'
import json, sys, os
root = sys.argv[1]
d = json.load(open(os.path.join(root, "normal-complete.json")))
assert d["ok"] is True
assert d["recovery"]["mode"] == "normal-already-complete", d["recovery"]
print("normal exit:", d["recovery"]["mode"])
PY

python3 - "$OUT" <<'PY'
import json, sys, os
root = sys.argv[1]
path = os.path.join(root, "manifest.json")
d = json.load(open(path))
d["timeline"]["masterStartPtsSeconds"] = 10.1
d["summaryPath"] = "sync-error-summary.json"
json.dump(d, open(os.path.join(root, "bad-sync.json"), "w"), indent=2)
PY
"$DOTNET" "$DLL" run --manifest "$OUT/bad-sync.json" >"$OUT/bad-sync-result.json"
python3 - "$OUT" <<'PY'
import json, sys, os
root = sys.argv[1]
d = json.load(open(os.path.join(root, "bad-sync-result.json")))
assert d["ok"] is False
assert d["error"]["code"] == "E_SYNC_FAILED"
assert d["error"]["layer"] == "main"
assert d["error"]["actualStartPtsSeconds"] == 10.0
print("business error:", d["error"]["code"], "layer=", d["error"]["layer"])
PY

FFPROBE="$ROOT/src/ReplayForge/bin/Release/net8.0/$RID/ffprobe"
FFMPEG="$ROOT/src/ReplayForge/bin/Release/net8.0/$RID/ffmpeg"
if [[ -x "$FFPROBE" && -x "$FFMPEG" ]]; then
  export LD_LIBRARY_PATH="$(dirname "$FFPROBE"):${LD_LIBRARY_PATH:-}"
  "$FFMPEG" -v error -i "$OUT/output/final.mp4" -f null -
  "$FFPROBE" -v error -show_entries format=duration,size:stream=codec_type,codec_name,profile,width,height,avg_frame_rate,nb_frames,sample_rate,channels -of json "$OUT/output/final.mp4"
fi

echo "PASS: $OUT"
