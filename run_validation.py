#!/usr/bin/env python3
"""One-shot local validation: generate samples, run both manifests, check
the acceptance criteria and print a summary. Exits non-zero on failure."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"


def sh(cmd):
    print("+", " ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd], cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {cmd}")


def check(report_path):
    report = json.loads(Path(report_path).read_text())
    ver = report["verification"]
    checks = {
        "ffprobe readable / 2 streams": len(ver["streams"]) == 2,
        "video DTS monotonic": ver["video_dts_monotonic"],
        "audio PTS monotonic": ver["audio_pts_monotonic"],
        "A/V sync within +/-20ms": ver["sync"]["within_tolerance"],
        "no frame gaps or duplicates": ver["continuity"]["continuous"],
        "PES lengths valid": not ver["ts_structure"]["pes_length_violations"],
        "distinct A/V PIDs": ver["ts_structure"]["video_pid"] != ver["ts_structure"]["audio_pid"],
    }
    return checks, report


def main():
    sh(["python3", "create_samples.py"])
    OUT.mkdir(exist_ok=True)
    failures = 0
    for manifest, name in (("manifest.json", "switched.ts"),
                           ("manifest_fallback.json", "switched_fallback.ts")):
        sh(["python3", "-m", "avswitch", manifest, OUT / name])
        checks, report = check(OUT / f"{name}.report.json")
        print(f"\n== {manifest} ==")
        for label, ok in checks.items():
            print(f"  [{'OK' if ok else 'FAIL'}] {label}")
            failures += 0 if ok else 1
        print(f"  switches={len(report['switches'])} fallbacks={len(report['fallbacks'])} "
              f"max_sync={report['verification']['sync']['max_abs_ms']}ms")
    fb = json.loads((OUT / "switched_fallback.ts.report.json").read_text())
    ok = len(fb["fallbacks"]) >= 1 and all(
        f.get("reason") and f.get("fallback_stream") for f in fb["fallbacks"])
    print(f"[{'OK' if ok else 'FAIL'}] fallback logged with reason and time")
    failures += 0 if ok else 1
    print("\nVALIDATION:", "PASS" if failures == 0 else f"FAIL ({failures} checks)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
