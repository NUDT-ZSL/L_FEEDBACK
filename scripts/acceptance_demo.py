"""End-to-end acceptance demonstration using deterministic local samples.

Generates the fixture scenario, runs the CLI twice into fresh output
directories and prints the measured acceptance evidence (exit codes, WAV
format, loudness, overlap records, warnings and byte-identical hashes).

Usage: ``python scripts/acceptance_demo.py [workdir]``
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import soundfile as sf

# Make the project root importable when run directly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.samples import build_full_scenario  # noqa: E402


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main(workdir: str) -> int:
    input_dir = build_full_scenario(os.path.join(workdir, "fixture"))
    runs = []
    for run_name in ("run1", "run2"):
        run_dir = os.path.join(workdir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        wav_path = os.path.join(run_dir, "mix.wav")
        report_path = os.path.join(run_dir, "report.json")
        proc = subprocess.run(
            [sys.executable, "-m", "audiopack", input_dir, wav_path, "--report", report_path],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        runs.append((proc, wav_path, report_path))

    proc1, wav1, report1 = runs[0]
    proc2, wav2, report2 = runs[1]
    info = sf.info(wav1)
    report = json.load(open(report1, encoding="utf-8"))

    print("=== CLI ===")
    print("exit codes:", proc1.returncode, proc2.returncode)
    print("stderr:", proc1.stderr.strip())
    print()
    print("=== WAV format ===")
    print(info.samplerate, "Hz,", info.channels, "channels,", info.subtype, info.format)
    print()
    print("=== Determinism (sha256) ===")
    wav_hashes = (sha256(wav1), sha256(wav2))
    report_hashes = (sha256(report1), sha256(report2))
    print("wav byte-identical:", wav_hashes[0] == wav_hashes[1], wav_hashes[0])
    print("report byte-identical:", report_hashes[0] == report_hashes[1], report_hashes[0])
    print()
    print("=== Report summary ===")
    print("status:", report["status"])
    print("warnings:", report["warnings"])
    print("error_codes:", report["error_codes"])
    print("output_frames:", report["output_frames"], "(3.6 s * 48000 = 172800)")
    print("loudness:", json.dumps(report["loudness"]))
    print()
    print("=== Overlap records ===")
    for record in report["overlaps"]:
        print(json.dumps(record, sort_keys=True))
    print()
    print("=== Warning details ===")
    for detail in report["warning_details"]:
        print(json.dumps(detail, sort_keys=True))
    print()
    print("=== Event slice alignment ===")
    for event in report["events"]:
        print(
            f"event {event['event_index']}: {event['status']} "
            f"samples [{event['start_sample']}, {event['end_sample']}) "
            f"fades in/out {event['fade_in_samples']}/{event['fade_out_samples']} "
            f"gain {event['gain_db']} dB"
        )
    return 0


if __name__ == "__main__":
    workdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "tmp", "acceptance")
    raise SystemExit(main(workdir))
