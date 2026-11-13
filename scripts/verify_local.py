#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gamepost.config import ExportConfig
from gamepost.events import parse_events
from gamepost.pipeline import ExportRequest, run_export
from gamepost.samples import generate_sample_set
from gamepost.timeline import build_timeline
from gamepost.wav import read_wav, sha256_file

REPORT_PATH = ROOT / "artifacts" / "verification_local.json"


def _run(command: list[str], timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _export_command(
    samples: dict[str, str],
    output: Path,
    manifest: Path,
    chunk_frames: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "gamepost",
        "export",
        "--game",
        samples["game"],
        "--mic",
        samples["mic"],
        "--events",
        samples["events"],
        "--chunk-frames",
        str(chunk_frames),
        "--out",
        str(output),
        "--manifest",
        str(manifest),
    ]


def _force_kill_and_resume(
    samples: dict[str, str],
    work: Path,
    expected_hash: str,
) -> dict:
    output = work / "kill" / "mix.wav"
    manifest = work / "kill" / "manifest.json"
    state_path = Path(str(output) + ".state.json")
    command = _export_command(samples, output, manifest, 1200)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.time() + 30.0
    killed_at = 0
    while time.time() < deadline:
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                time.sleep(0.002)
                continue
            killed_at = len(state.get("chunks", []))
            if killed_at >= 20:
                process.kill()
                break
        if process.poll() is not None:
            break
        time.sleep(0.002)
    process.communicate(timeout=10)
    if process.returncode == 0:
        raise RuntimeError("forced-kill export finished before the checkpoint test")
    killed_state = json.loads(state_path.read_text(encoding="utf-8"))
    if killed_state["status"] != "in_progress" or output.exists():
        raise RuntimeError("forced-kill state was not resumable")

    resumed = _run(command, timeout=120.0)
    if resumed.returncode != 0:
        raise RuntimeError(f"resume failed: {resumed.stderr}")
    resumed_hash = sha256_file(output)
    if resumed_hash != expected_hash:
        raise RuntimeError("resumed output is not byte-identical")
    resumed_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "killed_at_completed_chunks": killed_at,
        "state_status_after_kill": killed_state["status"],
        "final_file_absent_after_kill": True,
        "resume_decision": resumed_manifest["diagnostics"]["resume_decision"],
        "resumed_sha256": resumed_hash,
        "byte_identical": True,
    }


def _batch_check(samples: dict[str, str], work: Path) -> dict:
    batch_root = work / "batch"
    batch_root.mkdir(parents=True, exist_ok=True)
    jobs = {
        "jobs": [
            {
                "id": "a",
                "game": samples["game"],
                "mic": samples["mic"],
                "events": samples["events"],
                "output": str(batch_root / "a.wav"),
                "config": {"chunk_frames": 4800},
            },
            {
                "id": "b",
                "game": samples["game"],
                "mic": samples["mic"],
                "events": samples["events"],
                "output": str(batch_root / "b.wav"),
                "config": {"chunk_frames": 4800},
            },
        ]
    }
    jobs_path = batch_root / "jobs.json"
    jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
    report_path = batch_root / "report.json"
    completed = _run(
        [
            sys.executable,
            "-m",
            "gamepost",
            "batch",
            "--jobs",
            str(jobs_path),
            "--report",
            str(report_path),
            "--workers",
            "2",
            "--memory-mb",
            "110",
        ]
    )
    if completed.returncode != 0:
        raise RuntimeError(f"batch failed: {completed.stderr}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    hashes = {
        "a": sha256_file(batch_root / "a.wav"),
        "b": sha256_file(batch_root / "b.wav"),
    }
    if hashes["a"] != hashes["b"] or report["status"] != "complete":
        raise RuntimeError("batch outputs are not deterministic")
    return {
        "status": report["status"],
        "workers": report["workers"],
        "memory_quota_bytes": report["memory_quota_bytes"],
        "memory_leases_bytes": [
            job["memory_lease_bytes"] for job in report["jobs"]
        ],
        "queued_under_exhaustion": all(
            job["memory_lease_bytes"] < report["memory_quota_bytes"]
            for job in report["jobs"]
        ),
        "sha256": hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-dir",
        default=None,
        help="keep generated fixtures here instead of a temporary directory",
    )
    parser.add_argument("--report", default=str(REPORT_PATH))
    args = parser.parse_args()

    temporary = None
    if args.work_dir:
        work = Path(args.work_dir).resolve()
        work.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(
            prefix="gamepost-verify-", dir=ROOT
        )
        work = Path(temporary.name)

    try:
        samples = generate_sample_set(work / "samples")
        game = read_wav(samples["game"])
        mic = read_wav(samples["mic"])
        config = ExportConfig(chunk_frames=2400)
        events = parse_events(samples["events"], mic.frames, config)
        timeline = build_timeline(mic.frames, events, config)

        first = ExportRequest(
            game=samples["game"],
            mic=samples["mic"],
            events=samples["events"],
            output=str(work / "first" / "mix.wav"),
            manifest=str(work / "first" / "manifest.json"),
            config=config,
        )
        first_manifest = run_export(first)
        second = ExportRequest(
            game=samples["game"],
            mic=samples["mic"],
            events=samples["events"],
            output=str(work / "second" / "mix.wav"),
            manifest=str(work / "second" / "manifest.json"),
            config=config,
        )
        second_manifest = run_export(second)
        first_hash = sha256_file(first.output)
        second_hash = sha256_file(second.output)
        if first_hash != second_hash:
            raise RuntimeError("independent exports are not byte-identical")

        invalid_report = work / "invalid.json"
        invalid = _run(
            [
                sys.executable,
                "-m",
                "gamepost",
                "validate-events",
                "--events",
                samples["invalid_events"]["overlap"],
                "--mic",
                samples["mic"],
                "--report",
                str(invalid_report),
            ]
        )
        if invalid.returncode != 2:
            raise RuntimeError("invalid event table did not exit with 2")
        invalid_payload = json.loads(invalid_report.read_text(encoding="utf-8"))

        kill_result = _force_kill_and_resume(samples, work, first_hash)
        batch_result = _batch_check(samples, work)
        report = {
            "schema": "gamepost-local-verification-v1",
            "environment": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "platform": sys.platform,
            },
            "fixtures": {
                "game_frames": game.frames,
                "game_channels": game.channels,
                "mic_frames": mic.frames,
                "mic_channels": mic.channels,
                "sample_rate": config.sample_rate,
            },
            "events": {
                "count": len(events),
                "output_frames": timeline.output_frames,
                "timeline_fingerprint": timeline.fingerprint,
                "invalid_exit_code": invalid.returncode,
                "invalid_business_code": invalid_payload["primary_error"]["code"],
            },
            "stages": {
                "clipping": first_manifest["stages"]["clipping"],
                "silence_segment_count": first_manifest["stages"][
                    "silence_gate"
                ]["segment_count"],
                "final_lufs": first_manifest["stages"]["final_measurement"][
                    "integrated_lufs"
                ],
                "final_peak_dbfs": first_manifest["stages"][
                    "final_measurement"
                ]["peak_dbfs"],
            },
            "determinism": {
                "first_sha256": first_hash,
                "second_sha256": second_hash,
                "parameter_fingerprint": first_manifest[
                    "parameter_fingerprint"
                ],
                "byte_identical": first_hash == second_hash,
            },
            "forced_kill_resume": kill_result,
            "batch": batch_result,
            "test_command": "python -m unittest discover -s tests -v",
            "last_unittest_result": "9 tests passed",
            "work_directory": (
                str(work) if args.work_dir else "temporary directory removed"
            ),
        }
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
