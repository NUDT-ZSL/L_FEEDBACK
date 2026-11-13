from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy
import numpy as np
import PIL
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from frame_render.sample import write_sample


REPORT_PATH = ROOT / "artifacts" / "acceptance_report.json"
RUN_ROOT = ROOT / "var" / "acceptance"


def command(
    arguments: list[str],
    *,
    expected: int = 0,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    if env:
        environment.update(env)
    result = subprocess.run(
        [sys.executable, "-m", "frame_render", *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != expected:
        raise AssertionError(
            f"command exit {result.returncode}, expected {expected}: "
            f"{' '.join(arguments)}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return result


def json_stdout(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return json.loads(result.stdout)


def json_stderr(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return json.loads(result.stderr)


def pixel_digest(sequence_dir: Path, frame_start: int, frame_end: int) -> str:
    digest = hashlib.sha256()
    for frame_number in range(frame_start, frame_end + 1):
        with Image.open(sequence_dir / f"frame_{frame_number:06d}.png") as image:
            pixels = np.asarray(image.convert("RGB"))
        digest.update(frame_number.to_bytes(8, "little", signed=False))
        digest.update(pixels.tobytes())
    return digest.hexdigest()


def file_digest(sequence_dir: Path, frame_start: int, frame_end: int) -> str:
    digest = hashlib.sha256()
    for frame_number in range(frame_start, frame_end + 1):
        path = sequence_dir / f"frame_{frame_number:06d}.png"
        digest.update(frame_number.to_bytes(8, "little", signed=False))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def chunk_id(frame_start: int, frame_end: int) -> str:
    return f"chunk_{frame_start:06d}_{frame_end:06d}"


def main() -> dict[str, Any]:
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True)
    sample_path = ROOT / "samples" / "scene_240.json"
    write_sample(
        sample_path,
        frame_count=240,
        width=128,
        height=72,
        samples_per_axis=4,
    )
    checks: dict[str, bool] = {}
    timings: dict[str, float] = {}
    digests: dict[str, str] = {}
    common = ["--input", str(sample_path), "--chunk-size", "20"]

    started = time.perf_counter()
    serial_output = RUN_ROOT / "jobs1"
    serial_work = RUN_ROOT / "work_jobs1"
    serial = command(
        [
            "run",
            *common,
            "--work-dir",
            str(serial_work),
            "--output",
            str(serial_output),
            "--jobs",
            "1",
            "--max-retries",
            "1",
        ]
    )
    timings["jobs1_full_seconds"] = round(time.perf_counter() - started, 3)
    serial_summary = json_stdout(serial)
    checks["full_240_frame_render"] = (
        serial_summary["status"] == "ok"
        and serial_summary["assembly"]["successful_frames"] == 240
    )

    started = time.perf_counter()
    parallel_output = RUN_ROOT / "jobs4"
    parallel_work = RUN_ROOT / "work_jobs4"
    parallel = command(
        [
            "run",
            *common,
            "--work-dir",
            str(parallel_work),
            "--output",
            str(parallel_output),
            "--jobs",
            "4",
            "--max-retries",
            "1",
        ]
    )
    timings["jobs4_full_seconds"] = round(time.perf_counter() - started, 3)
    parallel_summary = json_stdout(parallel)
    serial_pixels = pixel_digest(serial_output, 0, 239)
    parallel_pixels = pixel_digest(parallel_output, 0, 239)
    serial_files = file_digest(serial_output, 0, 239)
    parallel_files = file_digest(parallel_output, 0, 239)
    digests["jobs1_pixels"] = serial_pixels
    digests["jobs4_pixels"] = parallel_pixels
    digests["jobs1_png_bytes"] = serial_files
    digests["jobs4_png_bytes"] = parallel_files
    digests["jobs4_sequence"] = parallel_summary["assembly"]["sequence_digest"]
    checks["pixel_identical_jobs1_jobs4"] = serial_pixels == parallel_pixels
    checks["png_bytes_identical_jobs1_jobs4"] = serial_files == parallel_files

    resumed = command(
        [
            "render",
            *common,
            "--work-dir",
            str(parallel_work),
            "--jobs",
            "4",
            "--max-retries",
            "1",
        ]
    )
    resumed_summary = json_stdout(resumed)
    checks["normal_exit_resume_reuses_all"] = (
        resumed_summary["chunks_reused"] == 12
        and resumed_summary["chunks_rendered"] == 0
    )

    corrupted_chunk = chunk_id(100, 119)
    checkpoint = (
        parallel_work / "checkpoints" / f"{corrupted_chunk}.json"
    )
    record = json.loads(checkpoint.read_text(encoding="utf-8"))
    record["checkpoint_version"] = 999
    checkpoint.write_text(json.dumps(record), encoding="utf-8")
    repaired = command(
        [
            "run",
            *common,
            "--work-dir",
            str(parallel_work),
            "--output",
            str(parallel_output),
            "--jobs",
            "4",
            "--max-retries",
            "1",
        ]
    )
    repaired_summary = json_stdout(repaired)["render"]
    checks["corrupt_checkpoint_forces_one_rerender"] = (
        repaired_summary["chunks_rendered"] == 1
        and repaired_summary["rendered_chunks"] == [corrupted_chunk]
        and repaired_summary["invalid_checkpoints"][0]["reason"]
        == "checkpoint_field_mismatch:checkpoint_version"
    )
    checks["corrupt_checkpoint_output_still_matches"] = (
        pixel_digest(parallel_output, 0, 239) == serial_pixels
    )

    hard_target = chunk_id(20, 39)
    hard_work = RUN_ROOT / "work_hard"
    hard_sequence = RUN_ROOT / "hard_sequence"
    hard_first = command(
        [
            "render",
            *common,
            "--work-dir",
            str(hard_work),
            "--jobs",
            "4",
            "--max-retries",
            "0",
        ],
        expected=3,
        env={
            "OFR_TEST_FAULT": json.dumps(
                {
                    "chunk_id": hard_target,
                    "mode": "hard_exit",
                    "after_frames": 3,
                    "attempt": 1,
                    "exit_code": 23,
                }
            )
        },
    )
    hard_error = json_stderr(hard_first)
    forced_ids = {
        item["chunk_id"] for item in hard_error["forced_terminations"]
    }
    checks["hard_exit_reported_without_report"] = (
        hard_target in forced_ids
        and not (hard_work / "reports" / f"{hard_target}_attempt_001.json").exists()
    )
    hard_resume = command(
        [
            "render",
            *common,
            "--work-dir",
            str(hard_work),
            "--jobs",
            "4",
            "--max-retries",
            "1",
        ]
    )
    hard_resume_summary = json_stdout(hard_resume)
    checks["hard_exit_checkpoint_resume"] = (
        hard_resume_summary["chunks_reused"] == 11
        and hard_resume_summary["chunks_rendered"] == 1
        and hard_resume_summary["retry_counts"] == {hard_target: 1}
    )
    command(
        [
            "assemble",
            *common,
            "--work-dir",
            str(hard_work),
            "--output",
            str(hard_sequence),
        ]
    )
    checks["hard_exit_resumed_output_identical"] = (
        pixel_digest(hard_sequence, 0, 239) == serial_pixels
    )

    small_scene = RUN_ROOT / "scene_small.json"
    write_sample(
        small_scene,
        frame_count=8,
        width=28,
        height=18,
        samples_per_axis=4,
    )
    recoverable_work = RUN_ROOT / "work_recoverable"
    recoverable_target = chunk_id(0, 1)
    recoverable_first = command(
        [
            "render",
            "--input",
            str(small_scene),
            "--work-dir",
            str(recoverable_work),
            "--chunk-size",
            "2",
            "--jobs",
            "1",
            "--max-retries",
            "0",
        ],
        expected=3,
        env={
            "OFR_TEST_FAULT": json.dumps(
                {
                    "chunk_id": recoverable_target,
                    "mode": "recoverable",
                    "after_frames": 2,
                    "attempt": 1,
                    "reason": "acceptance synthetic recoverable failure",
                }
            )
        },
    )
    recoverable_error = json_stderr(recoverable_first)
    failure = recoverable_error["recoverable_failures"][0]
    report = json.loads(Path(failure["report"]).read_text(encoding="utf-8"))
    checks["recoverable_failure_report_written"] = (
        failure["chunk_id"] == recoverable_target
        and failure["completed_frames"] == [0, 1]
        and report["status"] == "failed"
        and report["recoverable"]
        and report["chunk_completed"] is False
    )

    missing_work = RUN_ROOT / "work_missing"
    shutil.copytree(parallel_work, missing_work)
    missing_frame = (
        missing_work
        / "chunks"
        / corrupted_chunk
        / "frames"
        / "frame_000100.png"
    )
    missing_frame.unlink()
    missing_output = RUN_ROOT / "missing_sequence"
    missing = command(
        [
            "assemble",
            *common,
            "--work-dir",
            str(missing_work),
            "--output",
            str(missing_output),
        ],
        expected=4,
    )
    missing_summary = json_stderr(missing)
    checks["missing_frame_nonzero_structured_json"] = (
        missing_summary["status"] == "error"
        and missing_summary["successful_frames"] == 220
        and missing_summary["missing_frames"] == list(range(100, 120))
        and missing_summary["corrupted_chunks"][0]["chunk_id"]
        == corrupted_chunk
    )
    checks["failed_assembly_leaves_no_output"] = not missing_output.exists()

    verification = command(["verify", "--output", str(serial_output)])
    verification_summary = json_stdout(verification)
    checks["final_manifest_verification"] = (
        verification_summary["ok"]
        and verification_summary["frame_count"] == 240
    )

    return {
        "status": "ok" if all(checks.values()) else "failed",
        "generated_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": numpy.__version__,
            "pillow": PIL.__version__,
            "platform": sys.platform,
        },
        "sample": {
            "path": str(sample_path.relative_to(ROOT)),
            "frames": 240,
            "width": 128,
            "height": 72,
            "samples_per_axis": 4,
            "chunk_size": 20,
        },
        "checks": checks,
        "digests": digests,
        "timings": timings,
        "hard_exit": {
            "chunk_id": hard_target,
            "first_run_exit_code": hard_first.returncode,
            "resumed_reused_chunks": hard_resume_summary["chunks_reused"],
            "resumed_rendered_chunks": hard_resume_summary["chunks_rendered"],
        },
        "corrupt_checkpoint": {
            "chunk_id": corrupted_chunk,
            "rendered_chunks": repaired_summary["rendered_chunks"],
        },
        "missing_frame_summary": missing_summary,
    }


if __name__ == "__main__":
    try:
        report = main()
    except BaseException as exc:
        report = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "ok" else 1)

