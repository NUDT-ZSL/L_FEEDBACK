"""Command-line entry point and batch orchestration."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import soundfile as sf

from .config import (
    LOUDNESS_REPORT_FILENAME,
    MIX_FILENAME,
    SAMPLE_RATE_HZ,
    SUMMARY_FILENAME,
    SWITCH_REPORT_FILENAME,
)
from .errors import ParameterError, ProcessingError, SceneValidationError
from .loudness import analyze_loudness, write_json
from .mixing import mix_scene
from .reports import build_switch_report
from .scene import SceneSpec, load_scene


USAGE = """usage: python main.py INPUT_DIR OUTPUT_DIR

Mix all cinematic scenes in INPUT_DIR into OUTPUT_DIR.

INPUT_DIR may be either:
  * one scene directory containing scene.json and its WAV assets, or
  * a batch directory whose immediate subdirectories each contain scene.json.

Outputs per scene:
  mix.wav                 16-bit PCM stereo at 48000 Hz
  loudness_report.json    integrated LUFS, true peak, dialogue short-term maxima
  switch_check.json       camera-cut crossfade diagnostics and ducking checks

Batch mode also writes summary.json.

Exit codes:
  0  success (also printed usage when no arguments are supplied)
  1  invalid command-line parameters or input/output directory
  2  invalid scene JSON, missing field, invalid time, or invalid asset
  3  one or more scenes failed in batch mode
  4  processing failed in single-scene mode
"""

_REPORT_FILENAMES = {
    LOUDNESS_REPORT_FILENAME,
    SWITCH_REPORT_FILENAME,
    SUMMARY_FILENAME,
}


@dataclass(frozen=True)
class SceneJob:
    scene_dir: Path
    scene_json: Path
    output_dir: Path
    validation_error: Optional[str] = None


def _json_candidates(directory: Path) -> List[Path]:
    try:
        return sorted(
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".json"
            and path.name not in _REPORT_FILENAMES
        )
    except OSError:
        return []


def _find_scene_json(directory: Path) -> Path:
    preferred = directory / "scene.json"
    if preferred.is_file():
        return preferred
    candidates = _json_candidates(directory)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SceneValidationError(f"no scene JSON found in {directory}")
    names = ", ".join(path.name for path in candidates)
    raise SceneValidationError(
        f"multiple JSON files found in {directory}; expected scene.json or one JSON: {names}"
    )


def _discover_jobs(input_dir: Path, output_dir: Path) -> Tuple[List[SceneJob], bool]:
    if not input_dir.is_dir():
        raise ParameterError(f"input directory does not exist: {input_dir}")
    try:
        root_json = _json_candidates(input_dir)
        subdirectories = sorted(
            path for path in input_dir.iterdir() if path.is_dir()
        )
    except OSError as exc:
        raise ParameterError(f"cannot read input directory {input_dir}: {exc}") from exc
    preferred_root = input_dir / "scene.json"
    if preferred_root.is_file():
        root_json = [preferred_root]
    if len(root_json) > 1:
        names = ", ".join(path.name for path in root_json)
        raise ParameterError(
            f"input directory contains multiple scene JSON files: {names}"
        )
    if len(root_json) == 1:
        return (
            [SceneJob(input_dir, root_json[0], output_dir)],
            False,
        )

    jobs: List[SceneJob] = []
    output_resolved = output_dir.resolve()
    for subdirectory in subdirectories:
        if subdirectory.name.startswith("."):
            continue
        if subdirectory.resolve() == output_resolved:
            continue
        try:
            scene_json = _find_scene_json(subdirectory)
            validation_error = None
        except SceneValidationError as exc:
            # Keep the failed scene visible in the batch summary.
            scene_json = subdirectory / "scene.json"
            validation_error = str(exc)
        jobs.append(
            SceneJob(
                scene_dir=subdirectory,
                scene_json=scene_json,
                output_dir=output_dir / subdirectory.name,
                validation_error=validation_error,
            )
        )
    if not jobs:
        raise ParameterError(
            f"no scene JSON found in {input_dir} or its immediate subdirectories"
        )
    return jobs, True


def _run_scene(job: SceneJob, sample_rate: int) -> Dict[str, object]:
    if job.validation_error is not None:
        raise SceneValidationError(job.validation_error)
    scene: SceneSpec = load_scene(job.scene_json)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    mix_path = job.output_dir / MIX_FILENAME
    mix_result = mix_scene(scene, mix_path, sample_rate)

    switch_report = build_switch_report(scene, mix_result, sample_rate)
    switch_path = job.output_dir / SWITCH_REPORT_FILENAME
    write_json(switch_path, switch_report)

    # Read the finalized PCM16 file so the report describes exactly what was written.
    audio, actual_rate = sf.read(
        str(mix_path), dtype="float64", always_2d=True
    )
    if actual_rate != sample_rate:
        raise ProcessingError(
            f"output sample rate mismatch: expected {sample_rate}, got {actual_rate}"
        )
    if audio.shape[1] != 2:
        raise ProcessingError(
            f"output channel mismatch: expected 2, got {audio.shape[1]}"
        )
    loudness_report = analyze_loudness(
        mix_path, audio, sample_rate, scene.dialogue_intervals
    )
    loudness_report["scene_id"] = scene.scene_id
    loudness_report["source_scene"] = scene.path.name
    loudness_path = job.output_dir / LOUDNESS_REPORT_FILENAME
    write_json(loudness_path, loudness_report)

    return {
        "scene_id": scene.scene_id,
        "mix": str(mix_path),
        "loudness_report": str(loudness_path),
        "switch_check": str(switch_path),
        "total_samples": mix_result.total_samples,
        "peak_abs": mix_result.peak_abs,
        "clipped_samples": mix_result.clipped_samples,
        "cuts_passed": switch_report["all_cuts_passed"],
    }


def _validate_output_path(input_dir: Path, output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise ParameterError(f"output path is not a directory: {output_dir}")
    try:
        if input_dir.resolve() == output_dir.resolve():
            raise ParameterError("input and output directories must differ")
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ParameterError(f"cannot create output directory {output_dir}: {exc}") from exc


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _run_batch(
    jobs: Sequence[SceneJob], output_dir: Path, sample_rate: int
) -> int:
    results: List[Dict[str, object]] = []
    succeeded = 0
    failed = 0
    for job in jobs:
        try:
            result = _run_scene(job, sample_rate)
            succeeded += 1
            results.append(
                {
                    "scene_directory": str(job.scene_dir),
                    "scene_json": job.scene_json.name,
                    "status": "success",
                    "result": result,
                }
            )
            print(f"[ok]   {job.scene_dir.name}")
        except SceneValidationError as exc:
            failed += 1
            results.append(
                {
                    "scene_directory": str(job.scene_dir),
                    "scene_json": job.scene_json.name,
                    "status": "failed",
                    "error_type": "validation",
                    "error": _error_text(exc),
                }
            )
            print(f"[fail] {job.scene_dir.name}: {exc}", file=sys.stderr)
        except Exception as exc:
            failed += 1
            results.append(
                {
                    "scene_directory": str(job.scene_dir),
                    "scene_json": job.scene_json.name,
                    "status": "failed",
                    "error_type": "processing",
                    "error": _error_text(exc),
                }
            )
            print(f"[fail] {job.scene_dir.name}: {exc}", file=sys.stderr)

    exit_code = 0 if failed == 0 else 3
    summary = {
        "mode": "batch",
        "input_directory": str(jobs[0].scene_dir.parent) if jobs else "",
        "output_directory": str(output_dir),
        "sample_rate_hz": sample_rate,
        "total_scenes": len(jobs),
        "succeeded": succeeded,
        "failed": failed,
        "exit_code": exit_code,
        "scenes": results,
    }
    try:
        write_json(output_dir / SUMMARY_FILENAME, summary)
    except ProcessingError as exc:
        print(f"error: cannot write batch summary: {exc}", file=sys.stderr)
        return 3
    return exit_code


def run(input_dir: Path, output_dir: Path, sample_rate: int = SAMPLE_RATE_HZ) -> int:
    try:
        jobs, batch_mode = _discover_jobs(input_dir, output_dir)
        _validate_output_path(input_dir, output_dir)
    except ParameterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if batch_mode:
        return _run_batch(jobs, output_dir, sample_rate)

    job = jobs[0]
    try:
        result = _run_scene(job, sample_rate)
    except SceneValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {_error_text(exc)}", file=sys.stderr)
        return 4
    print(
        f"mixed {result['total_samples']} samples at {sample_rate} Hz -> "
        f"{result['mix']}"
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(USAGE, end="")
        return 0
    if len(args) == 1 and args[0] in ("-h", "--help"):
        print(USAGE, end="")
        return 0
    if len(args) != 2 or args[0].startswith("-") or args[1].startswith("-"):
        print(
            "error: expected exactly INPUT_DIR and OUTPUT_DIR "
            "(use --help for usage)",
            file=sys.stderr,
        )
        return 1
    return run(Path(args[0]).expanduser(), Path(args[1]).expanduser())
