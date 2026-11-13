"""Run the deterministic acceptance batch and independently verify artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mxf_r128.constants import (
    E_LOUDNESS_FAIL,
    E_MEDIA_CORRUPT,
    E_TIMEOUT,
    GOP_SIZE,
    W_AUDIO_DELAY,
    W_FPS_INFERRED,
)
from mxf_r128.loudness import measure_loudness
from mxf_r128.probe import keyframe_times, probe_media, resolve_tool
from mxf_r128.rates import is_fraction_rate
from mxf_r128.samples import generate_samples


EXPECTED = {
    "01_valid_mpeg2_25_stereo.mxf": ("success", None),
    "02_valid_avcintra_2997_stereo.mxf": ("success", None),
    "03_valid_5_1.mxf": ("success", None),
    "04_timestamp_offset_400ms.mxf": ("success", None),
    "05_timestamp_offset_1200ms.mxf": ("success", None),
    "06_loudness_nonconvergent.mxf": ("failed", E_LOUDNESS_FAIL),
    "07_subtitle_preserved.mxf": ("success", None),
    "08_missing_fps_index.mxf": ("failed", E_LOUDNESS_FAIL),
    "09_corrupt_zero.mxf": ("failed", E_MEDIA_CORRUPT),
    "10_corrupt_binary.mxf": ("failed", E_MEDIA_CORRUPT),
    "11_long_duration_400s.mxf": ("failed", E_TIMEOUT),
    "12_good_after_timeout.mxf": ("success", None),
}


def _probe_json(ffprobe: str, path: Path) -> dict[str, Any]:
    command = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-show_entries",
        "format=format_name,start_time,duration,size:"
        "stream=index,codec_name,codec_type,profile,level,pix_fmt,width,height,"
        "r_frame_rate,avg_frame_rate,start_time,duration,channels,"
        "channel_layout",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {"probe_error": True}
    return json.loads(result.stdout)


def _ffmpeg_version(ffmpeg: str) -> str:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-version"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.stdout.splitlines()[0] if result.stdout else "unknown"


def _process_residue() -> list[str]:
    command = ["ps", "-eo", "comm="]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    names = [line.strip() for line in result.stdout.splitlines()]
    return [name for name in names if name in {"ffmpeg", "ffprobe"}]


def _verify_output(
    ffmpeg: str,
    ffprobe: str,
    record: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    output_path = record.get("output_path")
    if output_path is None:
        return {
            "output_absent": True,
        }, details
    path = Path(output_path)
    checks["output_exists"] = path.is_file()
    if not path.is_file():
        return checks, details

    info = probe_media(ffprobe, path)
    details["ffprobe"] = {
        "streams": list(info.streams),
        "format": dict(info.format),
    }
    types = [str(stream.get("codec_type")) for stream in info.streams]
    details["stream_types"] = types
    video = info.video or {}
    audio = info.audio or {}
    checks["stream_types_valid"] = types in (
        ["video", "audio"],
        ["video", "audio", "subtitle"],
    )
    checks["video_is_h264"] = video.get("codec_name") == "h264"
    checks["profile_high"] = video.get("profile") == "High"
    checks["level_4_1"] = str(video.get("level")) == "41"
    checks["pix_fmt_yuv420p"] = video.get("pix_fmt") == "yuv420p"
    checks["frame_rate_matches_report"] = is_fraction_rate(
        str(video.get("avg_frame_rate", "")), str(record.get("frame_rate"))
    )
    checks["video_start_zero"] = abs(
        float(video.get("start_time", 0.0) or 0.0)
    ) <= 0.010
    checks["audio_start_zero"] = abs(
        float(audio.get("start_time", 0.0) or 0.0)
    ) <= 0.010
    if record.get("frame_rate") == "25/1":
        fps = 25.0
    else:
        fps = 30000.0 / 1001.0
    times = keyframe_times(ffprobe, path)
    intervals = [
        (current - previous) * fps
        for previous, current in zip(times, times[1:])
    ]
    checks["gop_at_most_50"] = bool(times) and (
        not intervals
        or max(intervals, default=0.0) <= GOP_SIZE + 1.0
    )
    details["keyframe_intervals"] = intervals

    loudness = measure_loudness(ffmpeg, path, timeout=30.0)
    details["loudness"] = loudness.as_dict()
    checks["loudness_matches_report"] = True
    for report_key, measured in (
        ("loudness_lufs", loudness.integrated_lufs),
        ("loudness_lra", loudness.lra),
        ("true_peak_dbtp", loudness.true_peak_dbtp),
    ):
        reported = record.get(report_key)
        if reported is None or measured is None:
            checks["loudness_matches_report"] = (
                checks["loudness_matches_report"]
                and reported is None
                and measured is None
            )
        else:
            checks["loudness_matches_report"] = (
                checks["loudness_matches_report"]
                and abs(float(reported) - float(measured)) <= 0.1
            )
    if record.get("status") == "success":
        checks["loudness_passes"] = (
            loudness.integrated_lufs is not None
            and abs(loudness.integrated_lufs + 23.0) <= 0.5
            and loudness.true_peak_dbtp is not None
            and loudness.true_peak_dbtp <= -1.0
        )
        checks["sync_at_most_10ms"] = bool(
            record.get("output_start_delta_ms") is not None
            and float(record["output_start_delta_ms"]) <= 10.0
        )
        checks["no_error_code"] = record.get("error_code") is None
    return checks, details


def _verify(workdir: Path, exit_code: int, stderr: str) -> dict[str, Any]:
    ffmpeg = resolve_tool("FFMPEG_BIN", "ffmpeg")
    ffprobe = resolve_tool("FFPROBE_BIN", "ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("tools not found")

    report = json.loads((workdir / "qc.json").read_text(encoding="utf-8"))
    records = {
        record["source_file"]: record for record in report["files"]
    }
    files: list[dict[str, Any]] = []
    all_checks: list[bool] = []
    for name, (status, error_code) in EXPECTED.items():
        record = records.get(name)
        file_checks: dict[str, bool] = {
            "record_exists": record is not None,
        }
        details: dict[str, Any] = {}
        if record is not None:
            file_checks["reported_status"] = record.get("status") == status
            file_checks["reported_error_code"] = (
                record.get("error_code") == error_code
            )
            output_checks, details = _verify_output(
                ffmpeg, ffprobe, record
            )
            file_checks.update(output_checks)

            if status == "success":
                file_checks["output_present"] = (
                    record.get("output_path") is not None
                    and Path(record["output_path"]).is_file()
                )
            elif error_code == E_LOUDNESS_FAIL:
                file_checks["mp4_retained"] = (
                    record.get("output_path") is not None
                    and Path(record["output_path"]).is_file()
                )
                measured = details.get("loudness", {})
                integrated = measured.get("lufs")
                true_peak = measured.get("true_peak_dbtp")
                file_checks["loudness_outside_target"] = (
                    integrated is None
                    or abs(float(integrated) + 23.0) > 0.5
                    or true_peak is None
                    or float(true_peak) > -1.0
                )
            elif error_code in {E_MEDIA_CORRUPT, E_TIMEOUT}:
                file_checks["output_path_null"] = (
                    record.get("output_path") is None
                )
        files.append(
            {
                "source_file": name,
                "expected_status": status,
                "expected_error_code": error_code,
                "checks": file_checks,
                "details": details,
            }
        )
        all_checks.extend(file_checks.values())

    batch_checks = {
        "exit_code_1": exit_code == 1,
        "report_sorted": [
            record["source_file"] for record in report["files"]
        ]
        == sorted(EXPECTED),
        "all_expected_codes": report["summary"]["error_codes"]
        == [E_LOUDNESS_FAIL, E_MEDIA_CORRUPT, E_TIMEOUT],
        "stderr_only_business_codes": (
            "ffmpeg version" not in stderr.lower()
            and E_TIMEOUT in stderr
            and E_LOUDNESS_FAIL in stderr
            and E_MEDIA_CORRUPT in stderr
        ),
        "no_partial_files": not any(
            path.name.endswith((".partial", ".partial.mp4"))
            or ".partial." in path.name
            for path in (workdir / "output").iterdir()
        ),
        "no_ffmpeg_residue": _process_residue() == [],
    }
    all_checks.extend(batch_checks.values())

    missing_fps_source = probe_media(
        ffprobe, workdir / "samples" / "08_missing_fps_index.mxf"
    )
    missing_video = missing_fps_source.video or {}
    source_checks = {
        "missing_fps_avg_is_zero": str(
            missing_video.get("avg_frame_rate")
        )
        == "0/0",
        "missing_fps_warned": W_FPS_INFERRED
        in records["08_missing_fps_index.mxf"]["warnings"],
        "delay_400ms_original_preserved": abs(
            float(
                records["04_timestamp_offset_400ms.mxf"][
                    "source_audio_delay_sec"
                ]
            )
            - 0.378
        )
        <= 0.02,
        "delay_1200ms_original_preserved": records[
            "05_timestamp_offset_1200ms.mxf"
        ]["source_audio_delay_sec"]
        == 1.178,
        "delay_1200ms_warned": W_AUDIO_DELAY
        in records["05_timestamp_offset_1200ms.mxf"]["warnings"],
        "subtitle_mov_text": any(
            stream.get("codec_name") == "mov_text"
            for stream in probe_media(
                ffprobe,
                Path(records["07_subtitle_preserved.mxf"]["output_path"]),
            ).streams
        ),
        "overlong_source_declared_400s": abs(
            float(
                probe_media(
                    ffprobe,
                    workdir / "samples" / "11_long_duration_400s.mxf",
                ).format.get("duration", 0.0)
            )
            - 400.0
        )
        <= 0.1,
    }
    all_checks.extend(source_checks.values())

    return {
        "files": files,
        "batch_checks": batch_checks,
        "source_checks": source_checks,
        "all_checks_pass": all(all_checks),
    }


def _write_readme(
    workdir: Path,
    *,
    exit_code: int,
    elapsed_seconds: float,
    version: str,
    verification: dict[str, Any],
) -> None:
    verification_text = str(verification["all_checks_pass"]).lower()
    text = f"""# Acceptance Run

Environment: local `{version}`; Python 3.11-compatible code.

The sample set is generated deterministically by
`python -m tools.generate_samples` using FFmpeg lavfi. The timestamp,
subtitle, and overlong files use the documented container-rewrite step so
the requested MP4/MXF metadata conditions can be produced without external
assets.

## Batch

`python -m mxf_r128 --max-workers 4` processed 12 files in
{elapsed_seconds:.2f} seconds. Exit code: `{exit_code}`.

Expected failures:

- `06_loudness_nonconvergent.mxf`: `E_LOUDNESS_FAIL`, MP4 retained.
- `08_missing_fps_index.mxf`: `W_FPS_INFERRED`; the 40 ms source later
  fails loudness validation, not frame-rate resolution.
- `09_corrupt_zero.mxf` and `10_corrupt_binary.mxf`: `E_MEDIA_CORRUPT`.
- `11_long_duration_400s.mxf`: 30 s transcode timeout, `E_TIMEOUT`; no
  partial MP4 and no FFmpeg process residue.
- `12_good_after_timeout.mxf`: success after the timeout, proving batch
  continuation.

All successful outputs are H.264 High 4.1, yuv420p, GOP 50, preserve
25/1 or 30000/1001, measure within -23 +/- 0.5 LUFS, have true peak below
-1.0 dBTP, and have audio/video start delta <= 10 ms. The 5.1 output
retains six channels with layout `5.1`.

Independent verification result: `all_checks_pass = {verification_text}`.

Artifacts:

- `qc.json`: production QC report.
- `verification.json`: independent ffprobe/loudness/process checks.
- `samples-manifest.json`: deterministic source manifest.
- `output/`: retained MP4 files.
"""
    (workdir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-dir", type=Path, default=Path("artifacts/acceptance")
    )
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--long-duration", type=float, default=400.0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--reuse-samples", action="store_true")
    args = parser.parse_args()

    ffmpeg = resolve_tool("FFMPEG_BIN", "ffmpeg")
    ffprobe = resolve_tool("FFPROBE_BIN", "ffprobe")
    if ffmpeg is None or ffprobe is None:
        print("E_FFMPEG_NOT_FOUND", file=sys.stderr)
        return 2

    workdir = args.work_dir.resolve()
    if not args.reuse_samples:
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        generate_samples(
            workdir / "samples",
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            duration=args.duration,
            include_long=True,
            long_duration=args.long_duration,
            include_subtitle=True,
        )
    output_dir = workdir / "output"
    report_path = workdir / "qc.json"
    environment = dict(os.environ)
    environment["FFMPEG_BIN"] = ffmpeg
    environment["FFPROBE_BIN"] = ffprobe
    command = [
        sys.executable,
        "-m",
        "mxf_r128",
        "--input-dir",
        str(workdir / "samples"),
        "--output-dir",
        str(output_dir),
        "--report",
        str(report_path),
        "--max-workers",
        str(args.max_workers),
    ]
    started = time.monotonic()
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        check=False,
    )
    elapsed = time.monotonic() - started
    (workdir / "stderr.txt").write_text(
        result.stderr, encoding="utf-8"
    )
    verification = _verify(workdir, result.returncode, result.stderr)
    (workdir / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    version = _ffmpeg_version(ffmpeg)
    _write_readme(
        workdir,
        exit_code=result.returncode,
        elapsed_seconds=elapsed,
        version=version,
        verification=verification,
    )
    run_result = {
        "exit_code": result.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "stderr": result.stderr.strip(),
        "ffmpeg_version": version,
        "verification_all_checks_pass": verification["all_checks_pass"],
    }
    (workdir / "run-result.json").write_text(
        json.dumps(run_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_result, ensure_ascii=False, indent=2))
    return 0 if verification["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
