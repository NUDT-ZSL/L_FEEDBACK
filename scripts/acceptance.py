#!/usr/bin/env python3
"""Run deterministic local acceptance and record measured results."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dialogue_sync.srt import parse_srt_text  # noqa: E402
from dialogue_sync.state import inspect_state  # noqa: E402
from tests.fixtures import create_project  # noqa: E402


def run(command: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def ffprobe(path: Path) -> dict:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            (
                "stream=index,codec_name,codec_type,sample_rate,channels,"
                "bits_per_sample,bits_per_raw_sample:stream_tags=language,title"
            ),
            "-show_chapters",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def ebu_r128(path: Path) -> dict[str, float]:
    result = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(path),
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    summary = result.stderr[result.stderr.rfind("Summary:") :]
    integrated = re.search(r"I:\s+(-?\d+(?:\.\d+)?)\s+LUFS", summary)
    peak = re.search(r"Peak:\s+(-?\d+(?:\.\d+)?)\s+dBFS", summary)
    if not integrated or not peak:
        raise RuntimeError("unable to parse ebur128 summary")
    return {
        "integrated_lufs": float(integrated.group(1)),
        "true_peak_dbtp": float(peak.group(1)),
    }


def main() -> int:
    acceptance_root = REPO / "artifacts" / "acceptance"
    if acceptance_root.exists():
        shutil.rmtree(acceptance_root)
    acceptance_root.mkdir(parents=True)
    input_root = acceptance_root / "input"
    fixture = create_project(input_root)
    output = acceptance_root / "output"

    build = run(
        [
            sys.executable,
            "-m",
            "dialogue_sync",
            "build",
            "--timeline",
            str(fixture["timeline"]),
            "--output-dir",
            str(output),
            "--quiet",
        ]
    )
    if build.returncode != 0:
        print(build.stderr, file=sys.stderr)
        return 1
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    entries = {entry["index"]: entry for entry in report["entries"]}
    subtitle_document = parse_srt_text(
        (output / "dialogue_mix.srt").read_text(encoding="utf-8")
    )
    mkv_probe = ffprobe(output / "dialogue_mix.mkv")
    audio_probe = ffprobe(output / "dialogue_mix.wav")
    measured = ebu_r128(output / "dialogue_mix.wav")
    with wave.open(str(output / "dialogue_mix.wav"), "rb") as handle:
        wav_frames = handle.getnframes()
        wav_rate = handle.getframerate()
        wav_channels = handle.getnchannels()
        wav_width = handle.getsampwidth()

    audio_stream = [
        stream for stream in mkv_probe["streams"] if stream["codec_type"] == "audio"
    ][0]
    subtitle_stream = [
        stream for stream in mkv_probe["streams"] if stream["codec_type"] == "subtitle"
    ][0]
    chapters = mkv_probe.get("chapters", [])
    duration_error_ms = abs(float(mkv_probe["format"]["duration"]) - 12.0) * 1_000.0
    expected_subtitles_ms = {
        1: (1_000, 2_500),
        2: (4_000, 5_500),
        3: (7_000, 8_500),
        4: (9_500, 11_000),
        5: (4_500, 5_500),
    }

    checks = {
        "build_exit_zero": build.returncode == 0,
        "operations": (
            entries[1]["operation"] == "stretch"
            and entries[2]["operation"] == "crop"
            and entries[3]["operation"] == "pad_silence"
            and entries[4]["operation"] == "skip"
            and entries[5]["operation"] == "none"
        ),
        "range_skip": "E_RANGE" in entries[4]["error_codes"],
        "mix_clip_attenuated": (
            report["mix"]["clipping_detected"]
            and report["mix"]["mix_gain_db"] < 0
            and "E_CLIP" in entries[2]["error_codes"]
            and "E_CLIP" in entries[5]["error_codes"]
        ),
        "output_wav_48k_24bit_stereo": (
            wav_rate == 48_000
            and wav_width == 3
            and wav_channels == 2
            and wav_frames == 576_000
        ),
        "loudness_within_0_5_lu": abs(measured["integrated_lufs"] + 23.0) <= 0.5,
        "true_peak_at_most_minus_1": measured["true_peak_dbtp"] <= -1.0,
        "subtitles_shifted_and_ordered": all(
            (cue.start_ms, cue.end_ms) == expected_subtitles_ms[cue.index]
            for cue in subtitle_document.cues
        ),
        "subtitle_overlap_warning": (
            len(report["subtitles"]["overlaps"]) == 1
            and report["subtitles"]["overlaps"][0]["code"]
            == "W_SUBTITLE_OVERLAP"
        ),
        "mkv_languages": (
            audio_stream["tags"]["language"] == "eng"
            and subtitle_stream["tags"]["language"] == "eng"
        ),
        "mkv_chapters": (
            len(chapters) == 5
            and all(re.search(r"Dialogue \d+", chapter["tags"]["title"]) for chapter in chapters)
        ),
        "mkv_duration_error_at_most_50ms": duration_error_ms <= 50.0,
    }

    recovery_input = acceptance_root / "recovery_input"
    recovery_fixture = create_project(recovery_input)
    recovery_output = acceptance_root / "recovery_output"
    recovery_base = [
        sys.executable,
        "-m",
        "dialogue_sync",
        "build",
        "--timeline",
        str(recovery_fixture["timeline"]),
        "--output-dir",
        str(recovery_output),
    ]
    aborted = run([*recovery_base, "--simulate-abort-after", "2"])
    work_dir = recovery_output / ".dialogue_sync_work"
    state_path = work_dir / "state.json"
    state_before = state_path.read_bytes()
    interrupted_state = inspect_state(work_dir)
    snapshot = json.loads((work_dir / "report_snapshot.json").read_text(encoding="utf-8"))

    with state_path.open("ab") as handle:
        handle.write(b"\n{forced-termination-truncation")
    corrupt_attempt = run(recovery_base)
    with state_path.open("wb") as handle:
        handle.write(state_before)
    resumed = run(recovery_base)
    resumed_state = inspect_state(work_dir)

    checks.update(
        {
            "forced_exit_code_75": aborted.returncode == 75,
            "intermediate_state_valid": (
                interrupted_state["valid"]
                and interrupted_state["entry_reports"] == 2
                and len(interrupted_state["fragments"]) == 2
                and all(
                    fragment["checksum_ok"]
                    for fragment in interrupted_state["fragments"]
                )
                and len(snapshot["entries"]) == 2
            ),
            "corrupt_state_rejected": (
                corrupt_attempt.returncode == 2
                and "E_STATE_INTEGRITY" in corrupt_attempt.stderr
            ),
            "resume_skips_completed_entries": (
                resumed.returncode == 0
                and "[3/5]" in resumed.stdout
                and "[1/5]" not in resumed.stdout
            ),
            "resume_completes": (
                resumed_state["valid"]
                and resumed_state["phase"] == "complete"
                and (recovery_output / "report.json").is_file()
                and (recovery_output / "dialogue_mix.mkv").is_file()
            ),
        }
    )

    version = run([sys.executable, "--version"]).stdout.strip()
    ffmpeg_version = run(["ffmpeg", "-version"]).stdout.splitlines()[0]
    result = {
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "passed": all(checks.values()),
        "checks": checks,
        "environment": {
            "python": version,
            "ffmpeg": ffmpeg_version,
        },
        "measured": {
            "ebur128_integrated_lufs": measured["integrated_lufs"],
            "ebur128_true_peak_dbtp": measured["true_peak_dbtp"],
            "wav": {
                "sample_rate": wav_rate,
                "channels": wav_channels,
                "bits_per_sample": wav_width * 8,
                "frames": wav_frames,
                "duration_ms": wav_frames * 1_000.0 / wav_rate,
            },
            "mix": report["mix"],
            "loudness_report": report["loudness"],
            "entries": [
                {
                    "index": entry["index"],
                    "status": entry["status"],
                    "operation": entry["operation"],
                    "stretch_rate": entry["stretch_rate"],
                    "error_codes": entry["error_codes"],
                    "applied_gain_db": entry["applied_gain_db"],
                }
                for entry in report["entries"]
            ],
            "subtitles": [
                {
                    "index": cue.index,
                    "start_ms": cue.start_ms,
                    "end_ms": cue.end_ms,
                }
                for cue in subtitle_document.cues
            ],
            "overlaps": report["subtitles"]["overlaps"],
            "mkv": {
                "duration_seconds": float(mkv_probe["format"]["duration"]),
                "duration_error_ms": duration_error_ms,
                "audio_language": audio_stream["tags"]["language"],
                "subtitle_language": subtitle_stream["tags"]["language"],
                "audio_codec": audio_stream["codec_name"],
                "audio_bits_per_raw_sample": audio_stream.get("bits_per_raw_sample"),
                "chapters": [
                    {
                        "start_time": chapter["start_time"],
                        "end_time": chapter["end_time"],
                        "title": chapter["tags"]["title"],
                    }
                    for chapter in chapters
                ],
            },
            "recovery": {
                "abort_exit_code": aborted.returncode,
                "checkpoint_entries_after_abort": interrupted_state["entry_reports"],
                "checkpoint_fragments_after_abort": len(
                    interrupted_state["fragments"]
                ),
                "corrupt_state_exit_code": corrupt_attempt.returncode,
                "resume_exit_code": resumed.returncode,
                "resume_stdout": resumed.stdout.splitlines(),
                "final_phase": resumed_state["phase"],
            },
        },
        "artifacts": {
            "input": str(fixture["timeline"].parent),
            "output": str(output),
            "recovery_output": str(recovery_output),
        },
    }
    verification_path = REPO / "artifacts" / "verification.json"
    verification_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "checks": checks,
                "loudness": measured,
                "duration_error_ms": duration_error_ms,
                "verification": str(verification_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

