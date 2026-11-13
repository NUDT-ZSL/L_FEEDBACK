from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
VTT_TIMING = re.compile(
    r"^(\d{2,}):(\d{2}):(\d{2})\.(\d{3}) --> "
    r"(\d{2,}):(\d{2}):(\d{2})\.(\d{3})$"
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_cli(
    manifest: Path,
    *,
    ffmpeg: Path,
    ffprobe: Path,
    output_dir: Path | None = None,
    jobs: int = 2,
    no_overwrite: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "localizer",
        str(manifest),
        "--ffmpeg",
        os.fspath(ffmpeg),
        "--ffprobe",
        os.fspath(ffprobe),
        "--jobs",
        str(jobs),
    ]
    if output_dir is not None:
        command.extend(["--output-dir", str(output_dir)])
    if no_overwrite:
        command.append("--no-overwrite")
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def json_lines(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def probe(ffprobe: Path, path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            os.fspath(ffprobe),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            os.fspath(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    check(result.returncode == 0, f"ffprobe can read {path.name}")
    return json.loads(result.stdout)


def vtt_milliseconds(match: re.Match[str], offset: int) -> int:
    hours = int(match.group(offset))
    minutes = int(match.group(offset + 1))
    seconds = int(match.group(offset + 2))
    millis = int(match.group(offset + 3))
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def verify_normal_run(
    root: Path,
    output_dir: Path,
    ffmpeg: Path,
    ffprobe: Path,
    result: subprocess.CompletedProcess[str],
    *,
    expected_jobs: int,
    expect_overwrite_safe: bool = True,
) -> list[dict[str, Any]]:
    check(result.returncode == 0, "normal pipeline exits 0")
    check(result.stderr == "", "normal pipeline emits no stderr errors")
    events = json_lines(result.stdout)
    check(bool(events), "stdout contains JSON Lines")
    times = [event["monotonic_ns"] for event in events]
    elapsed = [event["elapsed_ms"] for event in events]
    check(times == sorted(times), "progress timestamps are monotonic")
    check(elapsed == sorted(elapsed), "progress elapsed times are monotonic")
    prepare = next(event for event in events if event.get("stage") == "prepare")
    check(
        prepare["jobs_effective"] == expected_jobs,
        "--jobs is capped at the CPU count",
    )
    changed = [
        difference
        for event in events
        if event.get("stage") == "normalize"
        for clip in event.get("changes", [])
        for difference in clip.get("differences", {})
    ]
    check(
        {"sample_rate", "channels", "bit_depth"}.issubset(set(changed)),
        "format differences are detected and normalized",
    )
    summary = events[-1]
    check(summary["event"] == "summary" and summary["ok"], "summary reports success")

    expected_names = {"acceptance.mp4"}
    for language in ("eng", "fre", "jpn"):
        expected_names.update(
            {
                f"acceptance.{language}.stereo.wav",
                f"acceptance.{language}.mono.wav",
                f"acceptance.{language}.vtt",
            }
        )
    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    check(actual == expected_names, "all expected artifacts are present")
    check(
        not list(output_dir.rglob("*.tmp")),
        "successful run leaves no .tmp files",
    )

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for language in ("eng", "fre", "jpn"):
        for version, channels in (("stereo", 2), ("mono", 1)):
            path = output_dir / f"acceptance.{language}.{version}.wav"
            info = sf.info(path)
            check(
                (info.samplerate, info.channels, info.subtype, info.frames)
                == (48000, channels, "PCM_16", 124800),
                f"{path.name} has the normalized format and timeline length",
            )
            samples, _ = sf.read(path, dtype="float64", always_2d=True)
            peak_dbfs = 20.0 * math.log10(
                max(float(np.max(np.abs(samples))), 1e-15)
            )
            check(
                abs(peak_dbfs - (-1.0)) <= 0.02,
                f"{path.name} is peak-normalized to -1 dBFS",
            )

    fra_samples, _ = sf.read(
        output_dir / "acceptance.fre.stereo.wav",
        dtype="float64",
        always_2d=True,
    )
    check(
        float(np.max(np.abs(fra_samples[:2400]))) == 0.0,
        "leading silence is preserved before the first French clip",
    )
    check(
        float(np.max(np.abs(fra_samples[-2400:]))) == 0.0,
        "trailing timeline silence is preserved",
    )

    vtt_path = output_dir / "acceptance.eng.vtt"
    raw_vtt = vtt_path.read_bytes()
    check(not raw_vtt.startswith(b"\xef\xbb\xbf"), "WebVTT has no UTF-8 BOM")
    check(b"\r" not in raw_vtt, "WebVTT uses normalized LF newlines")
    vtt = raw_vtt.decode("utf-8")
    check(vtt.startswith("WEBVTT\n\n"), "WebVTT header is valid")
    check("&amp;" in vtt and "&lt;now&gt;" in vtt, "WebVTT text is escaped")
    check("--&gt;" in vtt, "WebVTT timing marker in text is escaped")
    fre_vtt = (output_dir / "acceptance.fre.vtt").read_text(encoding="utf-8")
    check("\u00a0" in fre_vtt, "blank subtitle lines are preserved as NBSP lines")
    for language_index, language in enumerate(("eng", "fre", "jpn")):
        language_vtt = (output_dir / f"acceptance.{language}.vtt").read_text(
            encoding="utf-8"
        )
        timings = [
            (vtt_milliseconds(match, 1), vtt_milliseconds(match, 5))
            for match in (
                VTT_TIMING.match(line) for line in language_vtt.splitlines()
            )
            if match
        ]
        expected = [
            (clip["start_ms"], clip["start_ms"] + clip["duration_ms"])
            for clip in manifest["languages"][language_index]["clips"]
        ]
        check(
            timings == expected,
            f"{language} WebVTT cue times match manifest milliseconds",
        )

    metadata = probe(ffprobe, output_dir / "acceptance.mp4")
    streams = metadata["streams"]
    video = [stream for stream in streams if stream["codec_type"] == "video"]
    audio = [stream for stream in streams if stream["codec_type"] == "audio"]
    check(len(video) == 1 and len(audio) == 3, "MP4 has one video and three audio tracks")
    check(
        [stream["tags"]["language"] for stream in audio] == ["eng", "fre", "jpn"],
        "MP4 audio tracks use ISO 639-2/B language tags",
    )
    video_duration = float(video[0]["duration"])
    for stream in audio:
        check(
            abs(float(stream["duration"]) - video_duration) * 1000.0 <= 10.0,
            "each MP4 audio track is within 10ms of the video duration",
        )
        check(
            int(stream["channels"]) == 2 and int(stream["sample_rate"]) == 48000,
            "each MP4 audio track is 48kHz stereo",
        )
    mp4_bytes = (output_dir / "acceptance.mp4").read_bytes()
    check(
        0 <= mp4_bytes.find(b"moov") < mp4_bytes.find(b"mdat"),
        "MP4 moov atom precedes mdat for streaming",
    )

    frame = subprocess.run(
        [
            os.fspath(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            os.fspath(output_dir / "acceptance.mp4"),
            "-frames:v",
            "1",
            "-vf",
            "scale=2:2,format=gray",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        timeout=30,
    )
    check(
        frame.returncode == 0 and max(frame.stdout, default=0) <= 20,
        "MP4 placeholder video frame is black",
    )
    return events


def verify_timeline_error(ffmpeg: Path, ffprobe: Path, root: Path) -> None:
    result = run_cli(
        root / "manifest-timeline-error.json",
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
    )
    check(result.returncode == 2, "E_TIMELINE exits 2")
    errors = json_lines(result.stderr)
    check(
        len(errors) == 1 and errors[0]["code"] == "E_TIMELINE",
        "E_TIMELINE JSON error is written to stderr",
    )


def verify_media_skip(ffmpeg: Path, ffprobe: Path, root: Path) -> None:
    result = run_cli(
        root / "manifest-media-error.json",
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
    )
    check(result.returncode == 4, "E_MEDIA aggregate exits nonzero")
    stderr = json_lines(result.stderr)
    check(
        any(error.get("code") == "E_MEDIA" for error in stderr),
        "E_MEDIA JSON error is written to stderr",
    )
    summary = json_lines(result.stdout)[-1]
    check(
        summary["event"] == "summary"
        and not summary["ok"]
        and summary["failed_languages"][0]["language"] == "ger",
        "failure is summarized while successful languages continue",
    )
    output = root / "out-media" / "media.mp4"
    metadata = probe(ffprobe, output)
    check(
        len([s for s in metadata["streams"] if s["codec_type"] == "audio"]) == 1,
        "MP4 contains only languages that passed media checks",
    )


def verify_exists(ffmpeg: Path, ffprobe: Path, root: Path) -> None:
    result = run_cli(
        root / "manifest.json",
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        output_dir=root / "out-a",
        no_overwrite=True,
    )
    check(result.returncode == 3, "--no-overwrite exits 3")
    errors = json_lines(result.stderr)
    check(
        errors and errors[0]["code"] == "E_EXISTS",
        "--no-overwrite emits E_EXISTS",
    )


def verify_interrupt(ffmpeg: Path, ffprobe: Path, root: Path) -> None:
    output = root / "out-interrupt"
    shutil.rmtree(output, ignore_errors=True)
    command = [
        sys.executable,
        "-m",
        "localizer",
        str(root / "manifest-interrupt.json"),
        "--output-dir",
        str(output),
        "--jobs",
        "1",
        "--ffmpeg",
        os.fspath(ffmpeg),
        "--ffprobe",
        os.fspath(ffprobe),
    ]
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        interrupt_signal = signal.CTRL_BREAK_EVENT
    else:
        interrupt_signal = signal.SIGINT
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        **kwargs,
    )
    watchdog = threading.Timer(20.0, process.kill)
    watchdog.start()
    signal_sent_at: float | None = None
    first_language: str | None = None
    try:
        assert process.stdout is not None
        for line in process.stdout:
            event = json.loads(line)
            if signal_sent_at is None and event.get("stage") == "wav_stereo":
                first_language = event["language"]
                process.send_signal(interrupt_signal)
                signal_sent_at = time.monotonic()
        returncode = process.wait(timeout=5)
    finally:
        watchdog.cancel()
    stderr = process.stderr.read() if process.stderr is not None else ""
    check(signal_sent_at is not None and first_language is not None, "interrupt is sent mid-language")
    assert signal_sent_at is not None
    check(
        returncode == 130 and time.monotonic() - signal_sent_at <= 2.0,
        "Ctrl+C-equivalent exits 130 within two seconds",
    )
    check(
        any(error.get("code") == "E_INTERRUPTED" for error in json_lines(stderr)),
        "interrupt emits E_INTERRUPTED",
    )
    expected = {
        f"interrupt.{first_language}.stereo.wav",
        f"interrupt.{first_language}.mono.wav",
        f"interrupt.{first_language}.vtt",
    }
    actual = {path.name for path in output.iterdir() if path.is_file()}
    check(expected == actual, "current language outputs are atomically committed")
    check(not list(output.rglob("*.tmp")), "interrupt leaves no .tmp files")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run localizer acceptance checks")
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, required=True)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    root = ROOT / "build" / "acceptance"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    generate = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "make_samples.py"), "--root", str(root)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    check(generate.returncode == 0, "deterministic samples are generated")

    run_a = run_cli(
        root / "manifest.json",
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        output_dir=root / "out-a",
        jobs=3,
    )
    cpu_count = os.cpu_count() or 1
    verify_normal_run(
        root,
        root / "out-a",
        args.ffmpeg,
        args.ffprobe,
        run_a,
        expected_jobs=min(3, cpu_count),
    )

    run_b = run_cli(
        root / "manifest.json",
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        output_dir=root / "out-b",
        jobs=999,
    )
    verify_normal_run(
        root,
        root / "out-b",
        args.ffmpeg,
        args.ffprobe,
        run_b,
        expected_jobs=cpu_count,
    )
    names = sorted(path.name for path in (root / "out-a").iterdir() if path.is_file())
    check(
        all(
            sha256(root / "out-a" / name) == sha256(root / "out-b" / name)
            for name in names
        ),
        "WAV, VTT and MP4 outputs are byte-identical across runs and job counts",
    )
    overwrite = run_cli(
        root / "manifest.json",
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        output_dir=root / "out-a",
        jobs=2,
    )
    check(
        overwrite.returncode == 0 and overwrite.stderr == "",
        "existing outputs are overwritten by default",
    )
    check(
        all(
            sha256(root / "out-a" / name) == sha256(root / "out-b" / name)
            for name in names
        ),
        "overwriting preserves byte-identical outputs",
    )

    verify_timeline_error(args.ffmpeg, args.ffprobe, root)
    verify_media_skip(args.ffmpeg, args.ffprobe, root)
    verify_exists(args.ffmpeg, args.ffprobe, root)
    sys.path.insert(0, str(ROOT))
    from localizer.errors import LocalizerError
    from localizer.ffmpeg import FFmpegRunner

    runner = FFmpegRunner(args.ffmpeg, args.ffprobe, threading.Event())
    try:
        runner.validate_mux(
            root / "out-a" / "acceptance.mp4",
            ["eng", "fre", "jpn"],
            expected_duration_ms=2700,
            expected_sample_rate=48000,
        )
    except LocalizerError as exc:
        check(exc.code == "E_MUX", "mux duration validation emits E_MUX")
    else:
        check(False, "mux duration validation emits E_MUX")
    verify_interrupt(args.ffmpeg, args.ffprobe, root)

    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)
    print("ALL ACCEPTANCE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
