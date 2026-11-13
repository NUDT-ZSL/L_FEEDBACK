"""Deterministic acceptance samples built with FFmpeg lavfi and remuxing."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SampleSet:
    directory: Path
    files: tuple[Path, ...]


def _run(ffmpeg: str, args: list[str]) -> None:
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-loglevel",
        "error",
        "-y",
        *args,
    ]
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("sample generation command failed")


def _stereo_mxf(
    ffmpeg: str,
    target: Path,
    *,
    duration: float,
    fps: str,
    codec: str,
) -> None:
    audio_duration = f"{duration:.6f}"
    args = [
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=640x360:rate={fps}:duration={duration:.6f}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:sample_rate=48000:duration={audio_duration}",
        "-filter_complex",
        "[1:a]volume=0.1,pan=stereo|c0=c0|c1=c0[a]",
        "-map",
        "0:v:0",
        "-map",
        "[a]",
    ]
    if codec == "avcintra":
        args.extend(
            [
                "-c:v",
                "libx264",
                "-profile:v",
                "high422",
                "-pix_fmt",
                "yuv422p10le",
                "-g",
                "1",
                "-bf",
                "0",
                "-crf",
                "18",
            ]
        )
    else:
        args.extend(["-c:v", "mpeg2video", "-q:v", "5"])
    args.extend(["-c:a", "pcm_s24le", "-f", "mxf", str(target)])
    _run(ffmpeg, args)


def _surround_mxf(ffmpeg: str, target: Path, *, duration: float) -> None:
    frequencies = (220, 330, 440, 110, 660, 770)
    args: list[str] = []
    for frequency in frequencies:
        args.extend(
            [
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={frequency}:sample_rate=48000:"
                f"duration={duration:.6f}",
            ]
        )
    labels = "".join(f"[a{index}]" for index in range(6))
    filters = ";".join(
        f"[{index}:a]volume=0.08,aformat=channel_layouts=mono[a{index}]"
        for index in range(6)
    )
    filters += (
        f";{labels}amerge=inputs=6,"
        "pan=5.1|FL=c0|FR=c1|FC=c2|LFE=c3|BL=c4|BR=c5[a]"
    )
    args.extend(
        [
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=640x360:rate=25:duration={duration:.6f}",
            "-filter_complex",
            filters,
            "-map",
            "6:v:0",
            "-map",
            "[a]",
            "-c:v",
            "mpeg2video",
            "-q:v",
            "5",
            "-c:a",
            "pcm_s24le",
            "-f",
            "mxf",
            str(target),
        ]
    )
    _run(ffmpeg, args)


def _transient_mxf(ffmpeg: str, target: Path, *, duration: float) -> None:
    expression = (
        "if(lt(mod(t\\,6)\\,0.001)\\,"
        "0.9*sin(2*PI*1000*t)\\,"
        "0.0001*sin(2*PI*100*t))"
    )
    _run(
        ffmpeg,
        [
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=640x360:rate=25:duration={duration:.6f}",
            "-f",
            "lavfi",
            "-i",
            f"aevalsrc={expression}:s=48000:d={duration:.6f}",
            "-filter_complex",
            "[1:a]pan=stereo|c0=c0|c1=c0[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "mpeg2video",
            "-q:v",
            "5",
            "-c:a",
            "pcm_s24le",
            "-f",
            "mxf",
            str(target),
        ],
    )


def _timestamp_offset_rewrite(
    ffmpeg: str,
    target: Path,
    *,
    duration: float,
    delay: float,
    fps: str = "25",
) -> None:
    audio_duration = max(duration - delay, 0.1)
    _run(
        ffmpeg,
        [
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=640x360:rate={fps}:duration={duration:.6f}",
            "-itsoffset",
            f"{delay:.6f}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=48000:"
            f"duration={audio_duration:.6f}",
            "-filter_complex",
            "[1:a]volume=0.1,pan=stereo|c0=c0|c1=c0[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-copyts",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "50",
            "-bf",
            "2",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-f",
            "mp4",
            str(target),
        ],
    )


def _subtitle_rewrite(
    ffmpeg: str, target: Path, *, duration: float, temporary_dir: Path
) -> None:
    subtitle = temporary_dir / ".sample-subtitle.srt"
    subtitle.write_text(
        "1\n00:00:00,300 --> 00:00:01,200\nSubtitle stream preserved\n",
        encoding="utf-8",
    )
    try:
        _run(
            ffmpeg,
            [
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=size=640x360:rate=25:duration={duration:.6f}",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:sample_rate=48000:"
                f"duration={duration:.6f}",
                "-i",
                str(subtitle),
                "-filter_complex",
                "[1:a]volume=0.1,pan=stereo|c0=c0|c1=c0[a]",
                "-map",
                "0:v:0",
                "-map",
                "[a]",
                "-map",
                "2:s:0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-g",
                "50",
                "-bf",
                "2",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-c:s",
                "mov_text",
                "-f",
                "mp4",
                str(target),
            ],
        )
    finally:
        subtitle.unlink(missing_ok=True)


def _missing_fps_mxf(ffmpeg: str, target: Path) -> None:
    _run(
        ffmpeg,
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=25:duration=0.04",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.04",
            "-filter_complex",
            "[1:a]volume=0.1,pan=stereo|c0=c0|c1=c0[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-frames:v",
            "1",
            "-c:v",
            "mpeg2video",
            "-q:v",
            "5",
            "-c:a",
            "pcm_s24le",
            "-f",
            "mxf",
            str(target),
        ],
    )


def _long_duration_rewrite(
    ffmpeg: str,
    target: Path,
    *,
    duration: float,
) -> None:
    _run(
        ffmpeg,
        [
            "-f",
            "lavfi",
            "-i",
            f"color=black:size=1920x1080:rate=25:duration={duration:.6f}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=48000:duration={duration:.6f}",
            "-filter_complex",
            "[1:a]volume=0.1,pan=stereo|c0=c0|c1=c0[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "50",
            "-crf",
            "51",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-f",
            "mp4",
            str(target),
        ],
    )


def _write_corrupt_files(directory: Path) -> list[Path]:
    zero_path = directory / "09_corrupt_zero.mxf"
    zero_path.touch()
    binary_path = directory / "10_corrupt_binary.mxf"
    binary_path.write_bytes((b"NOT_AN_MXF_CONTAINER\n" * 200)[:4096])
    return [zero_path, binary_path]


def _probe_summary(ffprobe: str, path: Path) -> dict[str, object]:
    command = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-show_entries",
        "format=format_name,start_time,duration:"
        "stream=index,codec_name,codec_type,width,height,r_frame_rate,"
        "avg_frame_rate,duration,start_time,channels,channel_layout",
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
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"probe_error": True}


def _write_manifest(
    directory: Path, files: Iterable[Path], ffprobe: str
) -> Path:
    entries = []
    for path in files:
        entries.append(
            {
                "source_file": path.name,
                "size": path.stat().st_size,
                "ffprobe": _probe_summary(ffprobe, path),
            }
        )
    manifest = {
        "generator": "mxf_r128.samples",
        "samples": entries,
    }
    manifest_path = directory / "samples-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def generate_samples(
    directory: Path,
    *,
    ffmpeg: str,
    ffprobe: str,
    duration: float = 2.0,
    include_long: bool = False,
    long_duration: float = 400.0,
    include_subtitle: bool = False,
    write_manifest: bool = True,
) -> SampleSet:
    directory.mkdir(parents=True, exist_ok=True)
    for old_file in directory.glob("*.mxf"):
        old_file.unlink(missing_ok=True)
    (directory / "samples-manifest.json").unlink(missing_ok=True)
    files: list[Path] = []

    sample = directory / "01_valid_mpeg2_25_stereo.mxf"
    _stereo_mxf(
        ffmpeg,
        sample,
        duration=duration,
        fps="25",
        codec="mpeg2",
    )
    files.append(sample)

    sample = directory / "02_valid_avcintra_2997_stereo.mxf"
    _stereo_mxf(
        ffmpeg,
        sample,
        duration=duration,
        fps="30000/1001",
        codec="avcintra",
    )
    files.append(sample)

    sample = directory / "03_valid_5_1.mxf"
    _surround_mxf(ffmpeg, sample, duration=duration)
    files.append(sample)

    sample = directory / "04_timestamp_offset_400ms.mxf"
    _timestamp_offset_rewrite(
        ffmpeg, sample, duration=duration, delay=0.4
    )
    files.append(sample)

    sample = directory / "05_timestamp_offset_1200ms.mxf"
    _timestamp_offset_rewrite(
        ffmpeg, sample, duration=duration + 0.8, delay=1.2
    )
    files.append(sample)

    sample = directory / "06_loudness_nonconvergent.mxf"
    _transient_mxf(ffmpeg, sample, duration=max(duration, 6.0))
    files.append(sample)

    if include_subtitle:
        sample = directory / "07_subtitle_preserved.mxf"
        _subtitle_rewrite(
            ffmpeg,
            sample,
            duration=duration,
            temporary_dir=directory,
        )
        files.append(sample)

    sample = directory / "08_missing_fps_index.mxf"
    _missing_fps_mxf(ffmpeg, sample)
    files.append(sample)

    files.extend(_write_corrupt_files(directory))

    if include_long:
        sample = directory / "11_long_duration_400s.mxf"
        _long_duration_rewrite(
            ffmpeg, sample, duration=long_duration
        )
        files.append(sample)

        sample = directory / "12_good_after_timeout.mxf"
        _stereo_mxf(
            ffmpeg,
            sample,
            duration=duration,
            fps="25",
            codec="mpeg2",
        )
        files.append(sample)

    files = sorted(files, key=lambda path: path.name)
    if write_manifest:
        _write_manifest(directory, files, ffprobe)
    return SampleSet(directory=directory, files=tuple(files))
