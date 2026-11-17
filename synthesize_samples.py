#!/usr/bin/env python3
"""Create deterministic local cutscene samples with FFmpeg lavfi."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


def run(command: Sequence[str]) -> None:
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"{command[0]} failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic FFmpeg samples for the localizer."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory for generated samples",
    )
    parser.add_argument(
        "--container",
        choices=("mkv", "mp4"),
        default="mkv",
        help="original container type (default: mkv)",
    )
    parser.add_argument(
        "--audio-codec",
        choices=("pcm_s16le", "aac"),
        default="pcm_s16le",
        help="original container audio codec (default: pcm_s16le)",
    )
    parser.add_argument(
        "--extra-track",
        action="store_true",
        help="include a third audio track to exercise ignored-track reporting",
    )
    return parser


def generate(args: argparse.Namespace) -> dict[str, object]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe must be available in PATH")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dialogue_path = output_dir / "original_dialogue.wav"
    effect_path = output_dir / "original_effect.wav"
    localized_path = output_dir / "localized_dialogue.wav"
    master_path = output_dir / f"original.{args.container}"

    dialogue_filter = (
        "aevalsrc="
        "0.20*sin(2*PI*440*t)|0.20*sin(2*PI*660*t)"
        ":s=48000:d=2.0"
    )
    effect_filter = (
        "aevalsrc="
        "0.04*sin(2*PI*1000*t)"
        "+0.65*between(t\\,0.50\\,0.502)*sin(2*PI*1000*t)"
        "|"
        "0.04*sin(2*PI*1500*t)"
        "-0.55*between(t\\,0.50\\,0.502)*sin(2*PI*1500*t)"
        ":s=48000:d=2.5"
    )
    localized_filter = "aevalsrc=0.25*sin(2*PI*220*t):s=44100:d=1.5"

    run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            dialogue_filter,
            "-c:a",
            "pcm_s16le",
            str(dialogue_path),
        ]
    )
    run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            effect_filter,
            "-c:a",
            "pcm_s16le",
            str(effect_path),
        ]
    )
    run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            localized_filter,
            "-c:a",
            "pcm_s16le",
            str(localized_path),
        ]
    )

    track_paths = [dialogue_path, effect_path]
    if args.extra_track:
        extra_path = output_dir / "ignored_extra.wav"
        run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "aevalsrc=0.05*sin(2*PI*330*t):s=48000:d=0.8",
                "-c:a",
                "pcm_s16le",
                str(extra_path),
            ]
        )
        track_paths.append(extra_path)

    mux_command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    for track_path in track_paths:
        mux_command.extend(["-i", str(track_path)])
    for ordinal in range(len(track_paths)):
        mux_command.extend(["-map", f"{ordinal}:a:0"])
    mux_command.extend(["-c:a", args.audio_codec, str(master_path)])
    run(mux_command)

    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_name,codec_type,channels,sample_rate",
            "-of",
            "json",
            str(master_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    audio_streams = [
        stream for stream in streams if stream.get("codec_type") == "audio"
    ]

    manifest = {
        "original_container": str(master_path),
        "localized_dialogue": str(localized_path),
        "audio_track_count": len(audio_streams),
        "ignored_audio_tracks_with_tool": max(0, len(audio_streams) - 2),
        "audio_streams": audio_streams,
        "sample_design": {
            "dialogue": "stereo 48 kHz, 440/660 Hz tones, 2.0 s",
            "effects": "stereo 48 kHz, low bed plus 0.50 s pulse, 2.5 s",
            "localized_dialogue": "mono 44.1 kHz, 220 Hz tone, 1.5 s",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = generate(args)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
