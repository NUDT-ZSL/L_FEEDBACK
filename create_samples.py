#!/usr/bin/env python3
"""Deterministically generate local sample streams for the ABR TS muxer.

Generates (no network required, ffmpeg must be on PATH):
  samples/video_low_800k.mp4    H.264 640x360   800 kbps  GOP 2s
  samples/video_mid_2500k.mp4   H.264 1280x720  2500 kbps GOP 2s
  samples/video_high_5000k.mp4  H.264 1920x1080 5000 kbps GOP 4s
  samples/audio_main.aac        AAC-LC 48 kHz stereo 128 kbps (ADTS)
  samples/video_mid_corrupt.mp4 truncated faststart copy of the mid stream,
                                used to exercise the fallback path

All sources use deterministic synthetic generators (testsrc2 / sine) so
re-running the script produces bit-identical timelines.
"""
import subprocess
import sys
from pathlib import Path

DURATION_SECONDS = 30
FPS = 24
SAMPLE_DIR = Path(__file__).resolve().parent / "samples"

VIDEO_PROFILES = [
    {"id": "low", "width": 640, "height": 360, "bitrate_kbps": 800, "gop_seconds": 2},
    {"id": "mid", "width": 1280, "height": 720, "bitrate_kbps": 2500, "gop_seconds": 2},
    {"id": "high", "width": 1920, "height": 1080, "bitrate_kbps": 5000, "gop_seconds": 4},
]

AUDIO_BITRATE_KBPS = 128
AUDIO_PATH = SAMPLE_DIR / "audio_main.aac"
CORRUPT_FRACTION = 0.45


def run(cmd):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def video_path(profile):
    return SAMPLE_DIR / f"video_{profile['id']}_{profile['bitrate_kbps']}k.mp4"


def make_video(profile):
    out = video_path(profile)
    gop = profile["gop_seconds"] * FPS
    bitrate = f"{profile['bitrate_kbps']}k"
    run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i",
        f"testsrc2=size={profile['width']}x{profile['height']}:rate={FPS}:duration={DURATION_SECONDS}",
        "-c:v", "libx264", "-preset", "veryfast",
        "-b:v", bitrate, "-minrate", bitrate, "-maxrate", bitrate,
        "-bufsize", f"{profile['bitrate_kbps'] * 2}k",
        "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
        "-bf", "0", "-pix_fmt", "yuv420p", "-an",
        "-movflags", "+faststart",
        out,
    ])
    return out


def make_audio():
    run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i",
        f"sine=frequency=440:sample_rate=48000:duration={DURATION_SECONDS}",
        "-c:a", "aac", "-b:a", f"{AUDIO_BITRATE_KBPS}k",
        "-f", "adts", AUDIO_PATH,
    ])


def make_corrupt_variant(source):
    """Truncate a faststart MP4 so probing works but decoding fails mid-file."""
    corrupt = SAMPLE_DIR / "video_mid_corrupt.mp4"
    data = source.read_bytes()
    corrupt.write_bytes(data[: int(len(data) * CORRUPT_FRACTION)])
    return corrupt


def main():
    SAMPLE_DIR.mkdir(exist_ok=True)
    mid_path = None
    for profile in VIDEO_PROFILES:
        out = make_video(profile)
        if profile["id"] == "mid":
            mid_path = out
    make_audio()
    corrupt = make_corrupt_variant(mid_path)
    print(f"samples written to {SAMPLE_DIR}")
    print(f"corrupt fallback-test sample: {corrupt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
