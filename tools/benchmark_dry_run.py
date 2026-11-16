from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.sample_factory import write_pcm24  # noqa: E402


SAMPLE_RATE = 48000
FRAME_RATE = 25
TOTAL_SECONDS = 3600
PREVIEW_SECONDS = 60


def _expand_wav_sparse(path: Path, total_seconds: int) -> None:
    data = bytearray(path.read_bytes()[:44])
    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise RuntimeError("unexpected WAV header")
    data_size = total_seconds * SAMPLE_RATE * 2 * 3
    struct.pack_into("<I", data, 4, 36 + data_size)
    struct.pack_into("<I", data, 40, data_size)
    with path.open("r+b") as handle:
        handle.seek(0)
        handle.write(data)
        handle.truncate(44 + data_size)


def _deterministic_preview(seed: int, frequency: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = PREVIEW_SECONDS * SAMPLE_RATE
    t = np.arange(count, dtype=np.float64) / SAMPLE_RATE
    left = (
        np.sin(2 * np.pi * frequency * t)
        + 0.35 * np.sin(2 * np.pi * frequency * 1.47 * t + 0.2)
        + 0.1 * rng.standard_normal(count)
    )
    left = left / (np.max(np.abs(left)) or 1.0) * 0.3
    right = np.roll(left, 31)
    return np.stack((left, right), axis=1).astype(np.float32)


def build_benchmark_sample(root: Path) -> Path:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    languages = {
        "zh-CN": _deterministic_preview(101, 233.0),
        "en-US": _deterministic_preview(103, 311.0),
        "ja-JP": _deterministic_preview(107, 389.0),
    }
    for language_id, preview in languages.items():
        path = root / f"{language_id}.wav"
        write_pcm24(path, preview, SAMPLE_RATE)
        _expand_wav_sparse(path, TOTAL_SECONDS)

    reference = languages["zh-CN"]
    delay = int(0.020 * SAMPLE_RATE)
    delayed = np.concatenate(
        (np.zeros((delay, 2), dtype=np.float32), reference[:-delay]), axis=0
    )
    reference_path = root / "reference.wav"
    write_pcm24(reference_path, delayed, SAMPLE_RATE)
    video_path = root / "program.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=160x90:r=25:d={PREVIEW_SECONDS}",
            "-i",
            str(reference_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s24le",
            "-shortest",
            str(video_path),
        ],
        check=True,
    )

    manifest = {
        "frame_rate": FRAME_RATE,
        "duration_frames": TOTAL_SECONDS * FRAME_RATE,
        "crossfade_ms": 15,
        "audio": {"sample_rate": SAMPLE_RATE, "channels": 2, "bit_depth": 24},
        "video": {"path": video_path.name, "reference_stream_index": 1},
        "languages": [
            {
                "id": language_id,
                "path": f"{language_id}.wav",
                "sample_rate": SAMPLE_RATE,
                "channels": 2,
                "bit_depth": 24,
            }
            for language_id in languages
        ],
        "switch_points": [
            {"frame": 0, "language": "zh-CN"},
            {"frame": 12 * 60 * FRAME_RATE, "language": "en-US"},
            {"frame": 30 * 60 * FRAME_RATE, "language": "ja-JP"},
            {"frame": 48 * 60 * FRAME_RATE, "language": "zh-CN"},
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest_path


def main() -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path("/tmp/bm_bench_1h")
    manifest_path = build_benchmark_sample(root)
    command = [
        sys.executable,
        "-m",
        "broadcast_master",
        "--manifest",
        str(manifest_path),
        "--dry-run",
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8"
    )
    wall_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        return completed.returncode
    payload = json.loads(completed.stdout)
    payload["benchmark_wall_seconds"] = round(wall_seconds, 3)
    payload["benchmark_requirement_seconds"] = 60
    payload["benchmark_passed"] = wall_seconds < 60.0
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["benchmark_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

