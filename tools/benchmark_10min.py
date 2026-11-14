from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def _write_bed(path: Path) -> None:
    sample_rate = 48_000
    duration = 6.0
    time = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    signal = 0.01 * np.sin(2 * np.pi * 997.0 * time)
    sf.write(path, signal.astype(np.float32), sample_rate, subtype="FLOAT")


def _write_scene(path: Path, bed_name: str, duration: float = 600.0) -> None:
    sample_rate = 48_000
    objects = []
    for object_index in range(64):
        clips = [
            {
                "path": bed_name,
                "start": clip_index * 6.0,
                "loops": 1,
            }
            for clip_index in range(100)
        ]
        azimuth = (object_index * 360.0) / 64.0
        objects.append(
            {
                "clips": clips,
                "id": f"object-{object_index:02d}",
                "keyframes": [
                    {
                        "azimuth": azimuth,
                        "distance": 0.5 + object_index / 64.0,
                        "elevation": 0.0,
                        "gain": 1.0,
                        "size": 0.1,
                        "time": 0.0,
                    },
                    {
                        "azimuth": (azimuth + 180.0) % 360.0,
                        "distance": 2.0,
                        "elevation": 5.0,
                        "gain": 0.5,
                        "size": 0.4,
                        "time": duration,
                    },
                ],
            }
        )
    scene = {
        "duration_seconds": duration,
        "frame_length": 1024,
        "objects": objects,
        "sample_rate": sample_rate,
    }
    path.write_text(json.dumps(scene, separators=(",", ":")), encoding="utf-8")


def _run(command: list[str]) -> dict[str, object]:
    process = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1_800,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(command)}\n{process.stderr}"
        )
    return json.loads(process.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic 64-object, 100-clips/object, 10-minute benchmark"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="retain generated inputs and outputs in this directory",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/benchmark-64x100-10min.json"),
    )
    args = parser.parse_args()

    temporary = None
    if args.work_dir is None:
        scratch = ROOT / "artifacts"
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix="iapano-benchmark-", dir=scratch
        )
        work = Path(temporary.name)
    else:
        work = args.work_dir
        work.mkdir(parents=True, exist_ok=True)

    bed = work / "bed-6s.wav"
    scene_path = work / "scene-64x100-10min.json"
    container = work / "scene-64x100-10min.iap"
    wav = work / "scene-64x100-10min-8ch.wav"
    pack_stats_path = work / "pack-stats.json"
    render_stats_path = work / "render-stats.json"
    _write_bed(bed)
    _write_scene(scene_path, bed.name)

    started = time.perf_counter()
    pack_stats = _run(
        [
            sys.executable,
            "-m",
            "immersive_audio",
            "pack",
            str(scene_path),
            str(container),
            "--stats",
            str(pack_stats_path),
        ]
    )
    render_stats = _run(
        [
            sys.executable,
            "-m",
            "immersive_audio",
            "render",
            str(container),
            str(wav),
            "--stats",
            str(render_stats_path),
        ]
    )
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is not available in PATH")
    probe_process = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,duration",
            "-of",
            "json",
            str(wav),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    probe = json.loads(probe_process.stdout)["streams"][0]
    report = {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "ffprobe": probe,
        "input": {
            "clips_per_object": 100,
            "objects": 64,
            "timeline_seconds": 600,
        },
        "output_bytes": {
            "container": container.stat().st_size,
            "wav": wav.stat().st_size,
        },
        "pack": pack_stats,
        "render": render_stats,
        "status": "ok",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if temporary is not None:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
