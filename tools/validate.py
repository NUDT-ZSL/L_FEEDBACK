from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from immersive_audio.container import ContainerReader
from immersive_audio.timeline import block_count

from tools.generate_demo import generate


def _run(command: list[str], timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _version(command: list[str]) -> str:
    process = _run(command, timeout=20)
    if process.returncode != 0:
        return "unavailable"
    return process.stdout.splitlines()[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic unit and end-to-end acceptance checks"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/validation-report.json"),
    )
    args = parser.parse_args()

    tests = _run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        timeout=300,
    )
    tests_ok = tests.returncode == 0
    match = re.search(r"Ran (\d+) tests", tests.stderr)
    test_count = int(match.group(1)) if match else 0

    probe_path = shutil.which("ffprobe")
    if not probe_path:
        raise RuntimeError("ffprobe is not available in PATH")
    ffmpeg_path = shutil.which("ffmpeg")

    with tempfile.TemporaryDirectory(prefix="iapano-validation-") as temporary:
        work = Path(temporary)
        scene_path, _, _ = generate(work)
        container = work / "demo.iap"
        wav = work / "demo-8ch.wav"
        pack_stats_path = work / "pack-stats.json"
        render_stats_path = work / "render-stats.json"
        packed = _run(
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
        if packed.returncode != 0:
            raise RuntimeError(packed.stderr)
        rendered = _run(
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
        if rendered.returncode != 0:
            raise RuntimeError(rendered.stderr)

        reader = ContainerReader(container)
        checksum = reader.verify_payload()
        wave_info = sf.info(wav)
        expected_frames = block_count(
            reader.header.sample_count, reader.header.frame_length
        )
        probe_process = _run(
            [
                probe_path,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate,channels,duration",
                "-of",
                "json",
                str(wav),
            ]
        )
        if probe_process.returncode != 0:
            raise RuntimeError(probe_process.stderr)
        probe = json.loads(probe_process.stdout)["streams"][0]
        decode_process = (
            _run(
                [
                    ffmpeg_path,
                    "-v",
                    "error",
                    "-i",
                    str(wav),
                    "-f",
                    "null",
                    "-",
                ],
                timeout=60,
            )
            if ffmpeg_path
            else None
        )
        pack_stats = json.loads(packed.stdout)
        render_stats = json.loads(rendered.stdout)
        validation = {
            "checks": {
                "checksum_matches_header": checksum
                == reader.header.payload_sha256.hex(),
                "ffmpeg_decode_ok": decode_process is not None
                and decode_process.returncode == 0,
                "ffprobe_channels_8": int(probe["channels"]) == 8,
                "ffprobe_codec_pcm_f32le": probe["codec_name"] == "pcm_f32le",
                "ffprobe_duration_seconds": float(probe["duration"]),
                "metadata_blocks_aligned": reader.header.frame_count
                == expected_frames,
                "pack_memory_within_512_mb": pack_stats["within_memory_limit"],
                "render_memory_within_512_mb": render_stats["within_memory_limit"],
                "wav_frames_equal_container_samples": wave_info.frames
                == reader.header.sample_count,
                "wav_sample_rate_equal_container": wave_info.samplerate
                == reader.header.sample_rate,
            },
            "container": {
                "audio_samples": reader.header.sample_count,
                "channels": reader.header.channel_count,
                "frame_length": reader.header.frame_length,
                "metadata_frames": reader.header.frame_count,
                "objects": reader.header.object_count,
                "payload_sha256": checksum,
                "sample_rate": reader.header.sample_rate,
                "version": reader.header.version,
            },
            "environment": {
                "ffmpeg": _version(["ffmpeg", "-version"]),
                "ffprobe": _version(["ffprobe", "-version"]),
                "numpy": np.__version__,
                "python": sys.version.split()[0],
                "soundfile": sf.__version__,
            },
            "ffprobe": probe,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "pack": pack_stats,
            "render": render_stats,
            "tests": {
                "count": test_count,
                "status": "ok" if tests_ok else "failed",
                "tail": tests.stderr.strip().splitlines()[-1:],
            },
            "status": "ok" if tests_ok else "failed",
        }
        assert all(
            value
            for value in validation["checks"].values()
            if isinstance(value, bool)
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
