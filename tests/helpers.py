from __future__ import annotations

import array
import json
import math
import shutil
import subprocess
import sys
import tempfile
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def temporary_workspace(prefix: str = "test-") -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=PROJECT_ROOT))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def write_wav(path: Path, sample_rate: int, channels: int, samples: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = array.array("h", samples)
    if sys.byteorder == "big":
        pcm.byteswap()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def sine_samples(
    frames: int,
    sample_rate: int,
    frequency: float = 220.0,
    amplitude: float = 0.1,
    channels: int = 1,
) -> list[int]:
    samples: list[int] = []
    for index in range(frames):
        value = int(
            round(
                amplitude
                * math.sin(2.0 * math.pi * frequency * index / sample_rate)
                * 32767.0
            )
        )
        samples.extend([value] * channels)
    return samples


def zero_samples(frames: int, channels: int = 1) -> list[int]:
    return [0] * (frames * channels)


def _commentary_track(
    name: str,
    path: str,
    channels: int,
    start_ms: float,
    end_ms: float,
    gain_db: float,
    pan: float,
    anchors: list[dict[str, float]],
    window_ms: float = 10.0,
    threshold_dbfs: float = -50.0,
    min_duration_ms: float = 30.0,
) -> dict[str, Any]:
    return {
        "name": name,
        "path": path,
        "channels": channels,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "gain_db": gain_db,
        "pan": pan,
        "anchors": anchors,
        "silence": {
            "window_ms": window_ms,
            "threshold_dbfs": threshold_dbfs,
            "min_duration_ms": min_duration_ms,
        },
    }


def _simple_track(
    name: str,
    path: str,
    channels: int,
    start_ms: float,
    end_ms: float,
    gain_db: float,
    pan: float,
) -> dict[str, Any]:
    return {
        "name": name,
        "path": path,
        "channels": channels,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "gain_db": gain_db,
        "pan": pan,
    }


def make_project(
    directory: Path,
    *,
    sample_rate: int = 8000,
    duration_ms: float = 400.0,
    output_channels: int = 1,
    block_frames: int = 128,
    target_rms_dbfs: float = -20.0,
    max_gain_db: float = 18.0,
    peak_limit_dbfs: float = -1.0,
    soft_knee_ratio: float = 0.9,
    commentary: list[dict[str, Any]] | None = None,
    engine_samples: list[int] | None = None,
    engine_channels: int = 1,
    ambience_samples: list[int] | None = None,
    ambience_channels: int = 1,
    output_path: str = "out\\mix.wav",
) -> Path:
    inputs = directory / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, int(round(duration_ms * sample_rate / 1000.0)))

    commentary = commentary or []
    commentary_dicts: list[dict[str, Any]] = []
    for track in commentary:
        filename = track["filename"]
        samples = track["samples"]
        channels = track["channels"]
        write_wav(inputs / filename, sample_rate, channels, samples)
        commentary_dicts.append(
            _commentary_track(
                track["name"],
                f"inputs\\{filename}",
                channels,
                track.get("start_ms", 0.0),
                track.get("end_ms", duration_ms),
                track.get("gain_db", 0.0),
                track.get("pan", 0.0),
                track["anchors"],
                track.get("window_ms", 10.0),
                track.get("threshold_dbfs", -50.0),
                track.get("min_duration_ms", 30.0),
            )
        )

    if engine_samples is None:
        engine_samples = zero_samples(frame_count, engine_channels)
    write_wav(inputs / "engine.wav", sample_rate, engine_channels, engine_samples)
    if ambience_samples is None:
        ambience_samples = zero_samples(frame_count, ambience_channels)
    write_wav(inputs / "ambience.wav", sample_rate, ambience_channels, ambience_samples)

    config = {
        "version": 1,
        "sample_rate": sample_rate,
        "bit_depth": 16,
        "channels": output_channels,
        "output_path": output_path,
        "replay": {
            "duration_ms": duration_ms,
            "frame_timestamps_ms": [0.0, duration_ms / 2.0, duration_ms],
        },
        "mix": {
            "target_rms_dbfs": target_rms_dbfs,
            "max_gain_db": max_gain_db,
            "peak_limit_dbfs": peak_limit_dbfs,
            "soft_knee_ratio": soft_knee_ratio,
            "block_frames": block_frames,
        },
        "tracks": {
            "commentary": commentary_dicts,
            "engine": _simple_track(
                "engine",
                "inputs\\engine.wav",
                engine_channels,
                0.0,
                duration_ms,
                -120.0,
                0.0,
            ),
            "ambience": _simple_track(
                "ambience",
                "inputs\\ambience.wav",
                ambience_channels,
                0.0,
                duration_ms,
                -120.0,
                0.0,
            ),
        },
    }
    config_path = directory / "replay.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return config_path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_cli(config_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "replay_mixer",
            "--config",
            str(config_path),
            *extra,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def parse_success(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    if result.returncode != 0:
        raise AssertionError(
            f"CLI failed rc={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise AssertionError(f"expected one stdout JSON line, got {lines!r}")
    return json.loads(lines[0])


def parse_error(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    lines = result.stderr.strip().splitlines()
    if len(lines) != 1:
        raise AssertionError(f"expected one stderr JSON line, got {lines!r}")
    return json.loads(lines[0])
