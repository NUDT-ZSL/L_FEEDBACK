from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dialogue_master.loudness import integrated_loudness
from dialogue_master.output import write_master_wav

from conftest import base_config, clip, run_cli, write_config, write_tone


@pytest.mark.parametrize(
    "bit_depth,subtype",
    [
        (8, "PCM_U8"),
        (16, "PCM_16"),
        (24, "PCM_24"),
        (32, "PCM_32"),
    ],
)
def test_wavex_writer_preserves_bit_depth(
    tmp_path: Path, bit_depth: int, subtype: str
) -> None:
    path = tmp_path / "out.wav"
    samples = np.zeros((441, 2), dtype=np.float64)
    samples[:, 0] = np.linspace(-0.5, 0.5, 441)
    write_master_wav(path, samples, 44100, bit_depth, ("FL", "FR"))
    info = sf.info(str(path))
    assert info.format == "WAVEX"
    assert info.samplerate == 44100
    assert info.channels == 2
    assert info.subtype == subtype
    assert info.frames == 441


def test_5_1_layout_is_written_with_channel_mask(tmp_path: Path) -> None:
    write_tone(
        tmp_path / "input" / "surround.wav",
        duration_ms=600,
        amplitude=0.01,
        channels=6,
    )
    config_path = tmp_path / "config.json"
    write_config(
        config_path,
        base_config(
            timeline_ms=1000,
            channels=6,
            channel_layout="5.1",
            clips=[clip("surround", "input/surround.wav", 0, -28, 10, 10)],
        ),
    )
    completed = run_cli(config_path, tmp_path / "out")
    assert completed.returncode == 0
    info = sf.info(str(tmp_path / "out" / "master.wav"))
    assert info.channels == 6
    assert info.frames == 48000
    log = json.loads((tmp_path / "out" / "processing_log.json").read_text())
    assert log["config"]["channel_layout"] == "5.1"
    assert log["config"]["channel_labels"] == [
        "FL", "FR", "FC", "LFE", "BL", "BR"
    ]

    if shutil.which("ffprobe"):
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=channels,channel_layout",
                "-of",
                "default=nw=1",
                str(tmp_path / "out" / "master.wav"),
            ],
            text=True,
            capture_output=True,
        )
        assert probe.returncode == 0
        assert "channels=6" in probe.stdout
        assert "channel_layout=5.1" in probe.stdout


def test_lfe_is_excluded_from_bs1770_loudness() -> None:
    sample_rate = 48000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    lfe_only = (0.1 * np.sin(2.0 * np.pi * 60.0 * time))[:, None]
    assert integrated_loudness(lfe_only, sample_rate, ("LFE",)) is None


@pytest.mark.skipif(
    not shutil.which("ffmpeg"),
    reason="FFmpeg is required for the independent loudness cross-check",
)
def test_loudness_matches_ffmpeg_ebur128(tmp_path: Path) -> None:
    sample_rate = 48000
    frame_count = sample_rate * 4
    rng = np.random.default_rng(12345)
    time = np.arange(frame_count, dtype=np.float64) / sample_rate
    mono = (
        0.035 * np.sin(2.0 * np.pi * 220.0 * time)
        + 0.02 * np.sin(2.0 * np.pi * 440.0 * time)
        + 0.004 * rng.standard_normal(frame_count)
    )
    envelope = 0.25 + 0.75 * np.abs(np.sin(2.0 * np.pi * 0.6 * time))
    data = np.repeat((mono * envelope)[:, None], 2, axis=1)
    path = tmp_path / "check.wav"
    sf.write(str(path), data, sample_rate, format="WAVEX", subtype="PCM_24")
    measured, _ = sf.read(str(path), dtype="float64", always_2d=True)
    ours = integrated_loudness(measured, sample_rate, ("FL", "FR"))
    assert ours is not None

    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-filter_complex",
            "ebur128",
            "-f",
            "null",
            "-",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0
    matches = re.findall(
        r"I:\s+(-?\d+(?:\.\d+)?)\s+LUFS", completed.stderr
    )
    assert matches
    ffmpeg_lufs = float(matches[-1])
    assert abs(ours - ffmpeg_lufs) < 0.15
