from pathlib import Path

import numpy as np
import soundfile as sf

from locdub import mixer

SR = 8000


def _write(path: Path, signal: np.ndarray, channels: int = 1) -> None:
    data = signal if signal.ndim == 2 else signal[:, None]
    if channels == 2 and data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, SR, subtype="PCM_16")


def test_mix_places_clips_at_timeline_positions(tmp_path):
    background = np.full(SR * 10, 0.1, dtype=np.float64)
    bg_path = tmp_path / "bg.wav"
    _write(bg_path, background, channels=2)

    clip = np.full(SR, 0.5, dtype=np.float64)  # 1 second tone
    clip_path = tmp_path / "clip.wav"
    _write(clip_path, clip, channels=1)

    jobs = [
        mixer.MixJob("c1", 2000, 3000, clip_path),
        mixer.MixJob("silent", 5000, 6000, None),
    ]
    out = tmp_path / "mix.wav"
    mixer.mix_stream(bg_path, jobs, out, sample_rate=SR, channels=2,
                     gain_db=0.0, block_seconds=2.0)

    mixed, _ = sf.read(str(out), dtype="float64")
    assert len(mixed) == len(background)
    # Before the clip only background is present.
    assert np.allclose(mixed[int(1.0 * SR):int(1.9 * SR)], 0.1, atol=1e-3)
    # Inside the clip window both background and clip are summed.
    inside = mixed[int(2.1 * SR):int(2.9 * SR)]
    assert np.allclose(inside, 0.6, atol=2e-2)
    # Silence placeholder window contains background only.
    silent_window = mixed[int(5.1 * SR):int(5.9 * SR)]
    assert np.allclose(silent_window, 0.1, atol=1e-3)


def test_mix_applies_gain_and_clips(tmp_path):
    background = np.zeros(SR * 4, dtype=np.float64)
    bg_path = tmp_path / "bg.wav"
    _write(bg_path, background, channels=1)
    clip = np.full(SR, 0.9, dtype=np.float64)
    clip_path = tmp_path / "clip.wav"
    _write(clip_path, clip, channels=1)
    jobs = [mixer.MixJob("c1", 0, 1000, clip_path)]
    out = tmp_path / "mix.wav"
    mixer.mix_stream(bg_path, jobs, out, sample_rate=SR, channels=1,
                     gain_db=6.0, block_seconds=1.0)
    mixed, _ = sf.read(str(out), dtype="float64")
    # +6 dB roughly doubles amplitude; clipping caps at 1.0.
    assert mixed.max() <= 1.0 + 1e-6
    assert np.allclose(mixed[int(0.1 * SR):int(0.9 * SR)], 1.0, atol=2e-2)


def test_mix_many_segments_bounded_blocks(tmp_path):
    # 60 short segments over 30 seconds exercises block boundary handling.
    background = np.zeros(SR * 30, dtype=np.float64)
    bg_path = tmp_path / "bg.wav"
    _write(bg_path, background, channels=1)
    jobs = []
    for index in range(60):
        start = index * 400 + 50
        end = start + 300
        clip = np.full(int(SR * 0.3), 0.4, dtype=np.float64)
        clip_path = tmp_path / f"clip_{index}.wav"
        _write(clip_path, clip, channels=1)
        jobs.append(mixer.MixJob(f"c{index}", start, end, clip_path))
    out = tmp_path / "mix.wav"
    mixer.mix_stream(bg_path, jobs, out, sample_rate=SR, channels=1,
                     gain_db=0.0, block_seconds=5.0)
    mixed, info = sf.read(str(out), dtype="float64")
    assert len(mixed) == SR * 30
    assert np.isfinite(mixed).all()
    # Every clip window carries energy.
    for index in range(60):
        start = index * 400 + 50
        sample = int((start + 100) / 1000 * SR)
        assert mixed[sample] > 0.3
