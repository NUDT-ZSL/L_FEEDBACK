from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from locdub import audioio

SR = 44100


def _write_wav(path: Path, seconds: float, channels: int = 1) -> None:
    frames = int(SR * seconds)
    t = np.arange(frames) / SR
    mono = 0.5 * np.sin(2 * np.pi * 330 * t)
    data = np.stack((mono, mono), axis=1) if channels == 2 else mono[:, None]
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, SR, subtype="PCM_16")


def test_ensure_wav_rejects_missing_and_non_wav(tmp_path):
    with pytest.raises(audioio.AudioError):
        audioio.ensure_wav(tmp_path / "missing.wav")
    other = tmp_path / "voice.mp3"
    other.write_text("not audio", encoding="utf-8")
    with pytest.raises(audioio.AudioError):
        audioio.ensure_wav(other)


def test_probe_reports_duration(tmp_path):
    path = tmp_path / "a.wav"
    _write_wav(path, 1.5, channels=2)
    info = audioio.probe_wav(path)
    assert info.sample_rate == SR
    assert info.channels == 2
    assert info.frames == int(SR * 1.5)
    assert abs(info.duration_ms - 1500.0) < 1.0


def test_adapt_segment_hits_target_and_clamps(tmp_path):
    path = tmp_path / "a.wav"
    _write_wav(path, 2.0)
    target_ms = 3000  # ratio 1.5
    adaptation = audioio.adapt_segment(
        path, target_ms, min_ratio=0.8, max_ratio=1.2,
        max_block_frames=SR * 60,
    )
    # Requested 1.5 exceeds max 1.2 -> clamped.
    assert adaptation.boundary_clamped is True
    assert adaptation.applied_ratio == 1.2
    expected_frames = int(round(SR * 2.0 * 1.2))
    assert adaptation.audio.shape[0] == expected_frames
    assert adaptation.sample_rate == SR


def test_adapt_segment_within_range_no_clamp(tmp_path):
    path = tmp_path / "a.wav"
    _write_wav(path, 2.0)
    adaptation = audioio.adapt_segment(
        path, 1500, min_ratio=0.5, max_ratio=2.0,
        max_block_frames=SR * 60,
    )
    assert adaptation.boundary_clamped is False
    assert abs(adaptation.applied_ratio - 0.75) < 1e-9
    assert adaptation.audio.shape[0] == int(round(SR * 2.0 * 0.75))


def test_adapt_segment_rejects_oversized_clip(tmp_path):
    path = tmp_path / "a.wav"
    _write_wav(path, 2.0)
    with pytest.raises(audioio.AudioError):
        audioio.adapt_segment(
            path, 1000, min_ratio=0.5, max_ratio=2.0,
            max_block_frames=SR,  # 1 second guard, file is 2 seconds
        )


def test_block_reader_roundtrip(tmp_path):
    path = tmp_path / "a.wav"
    _write_wav(path, 2.5, channels=2)
    blocks = list(audioio.read_wav_blocks(path, block_seconds=1.0))
    assert len(blocks) >= 2
    total = sum(len(block) for block in blocks)
    assert total == int(SR * 2.5)


def test_corrupt_wav_raises_audio_error(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEnotreallyaudio")
    with pytest.raises(audioio.AudioError):
        audioio.probe_wav(path)


def test_non_wav_extension_rejected(tmp_path):
    path = tmp_path / "voice.flac"
    path.write_bytes(b"dummy")
    with pytest.raises(audioio.AudioError):
        audioio.ensure_wav(path)
