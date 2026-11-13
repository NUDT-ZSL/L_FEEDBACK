"""Deterministic synthetic project generator used by the acceptance tests."""

from __future__ import annotations

from fractions import Fraction
import json
import os
from typing import Any

import numpy as np

from .dsp import integrated_loudness, make_sync_template
from .util import db_to_linear, round_half_even
from .wavio import write_wav


SCENARIOS = (
    "resample",
    "validate_ok",
    "validate_sr_bad",
    "validate_channels_bad",
    "sync",
    "sync_multirate",
    "sync_far",
    "sync_missing",
    "gain_ramp",
    "gain_bad_order",
    "gain_out_of_range",
    "smooth_step",
    "smooth_gradient",
    "smooth_clamped",
    "limiter_loud",
    "limiter_safe",
    "loudness_pink",
    "loudness_conflict",
    "loudness_equal",
    "format",
    "failure_partial",
)


def _write_json(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")


def _encoding_for_depth(bit_depth: int, float_source: bool = False) -> str:
    if float_source:
        return "float32"
    return {16: "pcm16", 24: "pcm24", 32: "pcm32"}[bit_depth]


def _default_sync(seconds: float, sample_rate: int) -> dict[str, Any]:
    return {
        "expected_sample_position": round_half_even(Fraction(str(seconds)) * sample_rate),
        "threshold_samples": 96,
        "frequency_hz": 2000,
        "cycles": 8,
        "amplitude": 0.1,
        "detection_threshold": 0.35,
    }


def _marker_duration(sample_rate: int, sync: dict[str, Any]) -> int:
    return max(
        2,
        round_half_even(
            Fraction(
                int(sync.get("cycles", 8)) * sample_rate,
                int(sync.get("frequency_hz", 2000)),
            )
        ),
    )


def _deterministic_noise(length: int, channels: int, seed: int) -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(seed))
    white = generator.standard_normal((length, channels), dtype=np.float64)
    spectrum = np.fft.rfft(white, axis=0)
    frequencies = np.fft.rfftfreq(length, d=1.0 / max(1.0, float(length)))
    scale = np.ones_like(frequencies)
    scale[1:] = 1.0 / np.sqrt(frequencies[1:])
    pink = np.fft.irfft(spectrum * scale[:, None], n=length, axis=0)
    current_rms = float(np.sqrt(np.mean(pink * pink, dtype=np.float64)))
    if current_rms > 0:
        pink /= current_rms
    return pink


def _track_signal(
    sample_rate: int,
    channels: int,
    duration_seconds: float,
    signal: str = "sine",
    frequency: float = 1000.0,
    amplitude_dbfs: float = -6.0,
    phase: float = 0.0,
    sync_seconds: float = 0.010,
    inserted_lead_samples: int = 0,
    add_marker: bool = True,
    noise_seed: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    frames = round_half_even(Fraction(str(duration_seconds)) * sample_rate)
    audio = np.zeros((frames, channels), dtype=np.float64)
    sync = _default_sync(sync_seconds, sample_rate)
    marker_position = int(sync["expected_sample_position"]) + inserted_lead_samples
    marker_frames = _marker_duration(sample_rate, sync)
    marker = make_sync_template(sample_rate, sync)
    content_start = marker_position + marker_frames + round_half_even(
        Fraction(2, 100) * sample_rate
    )
    if content_start >= frames:
        content_start = min(frames, marker_frames)
    content_frames = frames - content_start
    n = np.arange(content_frames, dtype=np.float64)
    amplitude = db_to_linear(amplitude_dbfs)
    if signal == "sine":
        waveform = amplitude * np.sin(
            2.0 * np.pi * frequency * n / float(sample_rate) + phase
        )
    elif signal == "constant":
        waveform = np.full(content_frames, amplitude, dtype=np.float64)
    elif signal == "pink":
        waveform = _deterministic_noise(content_frames, 1, noise_seed)[:, 0] * amplitude
    else:
        raise ValueError(f"unknown generator signal {signal!r}")
    for channel in range(channels):
        audio[content_start : content_start + content_frames, channel] = waveform
    if add_marker and marker_position + marker_frames <= frames:
        for channel in range(channels):
            audio[marker_position : marker_position + marker_frames, channel] += marker
    metadata = {
        "sync": sync,
        "marker_actual_position": marker_position,
        "content_start_sample": content_start,
        "content_start_seconds": float(Fraction(content_start, sample_rate)),
    }
    return audio, metadata


def _add_track(
    root: str,
    manifest_tracks: list[dict[str, Any]],
    name: str,
    sample_rate: int,
    channels: int,
    bit_depth: int,
    duration: float,
    automation: list[dict[str, float]],
    signal: str = "sine",
    frequency: float = 1000.0,
    amplitude_dbfs: float = -6.0,
    sync_seconds: float = 0.010,
    inserted_lead_samples: int = 0,
    add_marker: bool = True,
    declared_sample_rate: int | None = None,
    declared_channels: int | None = None,
    declared_bit_depth: int | None = None,
    float_source: bool = False,
    noise_seed: int = 1,
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    audio, metadata = _track_signal(
        sample_rate,
        channels,
        duration,
        signal=signal,
        frequency=frequency,
        amplitude_dbfs=amplitude_dbfs,
        sync_seconds=sync_seconds,
        inserted_lead_samples=inserted_lead_samples,
        add_marker=add_marker,
        noise_seed=noise_seed,
    )
    relative = f"sources/{name}.wav"
    write_wav(
        os.path.join(root, relative),
        audio,
        sample_rate,
        _encoding_for_depth(bit_depth, float_source),
    )
    declared_sr = sample_rate if declared_sample_rate is None else declared_sample_rate
    declared_ch = channels if declared_channels is None else declared_channels
    declared_depth = bit_depth if declared_bit_depth is None else declared_bit_depth
    track = {
        "name": name,
        "file": relative,
        "expected_sample_rate": int(declared_sr),
        "expected_bit_depth": int(declared_depth),
        "expected_channels": int(declared_ch),
        "gain_automation": automation,
        "sync": metadata["sync"],
    }
    if metadata_extra:
        track.update(metadata_extra)
    manifest_tracks.append(track)
    return metadata


def _base_config(**updates: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "sample_rate": 48000,
        "channels": 1,
        "encoding": "float32",
        "peak_limit_dbfs": -1.0,
        "limiter_lookahead_ms": 5.0,
        "smooth_window_ms": 10.0,
        "loudness_target": None,
        "resampler": {"half_width_samples": 64},
    }
    config.update(updates)
    return config


def _finish(
    root: str,
    tracks: list[dict[str, Any]],
    config: dict[str, Any],
    timeline_seconds: float,
    extra: dict[str, Any] | None = None,
) -> str:
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "timeline_seconds": timeline_seconds,
        "allowed_sample_rates": [44100, 48000],
        "allowed_bit_depths": [16, 24, 32],
        "output_config": "output_config.json",
        "tracks": tracks,
    }
    if extra:
        manifest.update(extra)
    _write_json(os.path.join(root, "manifest.json"), manifest)
    _write_json(os.path.join(root, "output_config.json"), config)
    return root


def generate_project(scenario: str, output_root: str) -> str:
    os.makedirs(output_root, exist_ok=True)
    root = os.path.abspath(output_root)
    tracks: list[dict[str, Any]] = []
    if scenario == "validate_ok":
        _add_track(
            root,
            tracks,
            "music_mono",
            48000,
            1,
            24,
            1.0,
            [{"time": 0.0, "gain_db": -6.0}, {"time": 1.0, "gain_db": -3.0}],
            amplitude_dbfs=-12.0,
        )
        return _finish(root, tracks, _base_config(), 1.0)
    if scenario == "validate_sr_bad":
        _add_track(
            root,
            tracks,
            "declared_48k_actual_44k1",
            44100,
            1,
            16,
            1.0,
            [{"time": 0.0, "gain_db": 0.0}],
            declared_sample_rate=48000,
        )
        return _finish(root, tracks, _base_config(), 1.0)
    if scenario == "validate_channels_bad":
        _add_track(
            root,
            tracks,
            "declared_stereo_actual_mono",
            48000,
            1,
            16,
            1.0,
            [{"time": 0.0, "gain_db": 0.0}],
            declared_channels=2,
        )
        return _finish(root, tracks, _base_config(), 1.0)
    if scenario == "resample":
        _add_track(
            root,
            tracks,
            "sine_44k1_mono",
            44100,
            1,
            16,
            1.0,
            [{"time": 0.0, "gain_db": 0.0}],
            amplitude_dbfs=-9.0,
        )
        _add_track(
            root,
            tracks,
            "sine_48k_stereo",
            48000,
            2,
            24,
            1.0,
            [{"time": 0.0, "gain_db": 0.0}],
            amplitude_dbfs=-15.0,
        )
        return _finish(root, tracks, _base_config(), 1.0)
    if scenario in {"sync", "sync_multirate", "sync_far"}:
        extra = 37 if scenario == "sync" else 200
        if scenario == "sync_multirate":
            extra = 37
        _add_track(
            root,
            tracks,
            "sync_reference",
            48000,
            1,
            16,
            1.0,
            [{"time": 0.0, "gain_db": 0.0}],
            amplitude_dbfs=-12.0,
            frequency=700.0,
        )
        _add_track(
            root,
            tracks,
            "sync_shifted",
            44100 if scenario == "sync_multirate" else 48000,
            1,
            24,
            1.0,
            [{"time": 0.0, "gain_db": 0.0}],
            amplitude_dbfs=-12.0,
            frequency=700.0,
            inserted_lead_samples=extra,
        )
        return _finish(root, tracks, _base_config(), 1.0)
    if scenario == "sync_missing":
        _add_track(
            root,
            tracks,
            "marker_absent",
            48000,
            1,
            16,
            1.0,
            [{"time": 0.0, "gain_db": 0.0}],
            amplitude_dbfs=-9.0,
            add_marker=False,
        )
        return _finish(root, tracks, _base_config(), 1.0)
    if scenario == "gain_ramp":
        _add_track(
            root,
            tracks,
            "ramp_up",
            48000,
            1,
            16,
            1.0,
            [{"time": 0.0, "gain_db": -12.0}, {"time": 1.0, "gain_db": 0.0}],
            amplitude_dbfs=-20.0,
            frequency=440.0,
        )
        _add_track(
            root,
            tracks,
            "ramp_down",
            48000,
            1,
            24,
            1.0,
            [{"time": 0.0, "gain_db": 0.0}, {"time": 1.0, "gain_db": -12.0}],
            amplitude_dbfs=-20.0,
            frequency=440.0,
        )
        return _finish(root, tracks, _base_config(), 1.0)
    if scenario == "gain_bad_order":
        _add_track(
            root,
            tracks,
            "bad_order",
            48000,
            1,
            16,
            1.0,
            [
                {"time": 0.0, "gain_db": 0.0},
                {"time": 0.5, "gain_db": -3.0},
                {"time": 0.4, "gain_db": -6.0},
            ],
        )
        return _finish(root, tracks, _base_config(), 1.0)
    if scenario == "gain_out_of_range":
        _add_track(
            root,
            tracks,
            "out_of_range",
            48000,
            1,
            16,
            1.0,
            [{"time": 0.0, "gain_db": 0.0}, {"time": 0.5, "gain_db": 20.0}],
        )
        return _finish(root, tracks, _base_config(), 1.0)
    if scenario in {"smooth_step", "smooth_gradient", "smooth_clamped"}:
        if scenario == "smooth_step":
            nodes = [
                {"time": 0.0, "gain_db": -20.0},
                {"time": 0.5, "gain_db": -20.0},
                {"time": 0.5, "gain_db": 0.0},
                {"time": 1.0, "gain_db": 0.0},
            ]
            duration = 1.0
            config = _base_config(smooth_window_ms=10.0)
        elif scenario == "smooth_gradient":
            nodes = [
                {"time": 0.0, "gain_db": -20.0},
                {"time": 0.5, "gain_db": -10.0},
                {"time": 1.0, "gain_db": 0.0},
            ]
            duration = 1.0
            config = _base_config(smooth_window_ms=10.0)
        else:
            nodes = [
                {"time": 0.0, "gain_db": -20.0},
                {"time": 0.05, "gain_db": -20.0},
                {"time": 0.05, "gain_db": 0.0},
                {"time": 0.1, "gain_db": 0.0},
            ]
            duration = 0.1
            config = _base_config(smooth_window_ms=200.0)
        _add_track(
            root,
            tracks,
            "automation_track",
            48000,
            1,
            16,
            duration,
            nodes,
            amplitude_dbfs=-9.0,
            frequency=440.0,
        )
        return _finish(root, tracks, config, duration)
    if scenario in {"limiter_loud", "limiter_safe"}:
        automation = (
            [{"time": 0.0, "gain_db": 0.0}]
            if scenario == "limiter_loud"
            else [{"time": 0.0, "gain_db": -10.0}]
        )
        _add_track(
            root,
            tracks,
            "hot_sine",
            48000,
            1,
            32,
            1.0,
            automation,
            amplitude_dbfs=3.0,
            float_source=True,
        )
        config = _base_config(peak_limit_dbfs=-1.0)
        return _finish(root, tracks, config, 1.0)
    if scenario in {"loudness_pink", "loudness_conflict", "loudness_equal"}:
        _add_track(
            root,
            tracks,
            "pink_noise",
            48000,
            2,
            32,
            2.0,
            [{"time": 0.0, "gain_db": 0.0}, {"time": 2.0, "gain_db": 0.0}],
            signal="pink",
            amplitude_dbfs=-20.0,
            float_source=True,
            noise_seed=20261113,
        )
        config = _base_config(channels=2, loudness_target=-16.0)
        if scenario == "loudness_conflict":
            config["peak_limit_dbfs"] = -1.0
            config["loudness_target"] = 0.0
        elif scenario == "loudness_equal":
            # Measure the actual deterministic source at the output rate.
            from .wavio import read_wav

            samples, _ = read_wav(os.path.join(root, "sources", "pink_noise.wav"))
            measured = integrated_loudness(samples, 48000)
            config["loudness_target"] = measured
        return _finish(root, tracks, config, 2.0)
    if scenario == "format":
        _add_track(
            root,
            tracks,
            "mono_sine",
            44100,
            1,
            16,
            1.0,
            [{"time": 0.0, "gain_db": -3.0}],
            amplitude_dbfs=-9.0,
        )
        _add_track(
            root,
            tracks,
            "stereo_sine",
            48000,
            2,
            24,
            1.0,
            [{"time": 0.0, "gain_db": -6.0}],
            amplitude_dbfs=-12.0,
            frequency=600.0,
        )
        return _finish(root, tracks, _base_config(encoding="float32"), 1.0)
    if scenario == "failure_partial":
        _add_track(
            root,
            tracks,
            "zeta_track",
            48000,
            1,
            16,
            1.0,
            [{"time": 0.0, "gain_db": 0.0}],
            amplitude_dbfs=-9.0,
            frequency=500.0,
        )
        _add_track(
            root,
            tracks,
            "alpha_track",
            48000,
            1,
            16,
            1.0,
            [{"time": 0.0, "gain_db": 0.0}],
            amplitude_dbfs=-9.0,
            frequency=900.0,
        )
        _add_track(
            root,
            tracks,
            "mid_track",
            48000,
            1,
            16,
            1.0,
            [{"time": 0.0, "gain_db": 0.0}],
            amplitude_dbfs=-9.0,
            frequency=1200.0,
        )
        return _finish(root, tracks, _base_config(), 1.0)
    raise ValueError(f"unknown scenario {scenario!r}; choose from {', '.join(SCENARIOS)}")
