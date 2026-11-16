"""Deterministic DSP primitives used by both generation and manifest replay."""

from __future__ import annotations

from collections import deque
import math
from typing import Any, Iterable, Mapping

import numpy as np

from .errors import SkillSFXError
from .recipe import (
    BusParams,
    CompressorParams,
    LimiterParams,
    NormalizationParams,
)


_LIMITER_QUANTIZATION_HEADROOM_DB = 0.02


def ms_to_frames(milliseconds: float, sample_rate: int) -> int:
    return int(math.floor(milliseconds * sample_rate / 1000.0 + 0.5))


def frames_to_ms(frames: int, sample_rate: int) -> float:
    return 1000.0 * frames / float(sample_rate)


def resampled_length(
    source_frames: int,
    source_sample_rate: int,
    target_sample_rate: int,
) -> int:
    if source_frames <= 0:
        return 0
    return max(
        1,
        int(
            math.floor(
                source_frames * target_sample_rate / source_sample_rate + 0.5
            )
        ),
    )


def resample_audio(
    audio: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float64)
    if source_sample_rate == target_sample_rate:
        return source.copy()
    if source.ndim != 2 or source.shape[0] == 0:
        raise SkillSFXError(
            "SFX-E2002",
            "cannot resample an empty or non-2D source",
            {"shape": list(source.shape)},
        )
    target_frames = resampled_length(
        source.shape[0],
        source_sample_rate,
        target_sample_rate,
    )
    source_frames = source.shape[0]
    spectrum = np.fft.rfft(source, axis=0)
    target_bins = target_frames // 2 + 1
    target_spectrum = np.zeros(
        (target_bins, source.shape[1]),
        dtype=np.complex128,
    )
    copy_bins = min(spectrum.shape[0], target_bins)
    target_spectrum[:copy_bins] = spectrum[:copy_bins]
    result = np.fft.irfft(target_spectrum, n=target_frames, axis=0)
    result *= target_frames / float(source_frames)
    return result


def _channel_error(
    message: str,
    context: Mapping[str, Any],
    reason: Any,
) -> SkillSFXError:
    merged = dict(context)
    merged["reason"] = reason
    return SkillSFXError("SFX-E2004", message, merged)


def resolve_channel_map(
    rule: Any,
    source_channels: int,
    target_channels: int,
    context: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(rule, str):
        mode = rule.strip().lower()
        if mode in ("identity", "stereo", "passthrough"):
            if source_channels != target_channels:
                raise _channel_error(
                    "channel map requires matching source and output channel counts",
                    context,
                    {
                        "mode": mode,
                        "source_channels": source_channels,
                        "target_channels": target_channels,
                    },
                )
            matrix = np.eye(target_channels, dtype=np.float64)
        elif mode in ("mono_to_stereo", "duplicate"):
            if source_channels != 1:
                raise _channel_error(
                    "mono-to-stereo mapping requires a mono source",
                    context,
                    {"mode": mode, "source_channels": source_channels},
                )
            matrix = np.ones((2, 1), dtype=np.float64)
            if target_channels != 2:
                raise _channel_error(
                    "duplicate mapping requires stereo output",
                    context,
                    {"mode": mode, "target_channels": target_channels},
                )
        elif mode in ("stereo_to_mono", "mono", "downmix"):
            if source_channels != 2:
                raise _channel_error(
                    "stereo-to-mono mapping requires a stereo source",
                    context,
                    {"mode": mode, "source_channels": source_channels},
                )
            if target_channels != 1:
                raise _channel_error(
                    "stereo-to-mono mapping requires mono output",
                    context,
                    {"mode": mode, "target_channels": target_channels},
                )
            matrix = np.array([[0.5, 0.5]], dtype=np.float64)
        elif mode == "sum_to_mono":
            if target_channels != 1:
                raise _channel_error(
                    "sum_to_mono requires mono output",
                    context,
                    {"mode": mode, "target_channels": target_channels},
                )
            matrix = np.ones((1, source_channels), dtype=np.float64)
        elif mode == "swap":
            if source_channels != 2 or target_channels != 2:
                raise _channel_error(
                    "swap mapping requires stereo input and output",
                    context,
                    {
                        "mode": mode,
                        "source_channels": source_channels,
                        "target_channels": target_channels,
                    },
                )
            matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        elif mode in ("left", "right"):
            if source_channels < (1 if mode == "left" else 2):
                raise _channel_error(
                    f"{mode} mapping requires the corresponding source channel",
                    context,
                    {"mode": mode, "source_channels": source_channels},
                )
            source_index = 0 if mode == "left" else 1
            matrix = np.zeros((target_channels, source_channels), dtype=np.float64)
            matrix[:, source_index] = 1.0
        else:
            raise _channel_error(
                "unknown channel mapping mode",
                context,
                {"mode": rule, "supported_modes": [
                    "identity",
                    "mono_to_stereo",
                    "duplicate",
                    "stereo_to_mono",
                    "downmix",
                    "sum_to_mono",
                    "swap",
                    "left",
                    "right",
                    "matrix",
                    "rules",
                ]},
            )
        descriptor = {"mode": mode, "matrix": matrix.tolist()}
        return matrix, descriptor

    if isinstance(rule, list):
        matrix = np.zeros((target_channels, source_channels), dtype=np.float64)
        normalized_rules: list[dict[str, Any]] = []
        for rule_index, item in enumerate(rule):
            if not isinstance(item, dict):
                raise _channel_error(
                    "channel mapping rule must be an object",
                    context,
                    {"rule_index": rule_index, "actual_type": type(item).__name__},
                )
            unknown = set(item) - {"source", "target", "gain"}
            if unknown:
                raise _channel_error(
                    "channel mapping rule contains an unknown field",
                    context,
                    {"rule_index": rule_index, "unknown_fields": sorted(unknown)},
                )
            if "source" not in item or "target" not in item:
                raise _channel_error(
                    "channel mapping rule is missing source or target",
                    context,
                    {"rule_index": rule_index},
                )
            source_index = item["source"]
            target_index = item["target"]
            gain = item.get("gain", 1.0)
            if (
                isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or isinstance(target_index, bool)
                or not isinstance(target_index, int)
            ):
                raise _channel_error(
                    "channel mapping indices must be integers",
                    context,
                    {"rule_index": rule_index},
                )
            if not 0 <= source_index < source_channels:
                raise _channel_error(
                    "channel mapping source index is out of range",
                    context,
                    {
                        "rule_index": rule_index,
                        "source_index": source_index,
                        "source_channels": source_channels,
                    },
                )
            if not 0 <= target_index < target_channels:
                raise _channel_error(
                    "channel mapping target index is out of range",
                    context,
                    {
                        "rule_index": rule_index,
                        "target_index": target_index,
                        "target_channels": target_channels,
                    },
                )
            if (
                isinstance(gain, bool)
                or not isinstance(gain, (int, float))
                or not math.isfinite(float(gain))
                or abs(float(gain)) > 64.0
            ):
                raise _channel_error(
                    "channel mapping gain is invalid",
                    context,
                    {"rule_index": rule_index, "gain": gain},
                )
            matrix[target_index, source_index] += float(gain)
            normalized_rules.append(
                {
                    "source": source_index,
                    "target": target_index,
                    "gain": float(gain),
                }
            )
        descriptor = {
            "mode": "rules",
            "rules": normalized_rules,
            "matrix": matrix.tolist(),
        }
        return matrix, descriptor

    if isinstance(rule, dict):
        mode = rule.get("mode")
        if mode == "rules":
            if set(rule) - {"mode", "rules"}:
                raise _channel_error(
                    "channel mapping object contains an unknown field",
                    context,
                    {"unknown_fields": sorted(set(rule) - {"mode", "rules"})},
                )
            return resolve_channel_map(
                rule.get("rules"),
                source_channels,
                target_channels,
                context,
            )
        if mode != "matrix":
            raise _channel_error(
                "channel mapping object mode must be 'matrix' or 'rules'",
                context,
                {"mode": mode},
            )
        if set(rule) - {"mode", "matrix"}:
            raise _channel_error(
                "channel mapping object contains an unknown field",
                context,
                {"unknown_fields": sorted(set(rule) - {"mode", "matrix"})},
            )
        raw_matrix = rule.get("matrix")
        if not isinstance(raw_matrix, list) or len(raw_matrix) != target_channels:
            raise _channel_error(
                "channel mapping matrix has the wrong target row count",
                context,
                {
                    "target_channels": target_channels,
                    "actual_rows": (
                        len(raw_matrix) if isinstance(raw_matrix, list) else None
                    ),
                },
            )
        matrix = np.zeros((target_channels, source_channels), dtype=np.float64)
        for target_index, row in enumerate(raw_matrix):
            if not isinstance(row, list) or len(row) != source_channels:
                raise _channel_error(
                    "channel mapping matrix has the wrong source column count",
                    context,
                    {
                        "target_row": target_index,
                        "source_channels": source_channels,
                        "actual_columns": len(row) if isinstance(row, list) else None,
                    },
                )
            for source_index, value in enumerate(row):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or abs(float(value)) > 64.0
                ):
                    raise _channel_error(
                        "channel mapping matrix contains an invalid coefficient",
                        context,
                        {
                            "target_row": target_index,
                            "source_column": source_index,
                            "value": value,
                        },
                    )
                matrix[target_index, source_index] = float(value)
        descriptor = {"mode": "matrix", "matrix": matrix.tolist()}
        return matrix, descriptor

    raise _channel_error(
        "channel_map must be a string, rule array, or matrix object",
        context,
        {"actual_type": type(rule).__name__},
    )


def apply_channel_map(audio: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(audio, dtype=np.float64) @ np.asarray(matrix, dtype=np.float64).T


def build_tail_harmonics(
    specification: Mapping[str, Any],
    frames: int,
    sample_rate: int,
    target_channels: int,
) -> np.ndarray:
    if frames <= 0:
        return np.zeros((0, target_channels), dtype=np.float64)
    t = np.arange(frames, dtype=np.float64) / float(sample_rate)
    attack_ms = float(specification["attack_ms"])
    decay_ms = float(specification["decay_ms"])
    attack = (
        np.clip(t / (attack_ms / 1000.0), 0.0, 1.0)
        if attack_ms > 0.0
        else np.ones(frames, dtype=np.float64)
    )
    envelope = attack * np.exp(-t / max(decay_ms / 1000.0, 1e-9))
    mono = np.zeros(frames, dtype=np.float64)
    for partial in specification["partials"]:
        frequency = float(specification["fundamental_hz"]) * float(partial["ratio"])
        amplitude = 10.0 ** (float(partial["gain_db"]) / 20.0)
        phase = float(partial["phase_rad"])
        mono += amplitude * np.sin(2.0 * np.pi * frequency * t + phase)
    mono *= envelope
    channel_gains = np.asarray(specification["channel_gains"], dtype=np.float64)
    if channel_gains.shape != (target_channels,):
        raise SkillSFXError(
            "SFX-E3001",
            "tail harmonic channel gain count does not match output channels",
            {
                "channel_gains": channel_gains.tolist(),
                "target_channels": target_channels,
            },
        )
    return mono[:, None] * channel_gains[None, :]


def remove_dc(audio: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    offsets = np.mean(audio, axis=0)
    result = audio - offsets[None, :]
    return result, {
        "method": "per_channel_mean_subtraction",
        "removed_dc": offsets.tolist(),
    }


def _soft_knee_gain_reduction_db(
    input_db: np.ndarray,
    threshold_db: float,
    ratio: float,
    knee_db: float,
) -> np.ndarray:
    result = np.zeros_like(input_db)
    if knee_db <= 0.0:
        active = input_db > threshold_db
        result[active] = (1.0 / ratio - 1.0) * (input_db[active] - threshold_db)
        return result
    lower = threshold_db - knee_db / 2.0
    upper = threshold_db + knee_db / 2.0
    above = input_db >= upper
    below = input_db <= lower
    knee = ~(above | below)
    result[above] = (1.0 / ratio - 1.0) * (input_db[above] - threshold_db)
    knee_delta = input_db[knee] - lower
    result[knee] = (
        (1.0 / ratio - 1.0)
        * knee_delta
        * knee_delta
        / (2.0 * knee_db)
    )
    return result


def soft_compress(
    audio: np.ndarray,
    sample_rate: int,
    params: CompressorParams | Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(params, Mapping):
        params = CompressorParams(
            threshold_db=float(params["threshold_db"]),
            ratio=float(params["ratio"]),
            knee_db=float(params["knee_db"]),
            attack_ms=float(params["attack_ms"]),
            release_ms=float(params["release_ms"]),
        )
    detector = np.max(np.abs(audio), axis=1)
    input_db = np.full(detector.shape, -200.0, dtype=np.float64)
    positive = detector > 0.0
    input_db[positive] = 20.0 * np.log10(detector[positive])
    desired = _soft_knee_gain_reduction_db(
        input_db,
        params.threshold_db,
        params.ratio,
        params.knee_db,
    )

    attack_samples = max(1e-9, params.attack_ms * sample_rate / 1000.0)
    release_samples = max(1e-9, params.release_ms * sample_rate / 1000.0)
    attack_coefficient = math.exp(-1.0 / attack_samples)
    release_coefficient = math.exp(-1.0 / release_samples)
    gain_reduction = np.empty_like(desired)
    previous = 0.0
    for index, target in enumerate(desired):
        coefficient = (
            attack_coefficient if target < previous else release_coefficient
        )
        previous = coefficient * previous + (1.0 - coefficient) * target
        gain_reduction[index] = previous

    gain = np.power(10.0, gain_reduction / 20.0)
    result = audio * gain[:, None]
    info = {
        "parameters": params.to_dict(),
        "detector": "stereo_linked_sample_peak",
        "gain_smoothing": "one_pole_attack_release_db",
        "max_gain_reduction_db": float(np.min(gain_reduction)),
        "active_frame_count": int(np.count_nonzero(gain_reduction < -1e-9)),
    }
    return result, info


def apply_tail_fade(
    audio: np.ndarray,
    sample_rate: int,
    requested_fade_ms: float,
    tail_margin_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    requested_frames = ms_to_frames(requested_fade_ms, sample_rate)
    margin = max(0, int(tail_margin_frames))
    applied_frames = min(requested_frames, margin)
    truncated = requested_frames > applied_frames
    result = audio.copy()
    if applied_frames > 0:
        indices = np.arange(1, applied_frames + 1, dtype=np.float64)
        curve = 0.5 * (
            1.0 + np.cos(np.pi * indices / float(applied_frames))
        )
        result[-applied_frames:, :] *= curve[:, None]
    warning = None
    if truncated:
        warning = {
            "code": "fade_truncated",
            "message": "requested tail fade exceeded available tail margin",
            "requested_fade_ms": requested_fade_ms,
            "applied_fade_ms": frames_to_ms(applied_frames, sample_rate),
            "tail_margin_ms": frames_to_ms(margin, sample_rate),
        }
    info = {
        "requested_fade_ms": requested_fade_ms,
        "applied_fade_ms": frames_to_ms(applied_frames, sample_rate),
        "requested_fade_frames": requested_frames,
        "applied_fade_frames": applied_frames,
        "tail_margin_frames": margin,
        "tail_margin_ms": frames_to_ms(margin, sample_rate),
        "curve": "raised_cosine",
        "truncated": truncated,
        "warning": warning,
    }
    return result, info


def oversample_for_true_peak(
    audio: np.ndarray,
    factor: int = 4,
) -> np.ndarray:
    frames = audio.shape[0]
    if frames == 0:
        return audio.copy()
    target_frames = frames * factor
    spectrum = np.fft.rfft(audio, axis=0)
    target_bins = target_frames // 2 + 1
    target_spectrum = np.zeros(
        (target_bins, audio.shape[1]),
        dtype=np.complex128,
    )
    copy_bins = min(spectrum.shape[0], target_bins)
    target_spectrum[:copy_bins] = spectrum[:copy_bins]
    oversampled = np.fft.irfft(target_spectrum, n=target_frames, axis=0)
    oversampled *= target_frames / float(frames)
    return oversampled


def true_peak_envelope(
    audio: np.ndarray,
    factor: int = 4,
) -> tuple[np.ndarray, float, int]:
    oversampled = oversample_for_true_peak(audio, factor)
    magnitude = np.abs(oversampled)
    peak = float(np.max(magnitude, initial=0.0))
    peak_position = int(np.argmax(magnitude)) if magnitude.size else 0
    frames = audio.shape[0]
    envelope = magnitude.reshape(frames, factor, audio.shape[1]).max(axis=(1, 2))
    return envelope, peak, peak_position


def peak_stats(audio: np.ndarray) -> dict[str, Any]:
    peak = float(np.max(np.abs(audio), initial=0.0))
    return {
        "peak_linear": peak,
        "peak_dbfs": 20.0 * math.log10(peak) if peak > 0.0 else -math.inf,
    }


def true_peak_stats(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    _, peak, position = true_peak_envelope(audio)
    return {
        "true_peak_linear": peak,
        "true_peak_dbtp": 20.0 * math.log10(peak) if peak > 0.0 else -math.inf,
        "true_peak_oversampled_index": position,
        "estimate_position_seconds": position / float(sample_rate * 4),
    }


def _k_weighting_coefficients(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    shelf_f0 = 1681.974450955533
    shelf_gain_db = 3.999843853973347
    shelf_q = 0.7071752369554196
    k = math.tan(math.pi * shelf_f0 / sample_rate)
    vh = 10.0 ** (shelf_gain_db / 20.0)
    vb = vh**0.4996667741545416
    denominator = 1.0 + k / shelf_q + k * k
    shelf_b = np.array(
        [
            (vh + vb * k / shelf_q + k * k) / denominator,
            2.0 * (k * k - vh) / denominator,
            (vh - vb * k / shelf_q + k * k) / denominator,
        ],
        dtype=np.float64,
    )
    shelf_a = np.array(
        [
            1.0,
            2.0 * (k * k - 1.0) / denominator,
            (1.0 - k / shelf_q + k * k) / denominator,
        ],
        dtype=np.float64,
    )

    highpass_f0 = 38.13547087602444
    highpass_q = 0.5003270373238773
    k = math.tan(math.pi * highpass_f0 / sample_rate)
    denominator = 1.0 + k / highpass_q + k * k
    highpass_b = np.array(
        [1.0 / denominator, -2.0 / denominator, 1.0 / denominator],
        dtype=np.float64,
    )
    highpass_a = np.array(
        [
            1.0,
            2.0 * (k * k - 1.0) / denominator,
            (1.0 - k / highpass_q + k * k) / denominator,
        ],
        dtype=np.float64,
    )
    return (shelf_b, shelf_a, highpass_b, highpass_a)  # type: ignore[return-value]


def _apply_biquad_fft(
    audio: np.ndarray,
    coefficients: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    b, a = coefficients
    frames = audio.shape[0]
    fft_length = 1
    while fft_length < max(2, frames * 2):
        fft_length <<= 1
    padded = np.zeros((fft_length, audio.shape[1]), dtype=np.float64)
    padded[:frames] = audio
    spectrum = np.fft.rfft(padded, axis=0)
    omega = 2.0 * np.pi * np.arange(spectrum.shape[0]) / fft_length
    z1 = np.exp(-1j * omega)
    z2 = z1 * z1
    response = (b[0] + b[1] * z1 + b[2] * z2) / (
        1.0 + a[1] * z1 + a[2] * z2
    )
    filtered = np.fft.irfft(
        spectrum * response[:, None],
        n=fft_length,
        axis=0,
    )
    return filtered[:frames]


def _k_weight(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    shelf_b, shelf_a, highpass_b, highpass_a = _k_weighting_coefficients(
        sample_rate
    )
    filtered = _apply_biquad_fft(audio, (shelf_b, shelf_a))
    filtered = _apply_biquad_fft(filtered, (highpass_b, highpass_a))
    return filtered


def measure_lufs(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    if audio.shape[0] == 0:
        return {
            "metric": "lufs",
            "value": -math.inf,
            "block_count": 0,
            "method": "itu_r_bs1770_4_k_weighting_fft_approximation",
        }
    weighted = _k_weight(audio, sample_rate)
    block_frames = max(1, int(round(0.4 * sample_rate)))
    hop_frames = max(1, int(round(0.1 * sample_rate)))
    frame_energy = np.sum(weighted * weighted, axis=1)
    blocks: list[float] = []
    start = 0
    while start + block_frames <= frame_energy.shape[0]:
        blocks.append(float(np.mean(frame_energy[start : start + block_frames])))
        start += hop_frames
    if not blocks:
        blocks = [float(np.mean(frame_energy))]
    block_loudness = -0.691 + 10.0 * np.log10(
        np.maximum(np.asarray(blocks, dtype=np.float64), 1e-30)
    )
    absolute_pass = block_loudness >= -70.0
    if np.any(absolute_pass):
        absolute_mean_energy = float(
            np.mean(np.asarray(blocks, dtype=np.float64)[absolute_pass])
        )
        relative_gate = (
            -0.691
            + 10.0 * math.log10(max(absolute_mean_energy, 1e-30))
            - 10.0
        )
        relative_pass = absolute_pass & (block_loudness >= relative_gate)
    else:
        relative_pass = absolute_pass
    if not np.any(relative_pass):
        value = -math.inf
    else:
        selected_energy = float(
            np.mean(np.asarray(blocks, dtype=np.float64)[relative_pass])
        )
        value = -0.691 + 10.0 * math.log10(max(selected_energy, 1e-30))
    return {
        "metric": "lufs",
        "value": value,
        "block_count": len(blocks),
        "gated_block_count": int(np.count_nonzero(relative_pass)),
        "method": "itu_r_bs1770_4_k_weighting_fft_approximation",
    }


def measure_rms_dbfs(audio: np.ndarray) -> dict[str, Any]:
    if audio.size == 0:
        value = -math.inf
    else:
        energy = float(np.mean(np.square(audio)))
        value = 10.0 * math.log10(energy) if energy > 0.0 else -math.inf
    return {
        "metric": "rms_dbfs",
        "value": value,
        "method": "mean_square_all_samples_channels",
    }


def measure_loudness(
    audio: np.ndarray,
    sample_rate: int,
    params: NormalizationParams | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(params, NormalizationParams):
        metric = params.metric
    else:
        metric = (
            "lufs" if params.get("target_lufs") is not None else "rms_dbfs"
        )
    if metric == "lufs":
        return measure_lufs(audio, sample_rate)
    return measure_rms_dbfs(audio)


def normalize_loudness(
    audio: np.ndarray,
    sample_rate: int,
    params: NormalizationParams | Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(params, Mapping):
        params = NormalizationParams(
            target_lufs=(
                float(params["target_lufs"])
                if params.get("target_lufs") is not None
                else None
            ),
            target_rms_dbfs=(
                float(params["target_rms_dbfs"])
                if params.get("target_rms_dbfs") is not None
                else None
            ),
            min_gain_db=float(params.get("min_gain_db", -24.0)),
            max_gain_db=float(params.get("max_gain_db", 24.0)),
        )
    before = measure_loudness(audio, sample_rate, params)
    target = params.target
    warning = None
    if not math.isfinite(float(before["value"])):
        desired_gain_db = 0.0
        warning = {
            "code": "silent_normalization_target",
            "message": "loudness measurement is silent; normalization gain set to 0 dB",
        }
    else:
        desired_gain_db = target - float(before["value"])
    applied_gain_db = min(
        params.max_gain_db,
        max(params.min_gain_db, desired_gain_db),
    )
    if abs(applied_gain_db - desired_gain_db) > 1e-9:
        warning = {
            "code": "normalization_gain_clamped",
            "message": "normalization gain was limited by configured bounds",
            "desired_gain_db": desired_gain_db,
            "applied_gain_db": applied_gain_db,
            "min_gain_db": params.min_gain_db,
            "max_gain_db": params.max_gain_db,
        }
    result = audio * (10.0 ** (applied_gain_db / 20.0))
    after = measure_loudness(result, sample_rate, params)
    return result, {
        "target_metric": params.metric,
        "target_value": target,
        "before": before,
        "after": after,
        "desired_gain_db": desired_gain_db,
        "applied_gain_db": applied_gain_db,
        "linear_gain": 10.0 ** (applied_gain_db / 20.0),
        "clamped": abs(applied_gain_db - desired_gain_db) > 1e-9,
        "warning": warning,
    }


def _sliding_past_minimum(values: np.ndarray, window: int) -> np.ndarray:
    frames = values.shape[0]
    if frames == 0:
        return values.copy()
    window = max(0, min(window, frames - 1))
    output = np.empty_like(values)
    queue: deque[int] = deque()
    for index in range(frames):
        while queue and values[queue[-1]] >= values[index]:
            queue.pop()
        queue.append(index)
        cutoff = index - window
        while queue and queue[0] < cutoff:
            queue.popleft()
        output[index] = values[queue[0]]
    return output


def _sliding_future_minimum(values: np.ndarray, window: int) -> np.ndarray:
    reversed_values = values[::-1]
    return _sliding_past_minimum(reversed_values, window)[::-1]


def _trigger_intervals(
    gain: np.ndarray,
    sample_rate: int,
) -> list[dict[str, Any]]:
    active = gain < 1.0 - 1e-9
    if not np.any(active):
        return []
    changes = np.diff(np.concatenate(([0], active.astype(np.int8), [0])))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    intervals: list[dict[str, Any]] = []
    for start, end in zip(starts, ends):
        intervals.append(
            {
                "start_frame": int(start),
                "end_frame_exclusive": int(end),
                "start_seconds": start / float(sample_rate),
                "end_seconds": end / float(sample_rate),
            }
        )
    return intervals


def lookahead_limit(
    audio: np.ndarray,
    sample_rate: int,
    params: LimiterParams | Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(params, Mapping):
        params = LimiterParams(
            ceiling_dbtp=float(params["ceiling_dbtp"]),
            lookahead_ms=float(params["lookahead_ms"]),
            attack_ms=float(params["attack_ms"]),
            release_ms=float(params["release_ms"]),
        )
    before = true_peak_stats(audio, sample_rate)
    internal_ceiling_dbtp = (
        params.ceiling_dbtp - _LIMITER_QUANTIZATION_HEADROOM_DB
    )
    ceiling = 10.0 ** (internal_ceiling_dbtp / 20.0)
    envelope, _, _ = true_peak_envelope(audio)
    required = np.ones_like(envelope)
    active = envelope > ceiling
    required[active] = ceiling / envelope[active]
    lookahead_frames = max(1, ms_to_frames(params.lookahead_ms, sample_rate))
    held = _sliding_future_minimum(required, lookahead_frames)

    attack_samples = max(1e-9, params.attack_ms * sample_rate / 1000.0)
    release_samples = max(1e-9, params.release_ms * sample_rate / 1000.0)
    attack_coefficient = math.exp(-1.0 / attack_samples)
    release_coefficient = math.exp(-1.0 / release_samples)
    gain = np.empty_like(held)
    previous = 1.0
    for index, target in enumerate(held):
        coefficient = (
            attack_coefficient if target < previous else release_coefficient
        )
        previous = coefficient * previous + (1.0 - coefficient) * target
        gain[index] = previous

    result = audio * gain[:, None]
    safety_trim_db = 0.0
    after_dynamic = true_peak_stats(result, sample_rate)
    if float(after_dynamic["true_peak_dbtp"]) > internal_ceiling_dbtp + 1e-9:
        trim = ceiling / max(float(after_dynamic["true_peak_linear"]), 1e-30)
        result *= trim
        safety_trim_db = 20.0 * math.log10(trim)
    final = true_peak_stats(result, sample_rate)
    intervals = _trigger_intervals(gain, sample_rate)
    info = {
        "parameters": params.to_dict(),
        "ceiling_linear": ceiling,
        "configured_ceiling_dbtp": params.ceiling_dbtp,
        "internal_ceiling_dbtp": internal_ceiling_dbtp,
        "quantization_headroom_db": _LIMITER_QUANTIZATION_HEADROOM_DB,
        "lookahead_frames": lookahead_frames,
        "before": before,
        "after_dynamic": after_dynamic,
        "final": final,
        "trigger_count": len(intervals),
        "trigger_intervals": intervals,
        "max_gain_reduction_db": (
            20.0 * math.log10(float(np.min(gain))) if np.min(gain) > 0 else -math.inf
        ),
        "safety_trim_db": safety_trim_db,
        "safety_trim_linear": 10.0 ** (safety_trim_db / 20.0),
    }
    return result, info


def clipping_sample_count(audio: np.ndarray) -> int:
    return int(np.count_nonzero(np.abs(audio) >= 1.0))


def peak_ceiling_satisfied(
    audio: np.ndarray,
    sample_rate: int,
    ceiling_dbtp: float,
) -> bool:
    stats = true_peak_stats(audio, sample_rate)
    return float(stats["true_peak_dbtp"]) <= ceiling_dbtp + 1e-6
