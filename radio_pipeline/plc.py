from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .audio import pcm_square_sum
from .config import AudioSettings, PlcSettings
from .jitter import FrameDecision


@dataclass
class PlcResult:
    pcm: np.ndarray
    pitch_frames: int
    comfort_frames: int
    muted_frames: int


def _decode(packet_bytes: bytes, frame_samples: int) -> np.ndarray:
    return np.frombuffer(packet_bytes, dtype="<i2", count=frame_samples).copy()


def _estimate_pitch_period(
    history: np.ndarray, sample_rate: int, min_hz: float, max_hz: float
) -> int:
    if history.size < 128:
        return max(1, int(round(sample_rate / 220.0)))
    minimum_lag = max(1, int(math.floor(sample_rate / max_hz)))
    maximum_lag = min(int(math.ceil(sample_rate / min_hz)), history.size // 2)
    if maximum_lag <= minimum_lag:
        return max(1, minimum_lag)
    search = history[-min(history.size, maximum_lag + 960) :].astype(np.float64)
    best_lag = minimum_lag
    best_score = -1.0
    for lag in range(minimum_lag, maximum_lag + 1):
        left = search[lag:]
        right = search[:-lag]
        if left.size < 96:
            continue
        denominator = math.sqrt(
            float(np.dot(left, left) * np.dot(right, right))
        ) + 1e-12
        score = float(np.dot(left, right)) / denominator
        if score > best_score:
            best_score = score
            best_lag = lag
    return max(1, best_lag)


def _pitch_frame(
    history: np.ndarray,
    period: int,
    frame_samples: int,
    run_index: int,
    gain: float,
) -> np.ndarray:
    if history.size == 0 or period <= 0:
        return np.zeros(frame_samples, dtype=np.float64)
    period = min(period, history.size)
    pattern = history[-period:].astype(np.float64)
    phase = (run_index * frame_samples) % period
    needed = phase + frame_samples
    repeats = int(math.ceil(needed / period))
    tiled = np.tile(pattern, repeats + 1)
    return tiled[phase : phase + frame_samples] * gain


def _comfort_frame(
    sequence: int,
    frame_samples: int,
    base_rms: float,
    gain: float,
    decay: float,
    seed: int,
) -> np.ndarray:
    if base_rms <= 0.0 or gain <= 0.0:
        return np.zeros(frame_samples, dtype=np.float64)
    mixed_seed = (seed ^ (sequence * 0x9E3779B1) ^ 0xC0FFEE) & 0xFFFFFFFF
    rng = np.random.default_rng(mixed_seed)
    noise = rng.standard_normal(frame_samples)
    smooth = np.empty_like(noise)
    accumulator = 0.0
    for index, value in enumerate(noise):
        accumulator = 0.22 * value + 0.78 * accumulator
        smooth[index] = accumulator
    smooth_rms = float(np.sqrt(np.mean(smooth * smooth))) + 1e-12
    smooth = smooth / smooth_rms
    energy_decay = np.linspace(1.0, decay, frame_samples, endpoint=True)
    return base_rms * gain * energy_decay * smooth


def render_plc(
    frames: list[FrameDecision],
    audio: AudioSettings,
    plc: PlcSettings,
    seed: int,
) -> PlcResult:
    frame_samples = audio.frame_samples
    rendered: list[np.ndarray] = []
    missing_flags: list[bool] = []
    fade_regions: list[list[dict[str, object]]] = [[] for _ in frames]
    recent_good: list[np.ndarray] = []
    pitch_frames = 0
    comfort_frames = 0
    muted_frames = 0

    index = 0
    while index < len(frames):
        decision = frames[index]
        if decision.packet is not None:
            rendered.append(_decode(decision.packet.payload, frame_samples))
            missing_flags.append(False)
            recent_good.append(rendered[-1])
            if len(recent_good) > 2:
                recent_good.pop(0)
            index += 1
            continue

        run_end = index
        while run_end < len(frames) and frames[run_end].packet is None:
            run_end += 1
        run_length = run_end - index
        history = (
            np.concatenate(recent_good[-2:])
            if recent_good
            else np.empty(0, dtype="<i2")
        )
        period = _estimate_pitch_period(
            history, audio.sample_rate, plc.pitch_min_hz, plc.pitch_max_hz
        )
        base_rms = (
            float(np.sqrt(np.mean(history.astype(np.float64) ** 2)))
            if history.size
            else 0.0
        )

        for offset in range(run_length):
            decision = frames[index + offset]
            if plc.strategy == "adaptive":
                strategy = "pitch" if run_length >= 2 else "comfort"
            else:
                strategy = plc.strategy
            aggression = (
                plc.conservative_aggression
                if decision.conservative
                else decision.plc_aggression
            )
            if strategy == "pitch":
                gain = aggression * (plc.pitch_decay ** offset)
                mono = _pitch_frame(history, period, frame_samples, offset, gain)
                pitch_frames += 1
            elif strategy == "comfort":
                gain = aggression * plc.comfort_noise_gain
                mono = _comfort_frame(
                    decision.sequence,
                    frame_samples,
                    base_rms,
                    gain,
                    plc.isolated_decay,
                    seed,
                )
                comfort_frames += 1
            else:
                mono = np.zeros(frame_samples, dtype=np.float64)
                muted_frames += 1

            rendered_frame = np.clip(
                np.rint(mono), -32768.0, 32767.0
            ).astype("<i2")
            rendered.append(rendered_frame)
            missing_flags.append(True)
            decision.plc_triggered = True
            decision.plc_strategy = strategy
            decision.plc_run_length = run_length
            decision.plc_run_index = offset
        index = run_end

    _apply_boundary_fades_dispatch(
        rendered, missing_flags, fade_regions, plc.fade_ms, audio.sample_rate, frame_samples
    )

    output = np.concatenate(rendered) if rendered else np.empty(0, dtype="<i2")
    for frame_index, decision in enumerate(frames):
        frame_pcm = output[
            frame_index * frame_samples : (frame_index + 1) * frame_samples
        ]
        decision.fade_regions = fade_regions[frame_index]
        decision.pcm_square_sum = pcm_square_sum(frame_pcm)

    return PlcResult(
        pcm=output,
        pitch_frames=pitch_frames,
        comfort_frames=comfort_frames,
        muted_frames=muted_frames,
    )


def _apply_boundary_fades_dispatch(
    rendered: list[np.ndarray],
    missing: list[bool],
    regions: list[list[dict[str, object]]],
    fade_ms: float,
    sample_rate: int,
    frame_samples: int,
) -> None:
    fade_samples = int(round(sample_rate * fade_ms / 1000.0))
    fade_samples = max(0, min(frame_samples, fade_samples))
    if fade_samples <= 0:
        return
    for boundary_index in range(len(rendered) - 1):
        if missing[boundary_index] == missing[boundary_index + 1]:
            continue
        count = fade_samples
        fade_out = np.linspace(1.0, 0.0, count, endpoint=True, dtype=np.float64)
        fade_in = np.linspace(0.0, 1.0, count, endpoint=True, dtype=np.float64)
        left = rendered[boundary_index][-count:].astype(np.float64)
        right = rendered[boundary_index + 1][:count].astype(np.float64)
        rendered[boundary_index][-count:] = np.rint(left * fade_out).astype("<i2")
        rendered[boundary_index + 1][:count] = np.rint(right * fade_in).astype("<i2")
        boundary_sample = (boundary_index + 1) * frame_samples
        region: dict[str, object] = {
            "boundary_sample": boundary_sample,
            "start_sample": boundary_sample - count,
            "end_sample": boundary_sample + count,
            "mode": "linear_fade_out_fade_in",
        }
        frame_with_plc = (
            boundary_index if missing[boundary_index] else boundary_index + 1
        )
        regions[frame_with_plc].append(region)
