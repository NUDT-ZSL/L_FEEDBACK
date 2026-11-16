"""In-project time stretching implementations.

Only NumPy primitives (FFT, slicing, matrix products) are used.  No external
audio-processing library performs the time-scale modification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .errors import LipAlignError


SUPPORTED_ALGORITHMS = ("wsola", "phase_vocoder")


@dataclass(frozen=True)
class StretchSettings:
    algorithm: str = "wsola"
    wsola_frame_ms: float = 40.0
    wsola_overlap_ms: float = 20.0
    wsola_search_ms: float = 10.0
    wsola_search_step: int = 2
    pv_frame_size: int = 2048
    pv_overlap: int = 4

    def validate(self) -> None:
        if self.algorithm not in SUPPORTED_ALGORITHMS:
            raise LipAlignError(
                "E_PARAMETER",
                "Unsupported stretch algorithm: {}".format(self.algorithm),
                exit_code=2,
            )
        if self.wsola_frame_ms <= 0 or self.wsola_overlap_ms <= 0:
            raise LipAlignError(
                "E_PARAMETER", "WSOLA frame/overlap must be positive", exit_code=2
            )
        if self.wsola_overlap_ms >= self.wsola_frame_ms:
            raise LipAlignError(
                "E_PARAMETER", "WSOLA overlap must be smaller than frame", exit_code=2
            )
        if self.wsola_search_ms < 0 or self.wsola_search_step < 1:
            raise LipAlignError(
                "E_PARAMETER",
                "WSOLA search_ms must be >= 0 and search_step >= 1",
                exit_code=2,
            )
        if self.pv_frame_size < 64 or self.pv_overlap < 2:
            raise LipAlignError(
                "E_PARAMETER", "Phase vocoder frame size/overlap is too small", exit_code=2
            )
        if self.pv_frame_size % self.pv_overlap != 0:
            raise LipAlignError(
                "E_PARAMETER",
                "Phase vocoder frame size must be divisible by overlap",
                exit_code=2,
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "wsola_frame_ms": self.wsola_frame_ms,
            "wsola_overlap_ms": self.wsola_overlap_ms,
            "wsola_search_ms": self.wsola_search_ms,
            "wsola_search_step": self.wsola_search_step,
            "pv_frame_size": self.pv_frame_size,
            "pv_overlap": self.pv_overlap,
        }

    def cache_identity(self) -> Dict[str, Any]:
        """Settings that can affect the selected algorithm's output only."""

        if self.algorithm == "wsola":
            return {
                "algorithm": self.algorithm,
                "frame_ms": self.wsola_frame_ms,
                "overlap_ms": self.wsola_overlap_ms,
                "search_ms": self.wsola_search_ms,
                "search_step": self.wsola_search_step,
            }
        return {
            "algorithm": self.algorithm,
            "frame_size": self.pv_frame_size,
            "overlap": self.pv_overlap,
        }


def stretch_factor_for(source_samples: int, target_samples: int) -> float:
    if source_samples <= 0:
        raise LipAlignError(
            "E_STRETCH", "Cannot stretch an empty source fragment", exit_code=9
        )
    return float(target_samples) / float(source_samples)


def wsola_time_stretch(
    samples: np.ndarray,
    target_samples: int,
    sample_rate: int,
    frame_ms: float = 40.0,
    overlap_ms: float = 20.0,
    search_ms: float = 10.0,
    search_step: int = 2,
) -> np.ndarray:
    """Waveform-similarity overlap-add time stretching."""

    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    target_len = int(target_samples)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float64)
    if len(x) == 0:
        raise LipAlignError("E_STRETCH", "Empty WSOLA input", exit_code=9)
    factor = stretch_factor_for(len(x), target_len)

    frame_len = max(16, int(round(frame_ms * sample_rate / 1000.0)))
    overlap = max(8, int(round(overlap_ms * sample_rate / 1000.0)))
    if overlap >= frame_len:
        overlap = max(1, frame_len // 2)
    hop_synthesis = frame_len - overlap
    search = max(0, int(round(search_ms * sample_rate / 1000.0)))
    step = max(1, int(search_step))
    analysis_hop = float(hop_synthesis) / max(factor, 1e-9)

    right_pad = frame_len + 2 * search + int(math.ceil(abs(analysis_hop))) + 16
    padded = np.concatenate((x, np.zeros(right_pad, dtype=np.float64)))
    window = np.hanning(frame_len).astype(np.float64)
    output = np.zeros(target_len, dtype=np.float64)
    weight = np.zeros(target_len, dtype=np.float64)

    previous_start = 0
    frame_index = 0
    max_full_start = len(padded) - frame_len
    while True:
        output_pos = frame_index * hop_synthesis
        if output_pos >= target_len:
            break
        nominal = int(round(frame_index * analysis_hop))
        nominal = max(0, min(nominal, max_full_start))

        if frame_index == 0:
            chosen = 0
        else:
            template_start = min(
                previous_start + hop_synthesis, max(0, max_full_start)
            )
            template = padded[template_start : template_start + overlap]
            if len(template) < overlap:
                template = np.pad(template, (0, overlap - len(template)))
            template = template - float(np.mean(template))

            candidate_min = max(0, nominal - search)
            candidate_max = min(max_full_start, nominal + search)
            if candidate_max < candidate_min:
                candidate_max = candidate_min
            region_end = min(len(padded), candidate_max + overlap)
            region = padded[candidate_min:region_end]
            if len(region) < overlap:
                region = np.pad(region, (0, overlap - len(region)))
            starts = np.arange(candidate_min, candidate_max + 1, step, dtype=np.int64)
            if starts.size == 0:
                starts = np.array([candidate_min], dtype=np.int64)
            windows = np.lib.stride_tricks.sliding_window_view(region, overlap)
            windows = windows[starts - candidate_min]
            centered = windows - np.mean(windows, axis=1, keepdims=True)
            correlations = centered @ template
            energies = np.sqrt(np.sum(centered * centered, axis=1) + 1e-12)
            scores = correlations / energies
            chosen = int(starts[int(np.argmax(scores))])

        frame = padded[chosen : chosen + frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        available = min(frame_len, target_len - output_pos)
        if available <= 0:
            break
        output[output_pos : output_pos + available] += (
            frame[:available] * window[:available]
        )
        weight[output_pos : output_pos + available] += window[:available]
        previous_start = chosen
        frame_index += 1

    nonzero = weight > 1e-7
    output[nonzero] /= weight[nonzero]
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def _principal_argument(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def phase_vocoder_time_stretch(
    samples: np.ndarray,
    target_samples: int,
    sample_rate: int,
    frame_size: int = 2048,
    overlap: int = 4,
) -> np.ndarray:
    """STFT phase-vocoder time stretching with phase advance correction."""

    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    target_len = int(target_samples)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float64)
    if len(x) == 0:
        raise LipAlignError(
            "E_STRETCH", "Empty phase-vocoder input", exit_code=9
        )
    factor = stretch_factor_for(len(x), target_len)
    frame_len = int(frame_size)
    if frame_len < 64 or overlap < 2 or frame_len % overlap != 0:
        raise LipAlignError(
            "E_STRETCH", "Invalid phase-vocoder frame configuration", exit_code=9
        )
    analysis_hop = frame_len // overlap
    synthesis_hop = max(1, int(round(analysis_hop * factor)))
    window = np.hanning(frame_len).astype(np.float64)
    padded = np.concatenate(
        (x, np.zeros(frame_len + analysis_hop * 2, dtype=np.float64))
    )
    bins = frame_len // 2 + 1
    frequencies = np.fft.rfftfreq(frame_len, d=1.0 / float(sample_rate))
    expected_advance = 2.0 * np.pi * frequencies * analysis_hop / float(sample_rate)

    output = np.zeros(target_len, dtype=np.float64)
    weight = np.zeros(target_len, dtype=np.float64)
    previous_phase = None
    accumulated_phase = np.zeros(bins, dtype=np.float64)
    frame_count = max(
        1,
        int(math.ceil(target_len / float(synthesis_hop))),
        int(math.ceil((len(x) + frame_len) / float(analysis_hop))),
    )
    for frame_index in range(frame_count):
        input_pos = frame_index * analysis_hop
        frame = padded[input_pos : input_pos + frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        spectrum = np.fft.rfft(frame * window)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        if previous_phase is None:
            accumulated_phase = phase.copy()
        else:
            deviation = _principal_argument(
                phase - previous_phase - expected_advance
            )
            true_advance = expected_advance + deviation
            accumulated_phase = accumulated_phase + true_advance * (
                float(synthesis_hop) / float(analysis_hop)
            )
        previous_phase = phase
        synthesis = np.fft.irfft(
            magnitude * np.exp(1j * accumulated_phase), n=frame_len
        )
        output_pos = frame_index * synthesis_hop
        if output_pos >= target_len:
            break
        available = min(frame_len, target_len - output_pos)
        output[output_pos : output_pos + available] += (
            synthesis[:available] * window[:available]
        )
        weight[output_pos : output_pos + available] += window[:available]

    nonzero = weight > 1e-7
    output[nonzero] /= weight[nonzero]
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def time_stretch(
    samples: np.ndarray,
    target_samples: int,
    sample_rate: int,
    settings: StretchSettings,
) -> np.ndarray:
    """Dispatch to an in-project time stretcher and enforce exact output length."""

    settings.validate()
    if target_samples < 1:
        raise LipAlignError("E_STRETCH", "Target sample count must be positive", exit_code=9)
    if settings.algorithm == "wsola":
        output = wsola_time_stretch(
            samples,
            target_samples,
            sample_rate,
            frame_ms=settings.wsola_frame_ms,
            overlap_ms=settings.wsola_overlap_ms,
            search_ms=settings.wsola_search_ms,
            search_step=settings.wsola_search_step,
        )
    elif settings.algorithm == "phase_vocoder":
        output = phase_vocoder_time_stretch(
            samples,
            target_samples,
            sample_rate,
            frame_size=settings.pv_frame_size,
            overlap=settings.pv_overlap,
        )
    else:
        raise LipAlignError(
            "E_PARAMETER",
            "Unsupported stretch algorithm: {}".format(settings.algorithm),
            exit_code=2,
        )
    if len(output) < target_samples:
        output = np.pad(output, (0, target_samples - len(output)))
    elif len(output) > target_samples:
        output = output[:target_samples]
    return np.asarray(output, dtype=np.float64)

