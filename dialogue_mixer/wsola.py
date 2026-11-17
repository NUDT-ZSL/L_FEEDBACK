"""Pitch-preserving WSOLA time stretching."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class WSOLAParams:
    frame_size: int = 2048
    synthesis_hop: int = 512
    search_radius: int = 480
    algorithm: str = "wsola"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _hann_window(size: int) -> np.ndarray:
    index = np.arange(size, dtype=np.float64)
    return 0.5 - 0.5 * np.cos((2.0 * np.pi * index) / float(size))


def _safe_window_matrix(region: np.ndarray, width: int) -> np.ndarray:
    if region.size < width:
        region = np.pad(region, (0, width - region.size))
    return np.lib.stride_tricks.sliding_window_view(region, width)


def wsola_stretch(
    source: np.ndarray,
    target_length: int,
    sample_rate: int,
    params: WSOLAParams | None = None,
) -> np.ndarray:
    """Time-stretch *source* to exactly *target_length* samples.

    The overlap search aligns each new source frame with the natural
    continuation of the previously selected frame.  This preserves local
    waveform periodicity (and therefore pitch) while changing duration.
    """

    source = np.asarray(source, dtype=np.float64).reshape(-1)
    target_length = int(target_length)
    if target_length < 0:
        raise ValueError("target_length must be non-negative")
    if target_length == 0:
        return np.zeros(0, dtype=np.float64)
    if source.size == 0:
        return np.zeros(target_length, dtype=np.float64)
    if target_length == source.size:
        return source.copy()

    if params is None:
        frame_size = 2048
        if sample_rate < 16000:
            frame_size = 512
        search_radius = max(32, min(frame_size // 4, int(round(sample_rate * 0.01))))
        synthesis_hop = max(1, frame_size // 4)
        params = WSOLAParams(
            frame_size=frame_size,
            synthesis_hop=synthesis_hop,
            search_radius=search_radius,
        )

    frame_size = max(64, int(params.frame_size))
    if frame_size % 2:
        frame_size += 1
    synthesis_hop = max(1, min(int(params.synthesis_hop), frame_size // 2))
    search_radius = max(0, int(params.search_radius))
    stretch_ratio = float(target_length) / float(source.size)
    analysis_hop = max(1.0, float(synthesis_hop) / max(stretch_ratio, 1e-12))

    window = _hann_window(frame_size)
    pad = frame_size
    padded_source = np.pad(source, (pad, pad), mode="constant")
    output_length_with_tail = target_length + frame_size
    output = np.zeros(output_length_with_tail, dtype=np.float64)
    weight = np.zeros(output_length_with_tail, dtype=np.float64)
    source_size = source.size

    first_frame = -max(1, int(np.ceil(frame_size / float(synthesis_hop))))
    last_frame = int(np.ceil(target_length / float(synthesis_hop))) + 2
    previous_local_start = None

    for frame_index in range(first_frame, last_frame):
        output_start = frame_index * synthesis_hop
        if output_start >= target_length:
            break
        nominal_local_start = int(round(float(frame_index) * analysis_hop))

        minimum_local_start = -pad
        maximum_local_start = source_size
        if previous_local_start is None:
            chosen_local_start = min(
                maximum_local_start,
                max(minimum_local_start, nominal_local_start),
            )
        else:
            template_local_start = previous_local_start + synthesis_hop
            template = padded_source[
                pad + template_local_start : pad + template_local_start + synthesis_hop
            ]
            if template.size < synthesis_hop:
                template = np.pad(template, (0, synthesis_hop - template.size))

            low = nominal_local_start - search_radius
            high = nominal_local_start + search_radius
            low = max(-pad, low)
            high = min(source_size - synthesis_hop, high)
            if high < low:
                chosen_local_start = min(
                    maximum_local_start,
                    max(minimum_local_start, nominal_local_start),
                )
            else:
                region = padded_source[pad + low : pad + high + synthesis_hop]
                candidates = _safe_window_matrix(region, synthesis_hop)
                if candidates.shape[0] == 0:
                    chosen_local_start = min(
                        maximum_local_start,
                        max(minimum_local_start, nominal_local_start),
                    )
                else:
                    dots = candidates @ template
                    energy = np.sqrt(np.einsum("ij,ij->i", candidates, candidates))
                    scores = dots / (energy + 1e-12)
                    offset = int(np.argmax(scores))
                    chosen_local_start = low + offset
                    # Very weak candidates are still allowed, but avoid a
                    # spurious match in near-silence.
                    if not np.isfinite(scores[offset]):
                        chosen_local_start = min(
                            maximum_local_start,
                            max(minimum_local_start, nominal_local_start),
                        )

        frame = padded_source[
            pad + chosen_local_start : pad + chosen_local_start + frame_size
        ]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))

        start_in_output = output_start
        end_in_output = start_in_output + frame_size
        clipped_start = max(0, start_in_output)
        clipped_end = min(output_length_with_tail, end_in_output)
        if clipped_end > clipped_start:
            local_frame_start = clipped_start - start_in_output
            local_frame_end = local_frame_start + (clipped_end - clipped_start)
            output[clipped_start:clipped_end] += (
                frame[local_frame_start:local_frame_end]
                * window[local_frame_start:local_frame_end]
            )
            weight[clipped_start:clipped_end] += window[
                local_frame_start:local_frame_end
            ]
        previous_local_start = chosen_local_start

    normalized = np.zeros_like(output)
    valid = weight > 1e-9
    normalized[valid] = output[valid] / weight[valid]
    stretched = normalized[:target_length]
    if stretched.size < target_length:
        stretched = np.pad(stretched, (0, target_length - stretched.size))

    # A short crossfade at both edges prevents the window normalization from
    # producing an audible step when the source is not padded by the caller.
    edge = min(64, target_length // 8)
    if edge > 1 and source.size > edge:
        fade_in = np.linspace(0.0, 1.0, edge, endpoint=True)
        fade_out = fade_in[::-1]
        stretched[:edge] = stretched[:edge] * fade_in + source[:edge] * (1.0 - fade_in)
        stretched[-edge:] = (
            stretched[-edge:] * fade_out + source[-edge:] * (1.0 - fade_out)
        )
    if not np.all(np.isfinite(stretched)):
        raise ValueError("WSOLA produced non-finite samples")
    return stretched

