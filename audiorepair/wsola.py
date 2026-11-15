"""Anchor-preserving WSOLA time stretching.

Each source span between annotated syllable boundaries is rendered to the
exact target span length.  Within a span, synthesis frames overlap-add with
a waveform-similarity search around the nominal analysis position.  Keeping
the span boundaries as hard synthesis anchors prevents cumulative timing
drift at phoneme transitions.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .timeline import SegmentSpec

FRAME_SECONDS = 0.024
SEARCH_SECONDS = 0.004


def _resample_linear(x: np.ndarray, target_length: int) -> np.ndarray:
    source_length = x.shape[0]
    if source_length == target_length:
        return x.copy()
    if target_length <= 0:
        return np.zeros((0, x.shape[1]), dtype=np.float64)
    if source_length == 1:
        return np.repeat(x, target_length, axis=0)
    positions = np.linspace(0.0, float(source_length - 1), target_length)
    result = np.empty((target_length, x.shape[1]), dtype=np.float64)
    for channel in range(x.shape[1]):
        result[:, channel] = np.interp(
            positions, np.arange(source_length, dtype=np.float64), x[:, channel]
        )
    return result


def wsola_span(
    x: np.ndarray,
    target_length: int,
    sample_rate: int,
) -> np.ndarray:
    """Stretch ``x`` to exactly ``target_length`` frames with WSOLA."""
    audio = np.asarray(x, dtype=np.float64)
    source_length = audio.shape[0]
    if source_length == 0 or target_length <= 0:
        return np.zeros((max(0, target_length), audio.shape[1]), dtype=np.float64)
    if source_length == target_length:
        return audio.copy()

    frame_length = int(round(FRAME_SECONDS * sample_rate))
    frame_length -= frame_length & 1
    frame_length = max(32, frame_length)
    synthesis_hop = max(1, frame_length // 2)
    if source_length < frame_length or target_length < frame_length:
        return _resample_linear(audio, target_length)

    overlap = frame_length - synthesis_hop
    search_radius = max(1, int(round(SEARCH_SECONDS * sample_rate)))
    search_radius = min(search_radius, max(1, synthesis_hop // 2))
    positions = np.arange(frame_length, dtype=np.float64)
    window = 0.5 - 0.5 * np.cos(
        2.0 * np.pi * (positions + 0.5) / float(frame_length)
    )
    window = window[:, None]
    accumulator = np.zeros((target_length + frame_length, audio.shape[1]),
                           dtype=np.float64)
    weight = np.zeros((target_length + frame_length, 1), dtype=np.float64)
    similarity_audio = np.mean(audio, axis=1)
    sliding = np.lib.stride_tricks.sliding_window_view(
        similarity_audio, overlap
    )
    previous_source = 0
    output_position = 0
    first = True

    while output_position < target_length:
        nominal = int(
            round(float(output_position) * float(source_length) / target_length)
        )
        nominal = min(max(0, nominal), source_length - frame_length)
        source_position = nominal
        if not first:
            previous_tail = similarity_audio[
                previous_source + synthesis_hop : previous_source + frame_length
            ]
            lower = max(0, nominal - search_radius)
            upper = min(source_length - frame_length, nominal + search_radius)
            candidate_starts = np.arange(lower, upper + 1, dtype=np.int64)
            if candidate_starts.size:
                windows = sliding[candidate_starts]
                numerator = np.sum(windows * previous_tail[None, :], axis=1)
                window_energy = np.sum(windows * windows, axis=1)
                tail_energy = float(np.sum(previous_tail * previous_tail))
                denominator = np.sqrt(
                    np.maximum(window_energy * tail_energy, 1e-30)
                )
                similarity = numerator / denominator
                distance_penalty = (
                    0.001
                    * (candidate_starts.astype(np.float64) - nominal)
                    / float(max(1, search_radius))
                ) ** 2
                best = int(np.argmax(similarity - distance_penalty))
                source_position = int(candidate_starts[best])
        source_position = min(
            max(0, source_position), source_length - frame_length
        )
        frame = audio[source_position : source_position + frame_length]
        end = output_position + frame_length
        accumulator[output_position:end] += frame * window
        weight[output_position:end] += window
        previous_source = source_position
        output_position += synthesis_hop
        first = False

    result = accumulator[:target_length] / np.maximum(
        weight[:target_length], 1e-12
    )
    return result


def stretch_segment(
    samples: np.ndarray,
    spec: SegmentSpec,
    sample_rate: int,
) -> Tuple[np.ndarray, List[float]]:
    """Render one timeline segment and return its realised span ratios."""
    audio = np.asarray(samples, dtype=np.float64)
    source_points = (0,) + spec.source_anchors + (audio.shape[0],)
    target_points = (0,) + spec.expected_anchors + (spec.output_length,)
    pieces: List[np.ndarray] = []
    ratios: List[float] = []
    for index in range(len(source_points) - 1):
        source_start = source_points[index]
        source_end = source_points[index + 1]
        target_length = target_points[index + 1] - target_points[index]
        piece = wsola_span(
            audio[source_start:source_end], target_length, sample_rate
        )
        pieces.append(piece)
        ratios.append(target_length / float(source_end - source_start))
    if not pieces:
        return np.zeros((0, audio.shape[1]), dtype=np.float64), []
    rendered = np.concatenate(pieces, axis=0)
    if rendered.shape[0] != spec.output_length:
        # This should be impossible; retain an explicit hard invariant.
        raise AssertionError("WSOLA output length invariant violated")
    return rendered, ratios


def _frame_energy(
    audio: np.ndarray, frame_length: int, hop: int
) -> np.ndarray:
    mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
    if mono.size < frame_length:
        return np.asarray([float(np.mean(mono * mono))])
    cumulative = np.concatenate(
        ([0.0], np.cumsum(mono * mono, dtype=np.float64))
    )
    starts = np.arange(0, mono.size - frame_length + 1, hop, dtype=np.int64)
    return (cumulative[starts + frame_length] - cumulative[starts]) / frame_length


def _spectral_flux(audio: np.ndarray, frame_length: int, hop: int) -> np.ndarray:
    mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
    if mono.size < frame_length * 2:
        return np.zeros(1, dtype=np.float64)
    count = 1 + (mono.size - frame_length) // hop
    starts = np.arange(count, dtype=np.int64) * hop
    indices = starts[:, None] + np.arange(frame_length, dtype=np.int64)[None, :]
    window = np.hanning(frame_length).astype(np.float64)
    spectra = np.abs(np.fft.rfft(mono[indices] * window[None, :], axis=1))
    difference = np.diff(spectra, axis=0)
    flux = np.sum(np.maximum(difference, 0.0), axis=1)
    return np.concatenate(([0.0], flux))


def detect_boundaries(
    samples: np.ndarray,
    sample_rate: int,
    expected_samples: Sequence[int],
    search_ms: float = 35.0,
) -> List[dict]:
    """Measure nearby waveform boundaries from the quantised output.

    The primary detector locates the centre of a low-energy inter-phoneme
    gap.  When no significant local energy dip exists it falls back to the
    strongest positive spectral-flux transition.
    """
    audio = np.asarray(samples, dtype=np.float64)
    frame_length = max(8, int(round(0.002 * sample_rate)))
    hop = max(1, int(round(0.001 * sample_rate)))
    energy = _frame_energy(audio, frame_length, hop)
    flux = _spectral_flux(
        audio,
        max(16, int(round(0.006 * sample_rate))),
        max(1, int(round(0.002 * sample_rate))),
    )
    flux_hop = max(1, int(round(0.002 * sample_rate)))
    search_radius = int(round(search_ms * sample_rate / 1000.0))
    results: List[dict] = []

    for expected in expected_samples:
        expected = int(expected)
        energy_lower = max(0, (expected - search_radius) // hop)
        energy_upper = min(
            energy.size - 1, (expected + search_radius) // hop
        )
        detected = expected
        method = "energy-gap-centre"
        if energy_upper >= energy_lower and energy.size:
            local = energy[energy_lower : energy_upper + 1]
            minimum = float(np.min(local))
            local_median = float(np.median(local))
            low_mask = local <= minimum + 0.02 * max(local_median, 1e-12)
            low_indices = np.flatnonzero(low_mask)
            if (
                low_indices.size
                and local_median > 1e-10
                and minimum < 0.15 * local_median
            ):
                centre = int(
                    round(float(np.mean(low_indices))) + energy_lower
                )
                detected = centre * hop + frame_length // 2
            else:
                method = "spectral-flux"
                flux_centre = expected // flux_hop
                flux_lower = max(0, flux_centre - search_radius // flux_hop)
                flux_upper = min(flux.size - 1, flux_centre + search_radius // flux_hop)
                if flux_upper >= flux_lower and flux.size:
                    detected = (
                        flux_lower + int(np.argmax(flux[flux_lower : flux_upper + 1]))
                    ) * flux_hop + max(16, int(round(0.006 * sample_rate))) // 2
        error_samples = detected - expected
        results.append(
            {
                "expected_sample": expected,
                "actual_sample": int(detected),
                "error_samples": int(error_samples),
                "error_ms": round(
                    error_samples * 1000.0 / float(sample_rate), 6
                ),
                "detection_method": method,
            }
        )
    return results

