"""Pitch-preserving time stretching with a streaming WSOLA implementation.

WSOLA (Waveform Similarity Overlap-Add) changes the duration of an audio
signal without altering its pitch:

* The signal is read through an *analysis* pointer that advances by a fixed
  analysis hop ``Sa``.
* A *synthesis* pointer advances by ``Ss = Sa / ratio``; when ``ratio > 1``
  the output is longer (slow-down), when ``ratio < 1`` it is shorter
  (speed-up).
* Before every new frame we search a small neighbourhood around the ideal
  analysis position and pick the offset whose windowed content best matches
  the previous output frame (normalised cross-correlation). This keeps
  successive frames phase-continuous and avoids the metallic artefacts of a
  naive overlap-add.

The module is fully NumPy based, deterministic for a given input and
parameters, and processes one frame at a time so memory stays bounded
regardless of the source file length.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np

from .constants import WSOLA_FRAME_MS, WSOLA_HOP_MS, WSOLA_MAX_SHIFT_MS


def stretch_samples(source: np.ndarray, ratio: float, sample_rate: int) -> np.ndarray:
    """Return ``source`` time-stretched by ``ratio``.

    ``source`` may be 1-D (mono) or 2-D with channels on the last axis.
    The returned array has the same number of channels and approximately
    ``len(source) * ratio`` samples.
    """
    if ratio <= 0:
        raise ValueError(f"stretch ratio must be positive, got {ratio}")
    if source.ndim == 1:
        return _stretch_mono(source, ratio, sample_rate)
    if source.ndim != 2:
        raise ValueError("source must be a 1-D or 2-D numpy array")
    channels = [
        _stretch_mono(source[:, channel], ratio, sample_rate)
        for channel in range(source.shape[1])
    ]
    return np.stack(channels, axis=1)


def target_sample_count(source_length: int, ratio: float) -> int:
    """Exact output length used by the pipeline (deterministic rounding)."""
    return int(round(source_length * ratio))


def _stretch_mono(signal: np.ndarray, ratio: float,
                  sample_rate: int) -> np.ndarray:
    frame_len = max(2, int(round(sample_rate * WSOLA_FRAME_MS / 1000.0)))
    analysis_hop = max(1, int(round(sample_rate * WSOLA_HOP_MS / 1000.0)))
    max_shift = max(1, int(round(sample_rate * WSOLA_MAX_SHIFT_MS / 1000.0)))

    # Hann window guarantees constant-overlap-add for hops that divide the
    # frame length; here it simply provides smooth cross-fades.
    window = np.hanning(frame_len).astype(np.float64, copy=False)
    padded = np.concatenate((
        np.zeros(frame_len, dtype=np.float64),
        signal.astype(np.float64, copy=False),
        np.zeros(frame_len * 2, dtype=np.float64),
    ))

    target_len = target_sample_count(len(signal), ratio)
    # Extra tail room because the last frame extends beyond the final sample.
    output = np.zeros(target_len + frame_len * 2, dtype=np.float64)
    weight = np.zeros_like(output)

    # Standard WSOLA pointer bookkeeping:
    #   analysis pointer advances by Sa (+ alignment offset)
    #   synthesis pointer advances by Ss = Sa / ratio
    # The alignment offset is compensated on the next analysis step but does
    # not move the synthesis grid, preserving the target time mapping.
    analysis_pos = 0.0
    synthesis_pos = 0.0
    synthesis_hop = analysis_hop / ratio
    natural_continuation: Optional[np.ndarray] = None

    while int(round(synthesis_pos)) < target_len:
        center = int(round(analysis_pos)) + frame_len  # account for padding
        frame, chosen_offset = _pick_frame(
            padded, center, frame_len, max_shift, window, natural_continuation
        )
        # Reference for the next search: the natural continuation of the
        # chosen frame, one *synthesis* hop later in source coordinates.
        # Aligning the next frame to this reference keeps the output
        # (assembled on a fixed Ss grid) phase continuous.
        natural_continuation = _safe_window(
            padded, center + chosen_offset + int(round(synthesis_hop)),
            frame_len, window
        )

        syn = int(round(synthesis_pos))
        end = min(len(output), syn + frame_len)
        take = end - syn
        if take > 0:
            output[syn:end] += frame[:take] * window[:take]
            weight[syn:end] += window[:take]

        analysis_pos += synthesis_hop + chosen_offset
        synthesis_pos += synthesis_hop
        if analysis_pos >= len(signal):
            break

    nonzero = weight > 1e-8
    output[nonzero] /= weight[nonzero]

    # Trim the padding head (first frame starts at sample 0 of real signal).
    result = output[:target_len]
    if len(result) < target_len:
        result = np.pad(result, (0, target_len - len(result)))
    return result.astype(signal.dtype, copy=False)


def _safe_window(padded: np.ndarray, center: int, frame_len: int,
                 window: np.ndarray) -> np.ndarray:
    """Windowed frame, zero-padded when ``center`` is near the edges."""
    start = center
    frame = np.zeros(frame_len, dtype=padded.dtype)
    lo = max(0, start)
    hi = min(len(padded), start + frame_len)
    if hi > lo:
        frame[lo - start:hi - start] = padded[lo:hi]
    return frame * window


def _pick_frame(padded: np.ndarray, center: int, frame_len: int,
                max_shift: int, window: np.ndarray,
                natural_continuation: Optional[np.ndarray]) -> Tuple[np.ndarray, int]:
    """Select the best-aligned frame around ``center`` via cross-correlation."""
    lowest = max(frame_len, center - max_shift)
    highest = min(len(padded) - frame_len, center + max_shift)
    if highest < lowest or natural_continuation is None:
        offset = 0
        start = min(max(center, frame_len), len(padded) - frame_len)
        return padded[start:start + frame_len].copy(), offset

    offsets = np.arange(lowest, highest + 1)
    reference = natural_continuation

    # Windowed cross-correlation via a *local* FFT. Only the neighbourhood
    # [lowest, highest+frame_len) is transformed, so the FFT size is fixed by
    # the search range, not the total signal length. Window weights are folded
    # into the kernel.
    search = padded[lowest:highest + frame_len]
    kernel = window * reference
    numerator = _local_correlation(search, kernel)[offsets - lowest]

    windows = np.lib.stride_tricks.sliding_window_view(
        padded, frame_len
    )[lowest:highest + 1]
    energy = (windows * windows) @ (window * window)
    reference_energy = float(reference @ reference)
    denominator = np.sqrt(np.maximum(energy * reference_energy, 1e-12))
    scores = numerator / denominator

    # Periodic signals produce several equally good offsets. WSOLA requires
    # the smallest displacement that reaches the (near-)best score, otherwise
    # argmax can lock onto a distant period and introduce pitch drift.
    best_score = float(np.max(scores))
    tolerance = 1e-6
    eligible = np.flatnonzero(scores >= best_score - tolerance)
    best = int(eligible[int(np.argmin(np.abs(offsets[eligible] - center)))])
    return padded[offsets[best]:offsets[best] + frame_len].copy(), \
        int(offsets[best] - center)


def _local_correlation(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Linear cross-correlation of ``kernel`` over a local signal window."""
    n = len(signal) + len(kernel) - 1
    fft_size = 1 << (n - 1).bit_length()
    spectrum = np.fft.rfft(signal, fft_size)
    kernel_spectrum = np.fft.rfft(kernel, fft_size)
    return np.fft.irfft(
        spectrum * np.conj(kernel_spectrum), fft_size
    )[:len(signal) - len(kernel) + 1]


class StreamingWSOLA:
    """Frame-by-frame WSOLA suitable for very long files.

    The API feeds fixed-size blocks through :meth:`push` and yields adapted
    blocks; internal state is at most a few analysis frames, keeping memory
    independent of total duration.
    """

    def __init__(self, ratio: float, sample_rate: int,
                 block_samples: Optional[int] = None) -> None:
        if ratio <= 0:
            raise ValueError("ratio must be positive")
        self.ratio = ratio
        self.sample_rate = sample_rate
        self.frame_len = max(2, int(round(sample_rate * WSOLA_FRAME_MS / 1000.0)))
        self.analysis_hop = max(1, int(round(sample_rate * WSOLA_HOP_MS / 1000.0)))
        self.max_shift = max(1, int(round(sample_rate * WSOLA_MAX_SHIFT_MS / 1000.0)))
        self.window = np.hanning(self.frame_len)
        self.block_samples = block_samples or int(sample_rate * 30)
        # Ring-style input buffer plus a base offset: consumed samples are
        # periodically trimmed so buffer size (and FFT cost) stays bounded.
        self._buffer = np.zeros(0, dtype=np.float64)
        self._consumed = 0          # samples already trimmed from the front
        self._analysis_pos = 0.0    # absolute position in the source stream
        self._synthesis_hop = self.analysis_hop / ratio
        self._previous: Optional[np.ndarray] = None
        self._trim_horizon = self.frame_len * 8

    def push(self, block: np.ndarray) -> Iterator[np.ndarray]:
        """Feed the next input block, yield any finished output blocks."""
        data = block.astype(np.float64, copy=False)
        if data.ndim == 2:
            # Streaming path is used per channel by the audio adapter.
            data = data[:, 0]
        self._buffer = np.concatenate((self._buffer, data))
        guard = self.frame_len * 2 + self.max_shift
        while self._available() >= guard + self.analysis_hop:
            frame = self._next_frame()
            if frame is not None:
                yield frame
            self._trim()

    def flush(self) -> Iterator[np.ndarray]:
        # Drain every frame that still fits inside the buffered signal.
        while True:
            if self._relative_center() + self.frame_len > len(self._buffer):
                break
            frame = self._next_frame()
            if frame is None:
                break
            yield frame
            self._trim()

    def _next_frame(self) -> Optional[np.ndarray]:
        center = self._relative_center()
        if center + self.frame_len > len(self._buffer):
            return None
        reference = None
        if self._previous is not None:
            reference = self._window_at(
                center + int(round(self._synthesis_hop))
            )
        frame, offset = _pick_frame(
            self._buffer, center, self.frame_len, self.max_shift,
            self.window, reference,
        )
        self._previous = frame
        self._analysis_pos += self._synthesis_hop + offset
        return frame * self.window

    def _window_at(self, center: int) -> np.ndarray:
        frame = np.zeros(self.frame_len, dtype=np.float64)
        lo = max(0, center)
        hi = min(len(self._buffer), center + self.frame_len)
        if hi > lo:
            frame[lo - center:hi - center] = self._buffer[lo:hi]
        return frame * self.window

    def _relative_center(self) -> int:
        return max(0, int(round(self._analysis_pos)) - self._consumed)

    def _available(self) -> int:
        return len(self._buffer) - self._relative_center()

    def _trim(self) -> None:
        center = self._relative_center()
        if center > self._trim_horizon:
            self._buffer = self._buffer[center:].copy()
            self._consumed += center
