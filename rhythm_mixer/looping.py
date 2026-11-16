"""Seamless periodic loop sources with short overlap-add cross-fades.

The loop point of a periodic source stays at its declared period (e.g. a
whole number of bars), so beat-aligned timing is unchanged. The final
``crossfade`` samples of each copy are overlap-added with the start of the
next copy using equal-power-free linear ramps, removing the waveform step
that would otherwise cause an audible click.
"""

import numpy as np


class PeriodicSource:
    """A clip repeated forever with a cross-fade at every loop point.

    The effective period is ``period = clip_length - crossfade``. For a
    requested global index ``n`` the contribution is

    ``sum_k w(n - k * period) * clip[n - k * period]``

    where ``w`` is 1 in the clip body and a linear ramp in the overlapping
    head/tail regions. At most two copies overlap at any sample.
    """

    def __init__(self, clip, crossfade_samples):
        clip = np.asarray(clip, dtype=np.float32)
        if clip.size == 0:
            raise ValueError("PeriodicSource requires a non-empty clip")
        length = clip.size
        crossfade_samples = int(crossfade_samples)
        if crossfade_samples < 0:
            raise ValueError("crossfade_samples must be non-negative")
        # Keep at least one fully preserved sample inside every copy.
        crossfade_samples = min(crossfade_samples, max(length - 2, 0))
        self.clip = clip
        self.length = length
        self.crossfade = crossfade_samples
        self.period = length - crossfade_samples

    def read(self, start, count):
        """Return ``count`` samples starting at absolute sample ``start``."""
        if count <= 0:
            return np.zeros(0, dtype=np.float32)
        end = start + count
        out = np.zeros(count, dtype=np.float64)
        period = self.period
        xfade = self.crossfade
        clip = self.clip

        first_copy = start // period - 1
        last_copy = end // period + 1
        base = np.arange(count, dtype=np.int64)
        for copy_index in range(int(first_copy), int(last_copy) + 1):
            copy_start = copy_index * period
            local = base + start - copy_start
            valid = (local >= 0) & (local < self.length)
            if not np.any(valid):
                continue
            local_valid = local[valid]
            weight = np.ones(local_valid.shape, dtype=np.float64)
            if xfade > 0:
                head = local_valid < xfade
                tail = local_valid >= self.length - xfade
                # Copy k head overlaps copy k-1 tail: ramp in.
                weight[head] = (local_valid[head] + 1) / xfade
                # Copy k tail overlaps copy k+1 head: ramp out.
                weight[tail] = (
                    (self.length - local_valid[tail] - 1) / xfade
                )
            out[valid] += weight * clip[local_valid].astype(np.float64)
        return out.astype(np.float32)


def build_layer_source(samples, crossfade_ms, sample_rate, beat_period_samples,
                       loop_enabled):
    """Build the absolute-timeline periodic source for one layer.

    With a declared musical loop the source period is forced to an exact
    number of bars on the beat grid; the raw material is tiled to fill one
    such period and then wrapped with the configured cross-fade. Without a
    declared loop the raw clip is simply repeated at its own length.
    """
    crossfade_samples = int(
        round(crossfade_ms * 0.001 * sample_rate)
    )
    base = PeriodicSource(samples, crossfade_samples)
    if not loop_enabled:
        return base

    # Musical period: an integer number of beats (bar grid supplied by caller
    # through beat_period_samples * beats_in_loop).
    period_samples = int(beat_period_samples)
    if period_samples <= 0:
        return base
    period_buffer = base.read(0, period_samples)
    return PeriodicSource(period_buffer, crossfade_samples)
