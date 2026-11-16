"""Beat-grid construction and millisecond-precision snapping."""

from dataclasses import dataclass
import math

from .errors import NotOnBeatError


@dataclass(frozen=True)
class SnappedTime:
    """Result of snapping an event time to the beat grid."""

    original_ms: float
    beat_index: int
    aligned_ms: float
    deviation_ms: float


class BeatGrid:
    """Beat boundaries derived from a constant BPM.

    Beat ``k`` starts at ``k * beat_period_seconds``. All computations use
    float64 so sub-millisecond jitter is preserved for the report while
    millisecond-precision inputs remain exact.
    """

    #: Events farther than half a beat from any boundary cannot be snapped.
    SNAP_LIMIT_BEATS = 0.5
    #: Control events must land on a beat boundary within ms precision.
    ON_BEAT_TOLERANCE_MS = 0.5

    def __init__(self, bpm, sample_rate):
        self.bpm = float(bpm)
        self.sample_rate = int(sample_rate)
        self.beat_period_s = 60.0 / self.bpm
        self.beat_period_ms = self.beat_period_s * 1000.0

    def beat_time_ms(self, beat_index):
        return beat_index * self.beat_period_ms

    def beat_time_s(self, beat_index):
        return beat_index * self.beat_period_s

    def snap(self, time_ms, *, require_on_beat=True):
        """Snap an event time to the nearest beat boundary.

        ``require_on_beat`` enforces the strict rule for control events: an
        event time must sit on a beat boundary with millisecond precision
        (tolerance 0.5 ms). The generic half-beat snap cap always applies.
        Ties at an exact half-beat position snap forward deterministically.
        """
        ms = float(time_ms)
        position = ms / self.beat_period_ms
        index = int(math.floor(position + 0.5))
        if index < 0:
            raise NotOnBeatError(
                f"event time {ms:g} ms is negative after snapping",
                detail={"time_ms": ms},
            )
        aligned_ms = self.beat_time_ms(index)
        deviation_ms = aligned_ms - ms
        if abs(deviation_ms) > self.beat_period_ms * self.SNAP_LIMIT_BEATS:
            raise NotOnBeatError(
                f"event at {ms:g} ms is {deviation_ms:+.3f} ms from beat "
                f"{index}, beyond the half-beat snap limit",
                detail={
                    "time_ms": ms,
                    "beat_index": index,
                    "deviation_ms": deviation_ms,
                },
            )
        if require_on_beat and abs(deviation_ms) > self.ON_BEAT_TOLERANCE_MS:
            raise NotOnBeatError(
                f"control event at {ms:g} ms is not on a beat boundary "
                f"(deviation {deviation_ms:+.3f} ms, tolerance "
                f"{self.ON_BEAT_TOLERANCE_MS:g} ms)",
                detail={
                    "time_ms": ms,
                    "beat_index": index,
                    "deviation_ms": deviation_ms,
                },
            )
        return SnappedTime(
            original_ms=ms,
            beat_index=index,
            aligned_ms=aligned_ms,
            deviation_ms=deviation_ms,
        )

    def sample_index_at_ms(self, time_ms):
        """Nearest output sample for a millisecond time."""
        return int(round(float(time_ms) * 0.001 * self.sample_rate))

    def total_samples(self, total_duration_ms):
        """Output length in samples, matching the chart duration within 1 sp."""
        seconds = float(total_duration_ms) / 1000.0
        return int(round(seconds * self.sample_rate))
