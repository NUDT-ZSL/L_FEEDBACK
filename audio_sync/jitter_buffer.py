"""Per-stream adaptive jitter buffer.

Frames are keyed by their global frame index (derived from the 32-bit
timestamp). The buffer absorbs reordering and delay jitter up to a configurable
target depth (2-10 frames). When full it applies backpressure by discarding the
oldest non-critical audio frame, and it enforces the 256 KB per-stream memory
budget.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from .constants import (
    EVENT_BACKPRESSURE,
    EVENT_LATE,
    EVENT_OVERFLOW,
    FRAME_SAMPLES,
    MAX_BUFFER_BYTES,
    MAX_BUFFER_DEPTH,
    MIN_BUFFER_DEPTH,
)


@dataclass
class BufferEvent:
    """A buffer lifecycle event tied to a global frame index."""

    frame_index: int
    kind: str
    detail: str = ""


class JitterBuffer:
    """Ordered, bounded store of float32 audio frames."""

    def __init__(
        self,
        stream_id: int,
        target_depth: int,
        max_depth: int = MAX_BUFFER_DEPTH,
    ) -> None:
        if not MIN_BUFFER_DEPTH <= target_depth <= MAX_BUFFER_DEPTH:
            raise ValueError(
                f"buffer depth must be in [{MIN_BUFFER_DEPTH},"
                f"{MAX_BUFFER_DEPTH}], got {target_depth}"
            )
        if not MIN_BUFFER_DEPTH <= max_depth <= MAX_BUFFER_DEPTH:
            raise ValueError(f"invalid max depth {max_depth}")
        if max_depth < target_depth:
            raise ValueError("max depth must be >= target depth")

        self.stream_id = stream_id
        self.max_depth = max_depth
        self._target_depth = target_depth
        self._frames: Dict[int, np.ndarray] = {}
        self._events: Deque[BufferEvent] = deque()
        self.next_playout_index: Optional[int] = None
        self._history_bytes = 0

    @property
    def target_depth(self) -> int:
        return self._target_depth

    def update_target_depth(self, depth: int) -> None:
        """Adapt the target depth, always within configured bounds."""
        self._target_depth = int(
            min(self.max_depth, max(MIN_BUFFER_DEPTH, depth))
        )

    @property
    def water_level(self) -> int:
        """Number of frames currently held (sampled at tick time)."""
        return len(self._frames)

    @property
    def memory_bytes(self) -> int:
        frame_bytes = len(self._frames) * FRAME_SAMPLES * 4  # float32
        return frame_bytes + self._history_bytes

    def set_history_bytes(self, history_bytes: int) -> None:
        """Allow the PLC history to be counted against the memory budget."""
        self._history_bytes = int(history_bytes)

    def drain_events(self) -> Tuple[BufferEvent, ...]:
        events = tuple(self._events)
        self._events.clear()
        return events

    def insert(
        self, frame_index: int, samples: np.ndarray, arrival_ms: float
    ) -> Optional[BufferEvent]:
        """Insert an arriving frame.

        Returns a :class:`BufferEvent` for duplicate / overflow / backpressure
        situations, otherwise ``None``. The receive path never blocks.
        """
        if (
            self.next_playout_index is not None
            and frame_index < self.next_playout_index
        ):
            event = BufferEvent(
                frame_index,
                EVENT_LATE,
                f"arrival_ms={arrival_ms:.2f} already_played",
            )
            self._events.append(event)
            return event

        if frame_index in self._frames:
            # Duplicate: keep the first copy, surface as backpressure signal.
            event = BufferEvent(
                frame_index,
                EVENT_BACKPRESSURE,
                "duplicate frame discarded",
            )
            self._events.append(event)
            return event

        overflow_event: Optional[BufferEvent] = None
        if len(self._frames) >= self.max_depth:
            oldest = min(self._frames)
            del self._frames[oldest]
            overflow_event = BufferEvent(
                oldest,
                EVENT_OVERFLOW,
                f"buffer_full={self.max_depth} oldest_dropped "
                f"arrival_ms={arrival_ms:.2f}",
            )
            self._events.append(overflow_event)
            # Backpressure is the runtime symptom of the overflow.
            self._events.append(
                BufferEvent(oldest, EVENT_BACKPRESSURE, "arrival exceeds drain")
            )

        self._frames[frame_index] = samples.astype(np.float32, copy=True)

        if self.memory_bytes > MAX_BUFFER_BYTES:
            # Defensive guard; normal depths stay far below this limit.
            raise MemoryError(
                f"stream {self.stream_id} jitter buffer exceeded "
                f"{MAX_BUFFER_BYTES} bytes"
            )
        return overflow_event

    def pop(self, frame_index: int) -> Tuple[Optional[np.ndarray], bool]:
        """Remove and return the frame for playout.

        Returns ``(samples, underflow)``. On underflow ``samples`` is ``None``
        and the caller runs PLC.
        """
        if self.next_playout_index is None:
            self.next_playout_index = frame_index
        frame = self._frames.pop(frame_index, None)
        self.next_playout_index = frame_index + 1
        if frame is None:
            self._events.append(
                BufferEvent(frame_index, "buffer_underflow", "missing at playout")
            )
            return None, True
        return frame, False

    def peek_indexes(self) -> Tuple[int, ...]:
        return tuple(sorted(self._frames))
