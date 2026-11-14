from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .config import JitterSettings, PlcSettings
from .packets import WirePacket


@dataclass
class FrameDecision:
    sequence: int
    timestamp_ms: int
    outcome: str
    packet: WirePacket | None
    buffer_water_ms: int
    missing: bool
    low_water_plc: bool
    fast_consume: bool
    conservative: bool
    plc_aggression: float
    degradation_event: dict[str, Any] | None = None
    plc_triggered: bool = False
    plc_strategy: str | None = None
    plc_run_length: int = 0
    plc_run_index: int = -1
    fade_regions: list[dict[str, Any]] = field(default_factory=list)
    pcm_square_sum: int = 0

    def to_log(self) -> dict[str, Any]:
        packet = self.packet
        return {
            "sequence": self.sequence,
            "timestamp_ms": self.timestamp_ms,
            "arrival_ms": packet.arrival_ms if packet is not None else None,
            "outcome": self.outcome,
            "source": "received" if packet is not None else self.outcome,
            "buffer_water_ms": self.buffer_water_ms,
            "plc_triggered": self.plc_triggered,
            "plc_strategy": self.plc_strategy,
            "plc_run_length": self.plc_run_length,
            "plc_run_index": self.plc_run_index,
            "fade_regions": self.fade_regions,
            "fast_consume": self.fast_consume,
            "low_water_plc": self.low_water_plc,
            "conservative": self.conservative,
            "plc_aggression": self.plc_aggression,
            "pcm_square_sum": self.pcm_square_sum,
        }


class JitterBuffer:
    """Clocked offline simulation of a packet jitter buffer."""

    def __init__(self, packets: int, config: JitterSettings, plc: PlcSettings):
        self.packet_count = packets
        self.config = config
        self.plc_config = plc
        self.frame_ms = 20
        self.buffer: dict[int, WirePacket] = {}
        self.next_sequence = 0
        self.playout_start_ms: float | None = None
        self.trim_ms = 0
        self.last_due_ms = 0.0
        self.max_water_ms = 0
        self.max_water_sequence = 0
        self.frames: list[FrameDecision] = []
        self.fast_consume_events: list[dict[str, Any]] = []
        self.degradation_events: list[dict[str, Any]] = []
        self.conservative = False
        self.effective_max_delay_ms = config.max_delay_ms
        self.buffer_depth_frames = max(1, int(math.ceil(config.max_delay_ms / self.frame_ms)))
        self.consecutive_loss = 0
        self.timeout_sequences: set[int] = set()
        self._mark_next_fast = False

    def water_level_ms(self) -> int:
        if not self.buffer or self.next_sequence >= self.packet_count:
            return 0
        highest = max(self.buffer)
        horizon = max(0, highest - self.next_sequence + 1)
        return int(horizon * self.frame_ms)

    def activate(self, first_arrival_ms: float) -> None:
        self.playout_start_ms = first_arrival_ms + self.config.min_delay_ms
        self.last_due_ms = self.playout_start_ms

    def insert(self, packet: WirePacket) -> dict[str, Any]:
        if self.playout_start_ms is None:
            self.activate(packet.arrival_ms)
        self.buffer[packet.sequence] = packet
        water = self.water_level_ms()
        if water > self.max_water_ms:
            self.max_water_ms = water
            self.max_water_sequence = self.next_sequence

        fast_event: dict[str, Any] | None = None
        if water > self.effective_max_delay_ms:
            before_trim = self.trim_ms
            deficit = water - self.effective_max_delay_ms
            step = min(max(1, self.config.fast_consume_step_ms), deficit)
            self.trim_ms += step
            self._mark_next_fast = True
            fast_event = {
                "sequence": packet.sequence,
                "arrival_ms": round(packet.arrival_ms, 6),
                "water_before_ms": water,
                "water_above_by_ms": deficit,
                "trim_before_ms": before_trim,
                "trim_after_ms": self.trim_ms,
                "action": "shorten_playout_lead",
            }
            self.fast_consume_events.append(fast_event)

        return {
            "water_ms": water,
            "fast_consume": fast_event is not None,
            "fast_consume_event": fast_event,
        }

    def due_ms(self, sequence: int) -> float:
        if self.playout_start_ms is None:
            return 0.0
        calculated = (
            self.playout_start_ms + sequence * self.frame_ms - self.trim_ms
        )
        return max(calculated, self.last_due_ms)

    def advance_to(self, arrival_ms: float) -> None:
        if self.playout_start_ms is None:
            return
        while (
            self.next_sequence < self.packet_count
            and self.due_ms(self.next_sequence) <= arrival_ms + 1e-9
        ):
            self._emit()

    def _activate_conservative(self, sequence: int) -> dict[str, Any]:
        old_max = self.effective_max_delay_ms
        self.conservative = True
        self.effective_max_delay_ms = self.config.conservative_max_delay_ms
        event = {
            "sequence": sequence,
            "timestamp_ms": sequence * self.frame_ms,
            "reason": "consecutive_loss_exceeds_buffer_depth",
            "consecutive_loss_frames": self.consecutive_loss,
            "buffer_depth_frames": self.buffer_depth_frames,
            "max_delay_before_ms": old_max,
            "max_delay_after_ms": self.effective_max_delay_ms,
            "plc_aggression_before": 1.0,
            "plc_aggression_after": self.plc_config.conservative_aggression,
        }
        self.degradation_events.append(event)
        return event

    def _emit(self) -> None:
        sequence = self.next_sequence
        water_before = self.water_level_ms()
        if water_before > self.max_water_ms:
            self.max_water_ms = water_before
            self.max_water_sequence = sequence
        packet = self.buffer.pop(sequence, None)
        degradation_event: dict[str, Any] | None = None
        if packet is None:
            self.consecutive_loss += 1
            if (
                self.consecutive_loss > self.buffer_depth_frames
                and not self.conservative
            ):
                degradation_event = self._activate_conservative(sequence)
            outcome = "timeout" if sequence in self.timeout_sequences else "lost"
        else:
            self.consecutive_loss = 0
            outcome = "received"

        decision = FrameDecision(
            sequence=sequence,
            timestamp_ms=sequence * self.frame_ms,
            outcome=outcome,
            packet=packet,
            buffer_water_ms=water_before,
            missing=packet is None,
            low_water_plc=packet is None
            and water_before < self.config.min_delay_ms,
            fast_consume=self._mark_next_fast,
            conservative=self.conservative,
            plc_aggression=(
                self.plc_config.conservative_aggression if self.conservative else 1.0
            ),
            degradation_event=degradation_event,
        )
        self._mark_next_fast = False
        self.frames.append(decision)
        self.next_sequence += 1
        self.last_due_ms = max(self.last_due_ms, self.due_ms(sequence))

    def mark_timeout(self, sequence: int) -> None:
        self.timeout_sequences.add(sequence)
        if 0 <= sequence < len(self.frames):
            frame = self.frames[sequence]
            if frame.outcome == "lost":
                frame.outcome = "timeout"
                frame.low_water_plc = frame.buffer_water_ms < self.config.min_delay_ms

    def flush(self) -> None:
        while self.next_sequence < self.packet_count:
            self._emit()

    def to_log(self) -> dict[str, Any]:
        return {
            "min_delay_ms": self.config.min_delay_ms,
            "max_delay_ms": self.config.max_delay_ms,
            "effective_max_delay_ms": self.effective_max_delay_ms,
            "timeout_ms": self.config.timeout_ms,
            "buffer_depth_frames": self.buffer_depth_frames,
            "max_buffer_delay_ms": self.max_water_ms,
            "max_buffer_delay_sequence": self.max_water_sequence,
            "trim_total_ms": self.trim_ms,
            "playout_start_ms": (
                round(self.playout_start_ms, 6)
                if self.playout_start_ms is not None
                else None
            ),
            "fast_consume_events": self.fast_consume_events,
            "degradation_events": self.degradation_events,
            "conservative_mode_activated": self.conservative,
        }
