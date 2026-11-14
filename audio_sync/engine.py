"""Single-threaded, virtual-clock audio pipeline engine.

A discrete event loop (no busy waiting) drives two event types per stream:

* packet arrivals scheduled at their simulated arrival time,
* 20 ms playout ticks that pop one frame from each jitter buffer.

Each stream owns an independent :class:`JitterBuffer` and :class:`PlcState`.
Frames are aligned by global frame index, concealed when missing, mixed, and
serialized to WAV plus a strictly time-ordered JSON Lines diagnostic log.
"""

from __future__ import annotations

import heapq
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, TextIO, Tuple

import numpy as np

from .constants import (
    EVENT_BACKPRESSURE,
    EVENT_LATE,
    EVENT_OVERFLOW,
    EVENT_RECOVERY,
    EVENT_UNDERFLOW,
    FRAME_DURATION_MS,
    FRAME_SAMPLES,
    MAX_BUFFER_DEPTH,
    MIN_BUFFER_DEPTH,
    MIN_STREAMS,
    MAX_STREAMS,
)
from .jitter_buffer import BufferEvent, JitterBuffer
from .mixer import Mixer
from .plc import PlcState
from .protocol import Packet, parse_packet_file, stream_id_from_path
from .signals import float_to_pcm_bytes, pcm_bytes_to_float
from .wavio import write_wav


@dataclass
class StreamState:
    """All per-stream runtime state."""

    stream_id: int
    buffer: JitterBuffer
    plc: PlcState
    packets: List[Packet]
    arrivals: List[float]
    next_arrival_index: int = 0
    min_seq: Optional[int] = None
    max_seq: Optional[int] = None
    backpressure_events: int = 0
    overflow_events: int = 0
    underflow_events: int = 0
    late_dropped: int = 0
    plc_frames: int = 0
    plc_methods: Dict[str, int] = field(default_factory=dict)
    clean_window: int = 0


@dataclass
class RunSummary:
    """Aggregate statistics printed by the CLI."""

    total_frames: int
    plc_frames: int
    average_buffer_water: float
    max_delay_ms: float
    backpressure_events: int
    stream_count: int
    wav_path: str
    log_path: str


class AudioPipeline:
    """Owns the event loop, buffers, PLC, mixer and outputs."""

    def __init__(
        self,
        packet_paths: Sequence[str],
        wav_path: str,
        log_path: str,
        buffer_depth: int,
        mixer_mode: str = "equal_power",
        frame_count: Optional[int] = None,
        meta: Optional[dict] = None,
        stderr_sink: Optional[TextIO] = None,
    ) -> None:
        if not MIN_BUFFER_DEPTH <= buffer_depth <= MAX_BUFFER_DEPTH:
            raise ValueError(
                f"buffer depth must be in [{MIN_BUFFER_DEPTH},"
                f"{MAX_BUFFER_DEPTH}], got {buffer_depth}"
            )
        if not MIN_STREAMS <= len(packet_paths) <= MAX_STREAMS:
            raise ValueError(
                f"stream count must be in [{MIN_STREAMS},{MAX_STREAMS}], "
                f"got {len(packet_paths)}"
            )

        self.packet_paths = list(packet_paths)
        self.wav_path = wav_path
        self.log_path = log_path
        self.buffer_depth = buffer_depth
        self.mixer = Mixer(mixer_mode)
        self.stderr_sink = stderr_sink or sys.stderr
        self.meta = meta or {}
        self.frame_count = frame_count
        self.states: Dict[int, StreamState] = {}
        self._water_samples: List[int] = []
        self._max_delay_ms = 0.0

    def _load_streams(self) -> None:
        schedule = self.meta.get("schedule", {}) if self.meta else {}
        for path in self.packet_paths:
            stream_id = stream_id_from_path(path)
            packets = parse_packet_file(
                path, f"stream_{stream_id}", self.stderr_sink
            )
            arrivals = self._arrival_schedule(stream_id, packets, schedule)
            buffer = JitterBuffer(
                stream_id,
                target_depth=self.buffer_depth,
                max_depth=self.buffer_depth,
            )
            state = StreamState(
                stream_id=stream_id,
                buffer=buffer,
                plc=PlcState(stream_id),
                packets=packets,
                arrivals=arrivals,
            )
            buffer.set_history_bytes(
                state.plc.history.size * state.plc.history.itemsize
            )
            self.states[stream_id] = state

        if self.frame_count is None:
            self.frame_count = self._infer_frame_count()

    def _arrival_schedule(
        self,
        stream_id: int,
        packets: Sequence[Packet],
        schedule: dict,
    ) -> List[float]:
        entries = schedule.get(str(stream_id))
        if entries:
            by_frame = {
                int(entry["frame_index"]): float(entry["arrival_ms"])
                for entry in entries
            }
            arrivals = [
                by_frame.get(
                    packet.frame_index,
                    packet.frame_index * FRAME_DURATION_MS,
                )
                for packet in packets
            ]
        else:
            # Fallback when no simulator metadata is present: nominal cadence.
            arrivals = [
                packet.frame_index * FRAME_DURATION_MS for packet in packets
            ]
        if arrivals:
            self._max_delay_ms = max(
                self._max_delay_ms,
                max(arrival - (index * FRAME_DURATION_MS)
                    for index, arrival in enumerate(arrivals)),
            )
        return arrivals

    def _infer_frame_count(self) -> int:
        highest = 0
        for state in self.states.values():
            for packet in state.packets:
                highest = max(highest, packet.frame_index + 1)
        return max(highest, 1)

    def run(self) -> RunSummary:
        self._load_streams()
        if self.frame_count is None or self.frame_count <= 0:
            raise ValueError("no playable frames found in packet directory")

        event_heap: List[Tuple[float, int, str, int]] = []
        sequence = 0
        for stream_id, state in self.states.items():
            if state.arrivals:
                heapq.heappush(
                    event_heap,
                    (state.arrivals[0], 0, sequence, "arrival", stream_id),
                )
                sequence += 1

        start_tick = max(self.buffer_depth, MIN_BUFFER_DEPTH)
        for tick in range(start_tick, self.frame_count + start_tick):
            heapq.heappush(
                event_heap,
                (tick * FRAME_DURATION_MS, 1, sequence, "tick", tick),
            )
            sequence += 1

        os.makedirs(os.path.dirname(os.path.abspath(self.wav_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)

        pcm_output = bytearray()
        log_handle = open(self.log_path, "w", encoding="utf-8")
        try:
            tick_index = start_tick
            while event_heap:
                event_time, _, _, kind, value = heapq.heappop(event_heap)
                if kind == "arrival":
                    self._handle_arrival(value, event_time, event_heap, sequence)
                    sequence += 1
                else:
                    frame_index = value - start_tick
                    record, mixed = self._handle_tick(frame_index)
                    pcm_output.extend(float_to_pcm_bytes(mixed))
                    log_handle.write(json.dumps(record, sort_keys=True) + "\n")
                    tick_index += 1
        finally:
            log_handle.close()

        write_wav(self.wav_path, pcm_output)
        total_plc = sum(state.plc_frames for state in self.states.values())
        backpressure = sum(
            state.backpressure_events for state in self.states.values()
        )
        average_water = (
            sum(self._water_samples) / len(self._water_samples)
            if self._water_samples
            else 0.0
        )
        return RunSummary(
            total_frames=self.frame_count,
            plc_frames=total_plc,
            average_buffer_water=round(average_water, 3),
            max_delay_ms=round(self._max_delay_ms, 3),
            backpressure_events=backpressure,
            stream_count=len(self.states),
            wav_path=self.wav_path,
            log_path=self.log_path,
        )

    def _handle_arrival(
        self,
        stream_id: int,
        event_time: float,
        event_heap: list,
        sequence: int,
    ) -> None:
        state = self.states[stream_id]
        index = state.next_arrival_index
        packet = state.packets[index]
        samples = pcm_bytes_to_float(packet.payload)
        event = state.buffer.insert(
            packet.frame_index, samples, event_time
        )
        if event is not None:
            self._classify_buffer_event(state, event)
        # Raw observed range of the 16-bit seq field (no wraparound folding:
        # the JSONL field documents which seq values actually arrived).
        state.min_seq = (
            packet.seq if state.min_seq is None
            else min(state.min_seq, packet.seq)
        )
        state.max_seq = (
            packet.seq if state.max_seq is None
            else max(state.max_seq, packet.seq)
        )

        state.next_arrival_index += 1
        if state.next_arrival_index < len(state.packets):
            next_time = state.arrivals[state.next_arrival_index]
            heapq.heappush(
                event_heap,
                (next_time, 0, sequence, "arrival", stream_id),
            )

    def _classify_buffer_event(
        self, state: StreamState, event: BufferEvent
    ) -> None:
        if event.kind == EVENT_OVERFLOW:
            state.overflow_events += 1
        elif event.kind == EVENT_UNDERFLOW:
            state.underflow_events += 1
        elif event.kind == EVENT_BACKPRESSURE:
            state.backpressure_events += 1
        elif event.kind == EVENT_LATE:
            state.late_dropped += 1

    def _handle_tick(
        self, frame_index: int
    ) -> Tuple[dict, np.ndarray]:
        frames: List[np.ndarray] = []
        stream_records: Dict[str, dict] = {}
        events_this_frame: List[str] = []

        ordered_ids = sorted(self.states)
        for stream_id in ordered_ids:
            state = self.states[stream_id]
            frame_seq = frame_index & 0xFFFF
            seq_covered = False
            samples, underflow = state.buffer.pop(frame_index)
            plc_flag = False
            plc_method = None
            recovery = False
            if underflow:
                result = state.plc.conceal()
                samples = result.frame
                plc_method = result.method
                if result.concealed:
                    plc_flag = True
                    state.plc_frames += 1
                    state.plc_methods[result.method] = (
                        state.plc_methods.get(result.method, 0) + 1
                    )
                state.underflow_events += 1
                if result.method == "silence":
                    events_this_frame.append(f"stream_{stream_id}:silence_run")
            else:
                seq_covered = True
                recovered = state.plc.accept_good_frame(samples)
                if recovered is not None:
                    samples = recovered
                    recovery = True
                    events_this_frame.append(
                        f"stream_{stream_id}:{EVENT_RECOVERY}"
                    )

            buffer_events = state.buffer.drain_events()
            for event in buffer_events:
                self._classify_buffer_event(state, event)
                events_this_frame.append(
                    f"stream_{stream_id}:{event.kind}"
                )
            self._adapt_depth(state, underflow)

            frames.append(samples)
            stream_records[f"stream_{stream_id}"] = {
                "seq_cover_min": frame_seq if seq_covered else None,
                "seq_cover_max": frame_seq if seq_covered else None,
                "observed_seq_min": state.min_seq,
                "observed_seq_max": state.max_seq,
                "frame_present": not underflow,
                "plc": plc_flag,
                "plc_method": plc_method,
                "plc_run_length": state.plc.consecutive_lost,
                "recovery": recovery,
                "buffer_water": state.buffer.water_level,
                "buffer_target": state.buffer.target_depth,
                "buffer_memory_bytes": state.buffer.memory_bytes,
            }
            self._water_samples.append(state.buffer.water_level)

        mixed_frame = self.mixer.mix(frames)
        record = {
            "frame_index": frame_index,
            "timestamp": frame_index * FRAME_SAMPLES,
            "time_ms": frame_index * FRAME_DURATION_MS,
            "streams": stream_records,
            "gains": list(mixed_frame.gains),
            "active": list(mixed_frame.active),
            "mix_peak": mixed_frame.peak,
            "events": events_this_frame,
        }
        return record, mixed_frame.mixed

    def _adapt_depth(self, state: StreamState, underflow: bool) -> None:
        """Closed-loop adaptation bounded by the configured depth.

        Underflow (drain faster than arrival) grows the target depth; a
        persistently deep buffer after a clean window shrinks it again.
        """
        if not hasattr(state, "_clean_window"):
            state._clean_window = 0  # type: ignore[attr-defined]
        if underflow:
            state.clean_window = 0
            state.buffer.update_target_depth(
                min(MAX_BUFFER_DEPTH, state.buffer.target_depth + 1)
            )
            return
        state.clean_window += 1
        if (
            state.clean_window >= 25
            and state.buffer.target_depth > self.buffer_depth
        ):
            state.buffer.update_target_depth(state.buffer.target_depth - 1)
            state.clean_window = 0
