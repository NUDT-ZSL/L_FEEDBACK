from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

from .mpegts import (
    TS_PACKET_SIZE,
    iter_ts_packets,
    make_ts_packet,
    packet_pid,
    packetize_pes,
    packetize_section,
    rewrite_pmt_packet,
)


@dataclass
class AuxAction:
    action_id: str
    base_packet_index: int
    target_ms: float
    kind: str
    pid: int
    payload: bytes
    event_id: int | None = None
    detail: dict[str, Any] | None = None


def _write_packets(handle, packets: list[bytes]) -> None:
    for packet in packets:
        handle.write(packet)


def inject_actions(
    base_ts_path: str | Path,
    partial_path: str | Path,
    final_path: str | Path,
    actions: list[AuxAction],
    pmt_pid: int,
    new_pmt_section: bytes,
    state: dict[str, Any],
    save_state: Callable[[dict[str, Any]], None],
    checkpoint_interval: int = 128,
) -> dict[str, Any]:
    base_data = Path(base_ts_path).read_bytes()
    base_packets = list(iter_ts_packets(base_data))
    if not base_packets:
        raise ValueError("base TS has no packets")
    partial_path = Path(partial_path)
    final_path = Path(final_path)
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    groups: dict[int, list[AuxAction]] = {}
    for action in sorted(actions, key=lambda item: (item.base_packet_index, item.target_ms, item.action_id)):
        groups.setdefault(action.base_packet_index, []).append(action)

    next_base_index = int(state.get("next_base_index", 0))
    next_action_offset = int(state.get("next_action_offset", 0))
    partial_size = int(state.get("partial_size", 0))
    completed = set(state.get("completed_aux_actions", []))
    completed_events = set(state.get("completed_event_ids", []))
    cc_state = {int(key): int(value) for key, value in state.get("continuity_counters", {}).items()}

    resume = partial_path.is_file() and partial_size > 0
    if not resume:
        next_base_index = 0
        next_action_offset = 0
        partial_size = 0
        completed = set()
        completed_events = set()
        cc_state = {}
    handle = partial_path.open("r+b" if resume else "wb")
    if resume:
        handle.truncate(partial_size)
        handle.seek(partial_size)
    else:
        handle.seek(0)

    state.update(
        {
            "stage": "injecting",
            "partial_path": str(partial_path),
            "completed_transcode_segments": [
                {"name": "base_audio_video_mpegts", "path": str(base_ts_path)}
            ],
            "completed_event_ids": sorted(completed_events),
            "completed_aux_actions": sorted(completed),
            "next_base_index": next_base_index,
            "next_action_offset": next_action_offset,
            "partial_size": partial_size,
            "continuity_counters": {str(key): value for key, value in cc_state.items()},
        }
    )
    save_state(state)

    def checkpoint(base_index: int, action_offset: int) -> None:
        nonlocal partial_size
        handle.flush()
        partial_size = handle.tell()
        state.update(
            {
                "stage": "injecting",
                "next_base_index": base_index,
                "next_action_offset": action_offset,
                "partial_size": partial_size,
                "completed_event_ids": sorted(completed_events),
                "completed_aux_actions": sorted(completed),
                "continuity_counters": {str(key): value for key, value in cc_state.items()},
            }
        )
        save_state(state)

    def emit(action: AuxAction) -> None:
        nonlocal cc_state
        cc = cc_state.get(action.pid, 0)
        if action.kind == "pes":
            packets, cc = packetize_pes(action.pid, action.payload, cc)
            _write_packets(handle, packets)
        elif action.kind == "section":
            packets, cc = packetize_section(action.pid, action.payload, cc)
            _write_packets(handle, packets)
        else:
            raise ValueError(f"unknown action kind: {action.kind}")
        cc_state[action.pid] = cc

    base_index = next_base_index
    action_offset = next_action_offset
    try:
        while base_index < len(base_packets):
            group = groups.get(base_index, [])
            while action_offset < len(group):
                action = group[action_offset]
                if action.action_id not in completed:
                    emit(action)
                    completed.add(action.action_id)
                    if action.event_id is not None:
                        completed_events.add(action.event_id)
                action_offset += 1
                checkpoint(base_index, action_offset)
            packet = base_packets[base_index]
            if packet_pid(packet) == pmt_pid:
                handle.write(rewrite_pmt_packet(packet, new_pmt_section))
            else:
                handle.write(packet)
            base_index += 1
            action_offset = 0
            if base_index % checkpoint_interval == 0:
                checkpoint(base_index, 0)
        group = groups.get(len(base_packets), [])
        while action_offset < len(group):
            action = group[action_offset]
            if action.action_id not in completed:
                emit(action)
                completed.add(action.action_id)
                if action.event_id is not None:
                    completed_events.add(action.event_id)
            action_offset += 1
            checkpoint(len(base_packets), action_offset)
        handle.flush()
    except KeyboardInterrupt:
        checkpoint(base_index, action_offset)
        raise
    finally:
        handle.close()

    state.update(
        {
            "stage": "injected",
            "completed_event_ids": sorted(completed_events),
            "completed_aux_actions": sorted(completed),
            "next_base_index": len(base_packets),
            "next_action_offset": action_offset,
            "continuity_counters": {str(key): value for key, value in cc_state.items()},
        }
    )
    temp_final = final_path.with_suffix(final_path.suffix + ".replace")
    temp_final.parent.mkdir(parents=True, exist_ok=True)
    if temp_final.exists():
        temp_final.unlink()
    partial_path.replace(temp_final)
    temp_final.replace(final_path)
    state["partial_size"] = 0
    save_state(state)
    return state
