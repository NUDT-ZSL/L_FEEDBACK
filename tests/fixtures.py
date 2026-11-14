"""Shared scene-config builders used by the test suite."""

from __future__ import annotations

import copy


def make_scene(
    scene_id: str = "unit_scene",
    duration: float = 4.0,
    project_sr: int = 48000,
    target_lufs: float = -23.0,
    peak_limit: float = -1.0,
    tracks: list | None = None,
    true_peak: bool = False,
    tolerance: float = 0.5,
) -> dict:
    return {
        "id": scene_id,
        "project_sample_rate": project_sr,
        "duration": duration,
        "asset_dir": "assets",
        "output": f"out/{scene_id}.wav",
        "report": f"out/{scene_id}.report.json",
        "output_sample_width": 4,
        "loudness": {
            "target_lufs": target_lufs,
            "peak_limit_dbfs": peak_limit,
            "peak_shortfall_tolerance_lu": tolerance,
            "true_peak": true_peak,
        },
        "tracks": copy.deepcopy(tracks or []),
    }


def make_event(
    event_id: str,
    start: float,
    duration: float,
    frequency: float = 440.0,
    amplitude: float = 0.4,
    pan: float = 0.0,
    sample_rate: int | None = None,
    channels: int = 1,
    fades: dict | None = None,
    asset: str | None = None,
) -> dict:
    event = {
        "id": event_id,
        "start": start,
        "duration": duration,
        "frequency": frequency,
        "amplitude": amplitude,
        "pan": pan,
        "channels": channels,
    }
    if sample_rate is not None:
        event["sample_rate"] = sample_rate
    if fades is not None:
        event["fades"] = fades
    if asset is not None:
        event["asset"] = asset
    return event


def make_track(track_id: str, events: list, gain_db: float = 0.0, muted: bool = False) -> dict:
    return {"id": track_id, "gain_db": gain_db, "muted": muted, "events": events}


def write_batch(path, scenes: list[dict]) -> None:
    import json

    path.write_text(json.dumps({"scenes": scenes}), encoding="utf-8")
