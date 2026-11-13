"""Deterministic local WAV and event-table sample generation."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .wavio import write_wav_pcm16


@dataclass(frozen=True)
class TrackPlan:
    track_id: int
    player: str
    rate: int
    start_ms: int
    end_ms: int
    amplitude: float
    silence_ms: list[list[int]]
    gaps_ms: list[list[int]]
    base_freq: float
    markers: list[dict[str, Any]]


CLEAN_PLANS = [
    TrackPlan(
        track_id=1,
        player="IGL",
        rate=44_100,
        start_ms=0,
        end_ms=12_000,
        amplitude=0.035,
        silence_ms=[[2_000, 2_700]],
        gaps_ms=[[6_500, 6_900]],
        base_freq=196.0,
        markers=[
            {"at_ms": 1_200, "type": "callout", "label": "open mid"},
            {"at_ms": 6_200, "type": "teamfight", "label": "mid contest"},
        ],
    ),
    TrackPlan(
        track_id=2,
        player="Entry",
        rate=48_000,
        start_ms=250,
        end_ms=11_750,
        amplitude=0.070,
        silence_ms=[[4_200, 4_800]],
        gaps_ms=[[7_800, 8_200]],
        base_freq=233.0,
        markers=[
            {"at_ms": 4_000, "type": "callout", "label": "hold angle"},
            {"at_ms": 8_500, "type": "teamfight", "label": "site hit"},
        ],
    ),
    TrackPlan(
        track_id=3,
        player="Support",
        rate=22_050,
        start_ms=600,
        end_ms=11_400,
        amplitude=0.120,
        silence_ms=[[3_300, 3_900]],
        gaps_ms=[[5_200, 5_700], [9_600, 10_000]],
        base_freq=261.0,
        markers=[
            {"at_ms": 5_000, "type": "callout", "label": "utility ready"},
            {"at_ms": 9_800, "type": "teamfight", "label": "retake"},
        ],
    ),
    TrackPlan(
        track_id=4,
        player="Analyst",
        rate=32_000,
        start_ms=0,
        end_ms=12_000,
        amplitude=0.022,
        silence_ms=[[1_500, 2_100], [7_000, 7_600]],
        gaps_ms=[[10_200, 10_600]],
        base_freq=174.0,
        markers=[
            {"at_ms": 3_000, "type": "callout", "label": "economy note"},
            {"at_ms": 10_800, "type": "teamfight", "label": "clutch"},
        ],
    ),
]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _synthesize_voice(
    *,
    frames: int,
    rate: int,
    amplitude: float,
    base_freq: float,
    rng: np.random.Generator,
    phase: float,
) -> np.ndarray:
    t = np.arange(frames, dtype=np.float64) / float(rate)
    carrier = (
        0.55 * np.sin(2.0 * np.pi * base_freq * t + phase)
        + 0.30 * np.sin(2.0 * np.pi * base_freq * 1.73 * t + 0.37)
        + 0.15 * np.sin(2.0 * np.pi * base_freq * 2.41 * t + 1.11)
    )
    carrier += 0.07 * rng.standard_normal(frames)
    peak = float(np.max(np.abs(carrier))) if frames else 0.0
    if peak > 0:
        carrier *= amplitude / peak
    return carrier


def _zero_intervals(
    samples: np.ndarray,
    rate: int,
    intervals: list[list[int]],
) -> None:
    for start_ms, end_ms in intervals:
        start = int(round(start_ms * rate / 1000.0))
        end = int(round(end_ms * rate / 1000.0))
        samples[start:end] = 0.0


def _scale_ms(value: int, session_ms: int, template_ms: int = 12_000) -> int:
    return int(round(value * session_ms / template_ms))


def _scale_plan(plan: TrackPlan, session_ms: int) -> TrackPlan:
    return replace(
        plan,
        start_ms=_scale_ms(plan.start_ms, session_ms),
        end_ms=_scale_ms(plan.end_ms, session_ms),
        silence_ms=[
            [_scale_ms(start, session_ms), _scale_ms(end, session_ms)]
            for start, end in plan.silence_ms
        ],
        gaps_ms=[
            [_scale_ms(start, session_ms), _scale_ms(end, session_ms)]
            for start, end in plan.gaps_ms
        ],
        markers=[
            {
                **marker,
                **(
                    {"at_ms": _scale_ms(int(marker["at_ms"]), session_ms)}
                    if "at_ms" in marker
                    else {}
                ),
            }
            for marker in plan.markers
        ],
    )


def _track_truth(
    plan: TrackPlan,
    *,
    rng: np.random.Generator,
    output_dir: Path,
    filename: str,
    actual_rate: int | None = None,
    declared_rate: int | None = None,
    declared_frames: int | None = None,
    missing: bool = False,
) -> dict[str, Any]:
    write_rate = actual_rate if actual_rate is not None else plan.rate
    interval_ms = plan.end_ms - plan.start_ms
    frames = int(round(interval_ms * write_rate / 1000.0))
    if not missing:
        samples = _synthesize_voice(
            frames=frames,
            rate=write_rate,
            amplitude=plan.amplitude,
            base_freq=plan.base_freq,
            rng=rng,
            phase=(plan.track_id * 0.37) % (2.0 * np.pi),
        )
        _zero_intervals(samples, write_rate, plan.silence_ms)
        _zero_intervals(samples, write_rate, plan.gaps_ms)
        write_wav_pcm16(output_dir / filename, samples, sample_rate_hz=write_rate)
    return {
        "track_id": plan.track_id,
        "player": plan.player,
        "file": filename,
        "sample_rate_hz": (
            declared_rate if declared_rate is not None else plan.rate
        ),
        "channels": 1,
        "frames": declared_frames if declared_frames is not None else frames,
        "write_start_ms": plan.start_ms,
        "write_end_ms": plan.end_ms,
        "silence_ms": plan.silence_ms,
        "gaps_ms": plan.gaps_ms,
        "markers": plan.markers,
    }


def _fault_tracks(
    output_dir: Path, seed: int, session_ms: int
) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 101)
    plan = _scale_plan(TrackPlan(
        5, "RateMismatch", 44_100, 1_000, 5_000, 0.05,
        [], [], 180.0, [{"at_ms": 2_000, "type": "callout", "label": "rate fault"}],
    ), session_ms)
    tracks.append(
        _track_truth(
            plan,
            rng=rng,
            output_dir=output_dir,
            filename="track_05_rate_mismatch.wav",
            actual_rate=48_000,
        )
    )

    plan = _scale_plan(TrackPlan(
        6, "Unsupported", 96_000, 0, 2_000, 0.05,
        [], [], 220.0, [{"at_ms": 1_000, "type": "callout", "label": "96k fault"}],
    ), session_ms)
    tracks.append(
        _track_truth(
            plan,
            rng=rng,
            output_dir=output_dir,
            filename="track_06_unsupported.wav",
            actual_rate=96_000,
            declared_rate=96_000,
        )
    )

    plan = _scale_plan(TrackPlan(
        7, "Overrun", 48_000, 10_500, 13_500, 0.05,
        [], [], 246.0, [{"at_ms": 11_000, "type": "teamfight", "label": "overrun"}],
    ), session_ms)
    tracks.append(
        _track_truth(
            plan,
            rng=rng,
            output_dir=output_dir,
            filename="track_07_overrun.wav",
        )
    )

    plan = _scale_plan(TrackPlan(
        8, "Missing", 16_000, 2_000, 6_000, 0.05,
        [], [], 300.0, [],
    ), session_ms)
    tracks.append(
        _track_truth(
            plan,
            rng=rng,
            output_dir=output_dir,
            filename="track_08_missing.wav",
            missing=True,
        )
    )

    plan = _scale_plan(TrackPlan(
        9, "ShortFile", 44_100, 3_000, 6_000, 0.05,
        [], [], 330.0, [],
    ), session_ms)
    actual_frames = int(
        round((plan.end_ms - plan.start_ms) * plan.rate / 1000.0)
    )
    truth = _track_truth(
        plan,
        rng=rng,
        output_dir=output_dir,
        filename="track_09_wrong_frames.wav",
        declared_frames=actual_frames + 480,
    )
    tracks.append(truth)
    return tracks


def generate_scenario(
    output_dir: str | Path,
    *,
    session_ms: int = 12_000,
    seed: int = 20_261_114,
    include_faults: bool = False,
) -> dict[str, Any]:
    """Generate deterministic mono WAV files and ``event_table.json``."""

    if not isinstance(session_ms, int) or session_ms <= 0:
        raise ValueError("session_ms must be a positive integer")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    generated_tracks: list[dict[str, Any]] = []
    for template in CLEAN_PLANS:
        plan = _scale_plan(template, session_ms)
        filename = f"track_{plan.track_id:02d}.wav"
        generated_tracks.append(
            _track_truth(
                plan,
                rng=rng,
                output_dir=output,
                filename=filename,
            )
        )
    if include_faults:
        generated_tracks.extend(_fault_tracks(output, seed, session_ms))

    template_events = [
        {"at_ms": 4_000, "type": "teamfight", "label": "mid contest"},
        {"at_ms": 5_000, "type": "callout", "label": "utility check"},
        {"at_ms": 8_500, "type": "teamfight", "label": "site hit"},
        {"at_ms": 10_800, "type": "teamfight", "label": "retake"},
    ]
    top_events = [
        {
            **event,
            "at_ms": _scale_ms(int(event["at_ms"]), session_ms),
        }
        for event in template_events
    ]
    table = {
        "schema_version": 1,
        "session_id": "local-deterministic-scrim",
        "session_ms": session_ms,
        "internal_sample_rate_hz": 48_000,
        "seed": seed,
        "tracks": generated_tracks,
        "events": top_events,
    }
    table_path = output / "event_table.json"
    _atomic_write_text(
        table_path,
        json.dumps(table, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return {
        "event_table": str(table_path),
        "output_dir": str(output),
        "track_count": len(generated_tracks),
    }
