"""Multi-track mixing engine with timeline alignment, pans, fades and mapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import EventSpec, SceneConfig, TrackSpec
from .dsp import db_to_gain, equal_power_pan, fade_curve
from .errors import ErrorCode, MixerError
from .resampler import resample_mono
from .wavio import read_wav


def _frame(time_seconds: float, sample_rate: int) -> int:
    return int(round(time_seconds * sample_rate))


def _resolve_asset(base_dir: Path, scene: SceneConfig, event: EventSpec, asset_dir: Path) -> Path:
    if event.asset:
        candidate = Path(event.asset)
        return candidate if candidate.is_absolute() else base_dir / candidate
    safe = event.event_id.replace("/", "_").replace("\\", "_")
    return asset_dir / f"{scene.scene_id}_{safe}.wav"


def _apply_fades(
    stereo: np.ndarray,
    event: EventSpec,
    project_sr: int,
    start_frame: int,
    crossfade_frames: int,
) -> list[dict]:
    """Apply fade-in/out (and crossfade tails) and return report fade entries."""
    frame_count = stereo.shape[0]
    entries: list[dict] = []
    fades = event.fades

    # A crossfade with the previous same-track event overrides the lead-in.
    fade_in_frames = (
        crossfade_frames if crossfade_frames > 0 else _frame(fades.fade_in, project_sr)
    )
    fade_in_curve = fades.crossfade_curve if crossfade_frames > 0 else fades.curve
    fade_in_frames = min(fade_in_frames, frame_count)
    if fade_in_frames > 0:
        envelope = fade_curve(fade_in_frames, fade_in_curve, fade_in=True)
        stereo[:fade_in_frames] *= envelope[:, None]
        entries.append(
            {
                "kind": "crossfade_in" if crossfade_frames > 0 else "fade_in",
                "curve": fade_in_curve,
                "event_relative_start_frame": 0,
                "event_relative_end_frame": fade_in_frames - 1,
                "project_start_frame": start_frame,
                "project_end_frame": start_frame + fade_in_frames - 1,
                "frame_count": fade_in_frames,
            }
        )

    fade_out_frames = _frame(fades.fade_out, project_sr)
    fade_out_frames = min(fade_out_frames, frame_count - fade_in_frames)
    if fade_out_frames > 0:
        envelope = fade_curve(fade_out_frames, fades.curve, fade_in=False)
        stereo[-fade_out_frames:] *= envelope[:, None]
        entries.append(
            {
                "kind": "fade_out",
                "curve": fades.curve,
                "event_relative_start_frame": frame_count - fade_out_frames,
                "event_relative_end_frame": frame_count - 1,
                "project_start_frame": start_frame + frame_count - fade_out_frames,
                "project_end_frame": start_frame + frame_count - 1,
                "frame_count": fade_out_frames,
            }
        )
    return entries


def _validate_crossfades(scene: SceneConfig) -> dict[str, int]:
    """Return mapping event_id -> crossfade frames in (toward the next event)."""
    crossfade_in: dict[str, int] = {}
    sr = scene.project_sample_rate
    for track in scene.tracks:
        ordered = sorted(track.events, key=lambda event: event.start)
        for previous, current in zip(ordered, ordered[1:]):
            frames = _frame(current.fades.crossfade, sr)
            if frames <= 0:
                continue
            gap = current.start - (previous.start + previous.duration)
            # Negative gap means the events physically overlap, which is the
            # classic crossfade layout; the overlap must not exceed the fade.
            overlap_frames = max(0, _frame(-gap, sr))
            if gap < 0 and overlap_frames > frames:
                raise MixerError(
                    ErrorCode.OUT_OF_RANGE,
                    f"events {previous.event_id!r} and {current.event_id!r} overlap by "
                    f"{overlap_frames} frames, beyond the {frames}-frame crossfade",
                    {
                        "previous": previous.event_id,
                        "current": current.event_id,
                        "overlap_frames": overlap_frames,
                        "crossfade_frames": frames,
                    },
                )
            previous_frames = _frame(previous.duration, sr)
            current_frames = _frame(current.duration, sr)
            if frames > previous_frames or frames > current_frames:
                raise MixerError(
                    ErrorCode.OUT_OF_RANGE,
                    f"crossfade of {frames} frames exceeds an event's duration",
                    {
                        "previous": previous.event_id,
                        "current": current.event_id,
                        "crossfade_frames": frames,
                    },
                )
            crossfade_in[current.event_id] = frames
    return crossfade_in


def mix_scene(scene: SceneConfig, base_dir: str | Path) -> tuple[np.ndarray, dict]:
    """Render the stereo mix (pre-normalisation) and the detailed report data."""
    base = Path(base_dir)
    asset_dir = (
        base / scene.asset_dir if not Path(scene.asset_dir).is_absolute() else Path(scene.asset_dir)
    )
    project_frames = _frame(scene.duration, scene.project_sample_rate)
    stereo_bus = np.zeros((project_frames, 2), dtype=np.float64)
    crossfade_in = _validate_crossfades(scene)

    event_reports: list[dict] = []
    track_reports: list[dict] = []

    for track in scene.tracks:
        track_gain = db_to_gain(track.gain_db)
        track_reports.append(
            {
                "track_id": track.track_id,
                "gain_db": track.gain_db,
                "gain_linear": track_gain,
                "muted": track.muted,
                "event_ids": [event.event_id for event in track.events],
            }
        )
        for event in sorted(track.events, key=lambda item: item.start):
            rendered, report = _render_event(
                scene=scene,
                track=track,
                event=event,
                base_dir=base,
                asset_dir=asset_dir,
                track_gain=track_gain,
                crossfade_frames=crossfade_in.get(event.event_id, 0),
            )
            start = report["project_start_frame"]
            end = start + rendered.shape[0]
            if end > project_frames:
                raise MixerError(
                    ErrorCode.OUT_OF_RANGE,
                    f"event {event.event_id!r} rendered past the project end",
                    {"event_id": event.event_id, "end_frame": end, "project_frames": project_frames},
                )
            if not track.muted:
                stereo_bus[start:end] += rendered
            event_reports.append(report)

    peak = float(np.max(np.abs(stereo_bus))) if stereo_bus.size else 0.0
    report = {
        "project_sample_rate": scene.project_sample_rate,
        "project_frames": project_frames,
        "duration_seconds": scene.duration,
        "tracks": track_reports,
        "events": event_reports,
        "pre_normalisation_peak_linear": peak,
        "pre_normalisation_peak_dbfs": (
            20.0 * np.log10(peak) if peak > 1e-12 else float("-inf")
        ),
    }
    return stereo_bus, report


def _render_event(
    scene: SceneConfig,
    track: TrackSpec,
    event: EventSpec,
    base_dir: Path,
    asset_dir: Path,
    track_gain: float,
    crossfade_frames: int,
) -> tuple[np.ndarray, dict]:
    asset_path = _resolve_asset(base_dir, scene, event, asset_dir)
    samples, asset_sr, asset_width = read_wav(asset_path)
    if asset_sr <= 0:
        raise MixerError(
            ErrorCode.UNSUPPORTED_FORMAT,
            f"asset {asset_path.name} has invalid sample rate {asset_sr}",
            {"path": str(asset_path)},
        )

    resample_entries: list[dict] = []
    if samples.shape[1] == 1:
        mono = samples[:, 0]
        mono, resample_info = resample_mono(mono, asset_sr, scene.project_sample_rate)
        resample_entries.append(resample_info)
        left_gain, right_gain = equal_power_pan(event.pan)
        stereo = np.stack((mono * left_gain, mono * right_gain), axis=1)
        mapping = {
            "input_channels": 1,
            "output_channels": 2,
            "pan": event.pan,
            "left_gain": left_gain,
            "right_gain": right_gain,
            "pan_law": "equal_power_sine_cosine",
        }
    else:
        channels = []
        for channel in range(2):
            converted, resample_info = resample_mono(
                samples[:, channel], asset_sr, scene.project_sample_rate
            )
            channels.append(converted)
            resample_entries.append({**resample_info, "source_channel": channel})
        stereo = np.stack(channels, axis=1)
        left_gain, right_gain = equal_power_pan(event.pan)
        stereo[:, 0] *= left_gain
        stereo[:, 1] *= right_gain
        mapping = {
            "input_channels": 2,
            "output_channels": 2,
            "pan": event.pan,
            "left_gain": left_gain,
            "right_gain": right_gain,
            "pan_law": "equal_power_sine_cosine_applied_per_channel",
        }

    expected_frames = _frame(event.duration, scene.project_sample_rate)
    if stereo.shape[0] > expected_frames:
        stereo = stereo[:expected_frames]
    elif stereo.shape[0] < expected_frames:
        stereo = np.pad(stereo, ((0, expected_frames - stereo.shape[0]), (0, 0)))

    start_frame = _frame(event.start, scene.project_sample_rate)
    fade_entries = _apply_fades(stereo, event, scene.project_sample_rate, start_frame, crossfade_frames)
    stereo *= track_gain

    report = {
        "event_id": event.event_id,
        "track_id": track.track_id,
        "asset_path": str(asset_path),
        "asset_sample_rate": asset_sr,
        "asset_sample_width_bytes": asset_width,
        "asset_input_frames": int(samples.shape[0]),
        "project_start_frame": start_frame,
        "project_end_frame_exclusive": start_frame + expected_frames,
        "project_end_frame_inclusive": start_frame + expected_frames - 1,
        "aligned_frames": expected_frames,
        "muted": bool(track.muted),
        "event_gain_db": 0.0,
        "track_gain_db": track.gain_db,
        "channel_mapping": mapping,
        "resampling": resample_entries,
        "fades": fade_entries,
    }
    return stereo, report
