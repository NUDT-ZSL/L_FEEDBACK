"""End-to-end deterministic offline mixing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .atomic_io import write_json_atomic, write_wav_atomic
from .config import (
    ProjectConfig,
    ShotSpec,
    load_choice_path,
    load_project_config,
    resolve_path_shots,
)
from .envelope import crossfade_envelopes, track_envelope
from .errors import ErrorCode, MixerError
from .media import MediaLoader
from .report import (
    build_error_report,
    build_partial_report,
    build_success_report,
    peak_of,
)
from .timeline import Overlap, validate_and_order


def _attach_partial_report(
    error: MixerError,
    config: ProjectConfig | None,
    selected: list[ShotSpec],
) -> None:
    if "partial_report" in error.context or config is None:
        return
    partial_ids = error.context.get("partial_shot_ids")
    if isinstance(partial_ids, list):
        partial_shots = [
            config.shots_by_id[shot_id]
            for shot_id in partial_ids
            if isinstance(shot_id, str) and shot_id in config.shots_by_id
        ]
    else:
        partial_shots = selected
    error.context["partial_report"] = build_partial_report(config, partial_shots)
    error.context["target_sample_rate"] = config.target_sample_rate


def _render(
    config: ProjectConfig,
    ordered: list[ShotSpec],
    overlaps: list[Overlap],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], int, bool, float]:
    total_samples = max(shot.end_sample for shot in ordered)
    buses = {
        kind: np.zeros(total_samples, dtype=np.float64)
        for kind in ("dialogue", "ambience", "music")
    }
    loader = MediaLoader(config.target_sample_rate)
    crossfades = crossfade_envelopes(ordered, overlaps)
    shot_peaks: dict[str, float] = {}

    for shot in ordered:
        shot_crossfade = crossfades[shot.shot_id]
        shot_peak = 0.0
        for track in shot.tracks:
            clip = loader.load(track.path).samples
            usable = min(shot.duration_samples, clip.size)
            if usable <= 0:
                continue
            try:
                gain = 10.0 ** (track.gain_db / 20.0)
            except OverflowError as exc:
                raise MixerError(
                    ErrorCode.E_SHOT_DESC,
                    f"{shot.shot_id}: gain_db is too large",
                ) from exc
            if not np.isfinite(gain):
                raise MixerError(
                    ErrorCode.E_SHOT_DESC,
                    f"{shot.shot_id}: gain_db produces a non-finite gain",
                )
            envelope = track_envelope(
                shot.duration_samples,
                track.fade_in_samples,
                track.fade_out_samples,
            )
            contribution = np.zeros(shot.duration_samples, dtype=np.float64)
            contribution[:usable] = clip[:usable]
            contribution *= envelope
            contribution *= gain
            shot_peak = max(shot_peak, peak_of(contribution))
            contribution *= shot_crossfade
            buses[track.kind][shot.start_sample : shot.end_sample] += contribution
        shot_peaks[shot.shot_id] = shot_peak

    weighted = (
        buses["dialogue"] * config.bus_weights["dialogue"]
        + buses["ambience"] * config.bus_weights["ambience"]
        + buses["music"] * config.bus_weights["music"]
    )
    if not np.all(np.isfinite(weighted)):
        raise MixerError(
            ErrorCode.E_SHOT_DESC,
            "gain or bus weights produced non-finite mix samples",
        )
    ceiling = 10.0 ** (config.limiter_ceiling_dbfs / 20.0)
    raw_peak = peak_of(weighted)
    limiter_active = raw_peak > ceiling and raw_peak > 0.0
    limiter_gain = ceiling / raw_peak if limiter_active else 1.0
    master = np.clip(weighted * limiter_gain, -ceiling, ceiling)
    return master, buses, shot_peaks, total_samples, limiter_active, limiter_gain


def _output_layout(
    buses: dict[str, np.ndarray],
    master: np.ndarray,
    channels: int,
) -> np.ndarray:
    """Place stems and master on the multichannel WAV channels.

    The canonical four-channel layout is dialogue, ambience, music, master.
    Mono/stereo configurations carry the master signal only; three channels
    carry the three unweighted stems; extra channels repeat the master.
    """

    stems = np.column_stack(
        (buses["dialogue"], buses["ambience"], buses["music"], master)
    )
    if channels >= 4:
        if channels == 4:
            return stems
        return np.column_stack(
            [stems, *[master[:, np.newaxis] for _ in range(channels - 4)]]
        )
    if channels == 3:
        return stems[:, :3]
    if channels == 1:
        return master
    return np.column_stack((master, master))


def mix_files(
    config_path: str | Path,
    choice_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Mix one selected path, write WAV and JSON, and return the report."""

    config: ProjectConfig | None = None
    selected: list[ShotSpec] = []
    try:
        config = load_project_config(config_path)
        choices = load_choice_path(choice_path)
        selected = resolve_path_shots(config, choices)
        ordered, overlaps = validate_and_order(selected)
        master, buses, shot_peaks, total_samples, limiter_active, limiter_gain = _render(
            config, ordered, overlaps
        )
        report = build_success_report(
            config,
            ordered,
            buses=buses,
            master=master,
            shot_peaks=shot_peaks,
            total_samples=total_samples,
            limiter_active=limiter_active,
            limiter_gain=limiter_gain,
        )

        output = Path(output_path).resolve()
        destination = Path(report_path).resolve()
        if output == destination:
            raise MixerError(
                ErrorCode.E_SHOT_DESC, "--out and --report must be different files"
            )
        output_audio = _output_layout(buses, master, config.output_channels)
        write_wav_atomic(
            output,
            output_audio,
            sample_rate=config.target_sample_rate,
            subtype=config.output_subtype,
        )
        write_json_atomic(destination, report)
        return report
    except MixerError as error:
        _attach_partial_report(error, config, selected)
        try:
            write_json_atomic(report_path, build_error_report(error))
        except OSError:
            pass
        raise


def estimate_duration_samples(
    shots: list[ShotSpec], start_sample: int = 0
) -> int:
    """Small public helper useful to callers validating their own timelines."""

    return max((shot.end_sample for shot in shots), default=start_sample)


__all__ = ["mix_files", "estimate_duration_samples"]
