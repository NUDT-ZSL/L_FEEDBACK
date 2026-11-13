"""CUE-style timestamp indexes and audio/index reconciliation."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .scheduler import LanguageSchedule, Placement


def _frames_to_cue_time(frames: int, sample_rate: int) -> str:
    """Convert sample frames to CUE ``MM:SS:FF`` (75 fps CUE frames)."""
    total_seconds = frames / sample_rate
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    cue_frames = round((total_seconds - (minutes * 60 + seconds)) * 75.0)
    if cue_frames == 75:
        cue_frames = 0
        seconds += 1
        if seconds == 60:
            seconds = 0
            minutes += 1
    return f"{minutes:02d}:{seconds:02d}:{cue_frames:02d}"


def build_index_entries(schedule: LanguageSchedule, config: Config) -> list[dict]:
    entries = []
    for placement in schedule.placements:
        entry = placement.to_index_dict()
        entry["cue_time"] = _frames_to_cue_time(
            placement.start, config.sample_rate
        )
        entries.append(entry)
    return entries


def render_cue_sheet(
    schedule: LanguageSchedule,
    wav_file: str,
    config: Config,
) -> str:
    """Render a human-readable CUE sheet for one language WAV."""
    lines = [
        f'FILE "{wav_file}" WAVE',
        f"REM LANG {schedule.lang}",
    ]
    for number, placement in enumerate(schedule.placements, start=1):
        cue_time = _frames_to_cue_time(placement.start, config.sample_rate)
        lines.extend(
            [
                f"  TRACK {number:02d} AUDIO",
                f'    TITLE "{placement.event_id}"',
                f"    REM LANG {placement.lang}",
                f"    REM SOURCE {placement.source_name}",
                f"    REM PRIORITY {placement.priority}",
                f"    REM SHIFT_SAMPLES {placement.shift_samples}",
                f"    INDEX 01 {cue_time}",
            ]
        )
    return "\n".join(lines) + "\n"


@dataclass
class ReconciliationError:
    code: str
    message: str
    detail: dict

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "detail": self.detail}


def reconcile(
    schedule: LanguageSchedule,
    entries: list[dict],
    wav_info: dict,
    config: Config,
) -> list[ReconciliationError]:
    """Verify every retained clip against the final WAV frame geometry."""
    errors: list[ReconciliationError] = []
    placements: list[Placement] = schedule.placements

    if len(entries) != len(placements):
        errors.append(
            ReconciliationError(
                "E_INDEX_MISMATCH",
                "index entry count differs from retained clip count",
                {"entries": len(entries), "placements": len(placements)},
            )
        )
        return errors

    total_frames = wav_info["frames"]
    for placement, entry in zip(placements, entries):
        prefix = f"event {placement.event_id}"
        if entry["event_id"] != placement.event_id:
            errors.append(
                ReconciliationError(
                    "E_INDEX_MISMATCH",
                    f"{prefix}: event id mismatch",
                    {"index": entry["event_id"], "placement": placement.event_id},
                )
            )
        if entry["sample_offset"] != placement.start:
            errors.append(
                ReconciliationError(
                    "E_INDEX_MISMATCH",
                    f"{prefix}: sample offset mismatch",
                    {"index": entry["sample_offset"], "placement": placement.start},
                )
            )
        if entry["frames"] != placement.frames:
            errors.append(
                ReconciliationError(
                    "E_INDEX_MISMATCH",
                    f"{prefix}: frame count mismatch",
                    {"index": entry["frames"], "placement": placement.frames},
                )
            )
        if entry["end_sample"] != placement.end:
            errors.append(
                ReconciliationError(
                    "E_INDEX_MISMATCH",
                    f"{prefix}: end sample mismatch",
                    {"index": entry["end_sample"], "placement": placement.end},
                )
            )
        if entry["source_file"] != placement.source_name:
            errors.append(
                ReconciliationError(
                    "E_INDEX_MISMATCH",
                    f"{prefix}: source file mismatch",
                    {
                        "index": entry["source_file"],
                        "placement": placement.source_name,
                    },
                )
            )
        if placement.start < 0 or placement.end > total_frames:
            errors.append(
                ReconciliationError(
                    "E_OUT_OF_RANGE",
                    f"{prefix}: clip extends outside final WAV",
                    {
                        "start": placement.start,
                        "end": placement.end,
                        "total_frames": total_frames,
                    },
                )
            )
        if entry["cue_time"] != _frames_to_cue_time(
            placement.start, config.sample_rate
        ):
            errors.append(
                ReconciliationError(
                    "E_INDEX_MISMATCH",
                    f"{prefix}: cue timestamp mismatch",
                    {"index": entry["cue_time"]},
                )
            )

    if wav_info["sample_rate"] != config.sample_rate:
        errors.append(
            ReconciliationError(
                "E_WAV_HEADER",
                "WAV sample rate does not match configuration",
                {"wav": wav_info["sample_rate"], "config": config.sample_rate},
            )
        )
    if wav_info["channels"] != 1 or wav_info["subtype"] != "PCM_16":
        errors.append(
            ReconciliationError(
                "E_WAV_HEADER",
                "WAV is not 16-bit mono PCM",
                {"channels": wav_info["channels"], "subtype": wav_info["subtype"]},
            )
        )
    return errors
