"""End-to-end rendering pipeline.

Pipeline stages:

1. Validate the project description (see :mod:`cinemixer.project`).
2. For every track: FFmpeg decode, resample to the project rate, normalize
   channels, run the soft-knee compressor, apply event/track gains while
   placing events on the common timeline. Failed tracks are skipped and
   replaced by silence; failure of every track aborts with
   ``E_ALL_TRACKS_FAILED``.
3. Sum tracks, peak-normalize to a safe ceiling to prevent clipping.
4. Measure BS.1770-4 integrated loudness and apply at most one linear gain
   correction when the target is missed by more than 0.5 LU. Sample offsets
   are unaffected because the correction is a pure scalar.
5. Write the WAV output plus the strictly sample-accurate JSON index.

Checkpoints cache fully rendered per-track buffers so an interrupted run
resumes from the most recently successful track.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .compressor import db_to_linear, make_compressor
from .errors import EXIT_CODES, CineMixerError
from .ffmpeg_io import SourceInfo, load_source
from .logging_util import StructuredLogger, logger as default_logger
from .loudness import integrated_loudness, linear_gain_for_target
from .project import Project, TrackSpec, timeline_length_samples
from .resample import normalize_source
from .state import CheckpointStore, checksum_file
from .timeline import IndexEntry, index_entry_for, ms_to_samples, place_event
from .wav_io import write_wav

PEAK_CEILING_LINEAR = float(db_to_linear(-1.0))  # -1 dBFS safety ceiling
LOUDNESS_TOLERANCE_LU = 0.5


@dataclass
class RenderResult:
    output_path: Path
    index_path: Path
    sample_rate: int
    channels: int
    length_samples: int
    measured_lufs: float
    target_lufs: float
    loudness_adjusted: bool
    failed_tracks: tuple[str, ...]
    index_entries: tuple[IndexEntry, ...]
    track_stats: dict[str, dict[str, int | str | bool]]


def _maybe_fault(fault_after: int | None, completed: int) -> None:
    """Deterministic crash injection used by the recovery tests."""
    if fault_after is not None and completed >= fault_after:
        os._exit(130)


class Renderer:
    def __init__(
        self,
        project: Project,
        project_root: Path,
        log: StructuredLogger | None = None,
        fault_after: int | None = None,
    ):
        self.project = project
        self.project_root = project_root
        self.log = log or default_logger
        self.fault_after = fault_after
        self.length = timeline_length_samples(project)
        self.channels = project.channels
        self.rate = project.sample_rate

    def _resolve_source(self, track: TrackSpec) -> Path:
        source = Path(track.source)
        if not source.is_absolute():
            source = (self.project_root / source).resolve()
        return source

    # ------------------------------------------------------------- per track
    def render_track(
        self,
        track: TrackSpec,
        source_checksum: str,
    ) -> tuple[np.ndarray, list[IndexEntry], SourceInfo | None, int]:
        """Render one track onto a zero-initialized timeline buffer."""
        buffer = np.zeros((self.length, self.channels), dtype=np.float64)
        entries: list[IndexEntry] = []
        source_path = self._resolve_source(track)
        audio, info = load_source(source_path)

        normalized = normalize_source(
            audio,
            info.sample_rate,
            info.channels,
            self.rate,
            self.channels,
        )
        self.log.info(
            "source decoded and normalized",
            "I_SOURCE_LOADED",
            track_id=track.track_id,
            native_rate=info.sample_rate,
            native_channels=info.channels,
            native_samples=info.samples,
            normalized_samples=normalized.shape[0],
            resampled=info.sample_rate != self.rate,
        )

        if track.muted:
            # Muted tracks keep their index positions but contribute silence.
            for event in track.events:
                entry = index_entry_for(
                    track.track_id, event, self.project, normalized.shape[0], self.length
                )
                entries.append(entry)
            return buffer, entries, info, normalized.shape[0]

        compressor = make_compressor(self.project.compressor, self.rate)
        track_gain = float(db_to_linear(track.gain_db))
        for event in track.events:
            # Event length on the project clock (pre-clamp), used to slice a
            # contiguous region out of the normalized source.
            start = ms_to_samples(event.start_ms, self.rate)
            duration = ms_to_samples(event.duration_ms, self.rate)
            offset = ms_to_samples(event.source_offset_ms, self.rate)

            # Bounds for source slicing (clamp negatives before indexing).
            src_begin = max(offset + max(0, -start), 0)
            src_end = max(src_begin, offset + max(0, -start) + max(duration, 0))
            src_end = min(src_end, normalized.shape[0])
            source_slice = normalized[src_begin:src_end] if src_end > src_begin else np.zeros(
                (0, self.channels), dtype=np.float64
            )

            # Compression runs over contiguous source content so the envelope
            # state evolves naturally across consecutive/overlapping events.
            compressed = compressor.process(source_slice)
            event_gain = float(db_to_linear(event.gain_db))
            prepared = compressed * (event_gain * track_gain)

            entry = place_event(
                buffer,
                prepared,
                track.track_id,
                event,
                self.project,
                self.length,
                self.log,
            )
            entries.append(entry)
        return buffer, entries, info, normalized.shape[0]

    # ------------------------------------------------------------------ main
    def run(self, state: CheckpointStore) -> RenderResult:
        project = self.project
        track_buffers: list[np.ndarray] = []
        index_entries: list[IndexEntry] = []
        failed_tracks: list[str] = []
        track_stats: dict[str, dict[str, int | str | bool]] = {}
        completed = 0

        for track in project.tracks:
            source_path = self._resolve_source(track)
            try:
                source_checksum = checksum_file(source_path)
            except OSError:
                source_checksum = "missing"

            cached = state.load_cached_track(track.track_id) if state.is_track_done(
                track.track_id, source_checksum
            ) else None
            if cached is not None:
                self.log.info(
                    "resuming track from checkpoint",
                    "I_TRACK_RESUMED",
                    track_id=track.track_id,
                )
                buffer = cached
                entries = self._index_entries_for_track(
                    track, state.tracks[track.track_id].source_length
                )
                track_stats[track.track_id] = {
                    "status": "resumed",
                    "muted": track.muted,
                    "normalized_samples": state.tracks[track.track_id].source_length,
                    "timeline_samples": self.length,
                }
            else:
                try:
                    buffer, entries, _info, source_length = self.render_track(
                        track, source_checksum
                    )
                except CineMixerError as exc:
                    if exc.code != "E_TRACK_LOAD_FAIL":
                        raise
                    self.log.error(
                        exc.code,
                        exc.message,
                        track_id=track.track_id,
                        source=str(source_path),
                    )
                    failed_tracks.append(track.track_id)
                    state.mark_track(track.track_id, source_checksum, "failed", source_length=0)
                    buffer = np.zeros((self.length, self.channels), dtype=np.float64)
                    # Failed tracks are silence over the full project length;
                    # source length 0 forces complete silence padding while the
                    # timeline clamping still determines valid sample ranges.
                    entries = self._index_entries_for_track(track, 0)
                    track_stats[track.track_id] = {
                        "status": "failed",
                        "muted": track.muted,
                        "normalized_samples": 0,
                        "timeline_samples": self.length,
                    }
                else:
                    state.mark_track(
                        track.track_id,
                        source_checksum,
                        "done",
                        buffer,
                        source_length=source_length,
                    )
                    track_stats[track.track_id] = {
                        "status": "muted" if track.muted else "rendered",
                        "muted": track.muted,
                        "normalized_samples": source_length,
                        "timeline_samples": self.length,
                    }
            track_buffers.append(buffer)
            index_entries.extend(entries)
            completed += 1
            _maybe_fault(self.fault_after, completed)

        if len(failed_tracks) == len(project.tracks):
            raise CineMixerError(
                "E_ALL_TRACKS_FAILED",
                "every track failed to load; no output was produced",
                EXIT_CODES["E_ALL_TRACKS_FAILED"],
            )

        mix = np.sum(np.stack(track_buffers, axis=0), axis=0)
        mix, peak_scale = self._peak_normalize(mix)

        measured = integrated_loudness(mix, self.rate)
        adjusted = False
        if abs(measured - project.target_lufs) > LOUDNESS_TOLERANCE_LU:
            correction = linear_gain_for_target(measured, project.target_lufs)
            self.log.info(
                "applying one-shot loudness correction",
                "I_LOUDNESS_ADJUST",
                measured_lufs=round(measured, 3),
                target_lufs=project.target_lufs,
                gain_db=round(20.0 * np.log10(correction), 3),
            )
            mix = self._apply_safe_correction(mix, correction)
            measured = integrated_loudness(mix, self.rate)
            adjusted = True

        output_path = self._resolve_output(project.output.path)
        write_wav(output_path, mix, self.rate, project.output.bit_depth)

        index_path = output_path.with_suffix(".index.json")
        self._write_index(
            index_path, index_entries, measured, adjusted, peak_scale, track_stats
        )

        return RenderResult(
            output_path=output_path,
            index_path=index_path,
            sample_rate=self.rate,
            channels=self.channels,
            length_samples=self.length,
            measured_lufs=measured,
            target_lufs=project.target_lufs,
            loudness_adjusted=adjusted,
            failed_tracks=tuple(failed_tracks),
            index_entries=tuple(index_entries),
            track_stats=track_stats,
        )

    # --------------------------------------------------------------- helpers
    def _index_entries_for_track(self, track: TrackSpec, source_length: int) -> list[IndexEntry]:
        return [
            index_entry_for(track.track_id, event, self.project, source_length, self.length)
            for event in track.events
        ]

    def _peak_normalize(self, mix: np.ndarray) -> tuple[np.ndarray, float]:
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak <= 0.0:
            return mix, 1.0
        scale = min(1.0, PEAK_CEILING_LINEAR / peak)
        return mix * scale, scale

    def _apply_safe_correction(self, mix: np.ndarray, correction: float) -> np.ndarray:
        corrected = mix * correction
        peak = float(np.max(np.abs(corrected))) if corrected.size else 0.0
        if peak > PEAK_CEILING_LINEAR:
            # Never clip: fold the excess back while keeping a pure scalar
            # (sample offsets in the index remain unchanged).
            corrected = corrected * (PEAK_CEILING_LINEAR / peak)
        return corrected

    def _resolve_output(self, configured: str) -> Path:
        output = Path(configured)
        if not output.is_absolute():
            output = (self.project_root / output).resolve()
        return output

    def _write_index(
        self,
        index_path: Path,
        entries: list[IndexEntry],
        measured_lufs: float,
        adjusted: bool,
        peak_scale: float,
        track_stats: dict[str, dict[str, int | str | bool]],
    ) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "project_id": self.project.project_id,
            "sample_rate": self.rate,
            "channels": self.channels,
            "length_samples": self.length,
            "duration_ms": self.project.duration_ms,
            "target_lufs": self.project.target_lufs,
            "measured_lufs": None if not np.isfinite(measured_lufs) else round(measured_lufs, 4),
            "loudness_adjusted": adjusted,
            "peak_normalize_scale": peak_scale,
            "tracks": track_stats,
            "events": [entry.as_dict() for entry in entries],
        }
        index_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
