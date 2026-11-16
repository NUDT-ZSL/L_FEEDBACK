"""Source loading, timeline scheduling and per-section rendering."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .dsp import (
    INTERNAL_SAMPLE_RATE,
    apply_gain_db,
    normalize_channels,
    resample_to_rate,
    time_stretch,
)
from .errors import BuildError, ExitCode
from .timing import Section, TimeMap
from .waveio import read_wav


@dataclass
class LayerSpec:
    layer_id: str
    path: str
    enabled: bool
    gain_db: float
    onset_beat: float
    lead_in_seconds: float
    native_bpm: float


@dataclass
class LoadedSource:
    spec: LayerSpec
    audio: np.ndarray
    source_rate: int
    channels: int
    sample_width: int
    onset_frame_internal: int
    native_frames: int


def parse_layers(payload: dict, samples_dir: str) -> list[LayerSpec]:
    raw = payload.get("layers")
    if not isinstance(raw, list) or not raw:
        raise BuildError(ExitCode.CHART_SYNTAX, "layers must be a non-empty list")
    layers: list[LayerSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise BuildError(ExitCode.CHART_SYNTAX,
                             f"layers[{index}] must be an object")
        layer_id = item.get("id")
        file_name = item.get("file")
        if not isinstance(layer_id, str) or not layer_id:
            raise BuildError(ExitCode.CHART_SYNTAX,
                             f"layers[{index}].id must be a non-empty string")
        if layer_id in seen:
            raise BuildError(ExitCode.CHART_SYNTAX,
                             f"duplicate layer id: {layer_id}",
                             {"layer": layer_id})
        seen.add(layer_id)
        if not isinstance(file_name, str) or not file_name:
            raise BuildError(ExitCode.CHART_SYNTAX,
                             f"layers[{index}].file must be a non-empty string")
        gain_db = item.get("gain_db", 0.0)
        if not isinstance(gain_db, (int, float)) or isinstance(gain_db, bool):
            raise BuildError(ExitCode.CHART_SYNTAX,
                             f"layers[{index}].gain_db must be numeric")
        onset_beat = item.get("onset_beat", 0.0)
        lead_in_seconds = item.get("lead_in_seconds", 0.0)
        native_bpm = item.get("native_bpm", 120.0)
        for field_name, value in (("onset_beat", onset_beat),
                                  ("lead_in_seconds", lead_in_seconds),
                                  ("native_bpm", native_bpm)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or not np.isfinite(value):
                raise BuildError(ExitCode.CHART_SYNTAX,
                                 f"layers[{index}].{field_name} must be a finite number")
        if native_bpm <= 0.0 or lead_in_seconds < 0.0:
            raise BuildError(ExitCode.MAPPING_CONFLICT,
                             f"layers[{index}] has invalid native_bpm/lead_in_seconds")
        layers.append(LayerSpec(
            layer_id=layer_id,
            path=os.path.join(samples_dir, file_name),
            enabled=bool(item.get("enabled", True)),
            gain_db=float(gain_db),
            onset_beat=float(onset_beat),
            lead_in_seconds=float(lead_in_seconds),
            native_bpm=float(native_bpm),
        ))
    return layers


def parse_notes(payload: dict, known_layers: set[str]) -> list[dict]:
    raw = payload.get("notes", [])
    if not isinstance(raw, list):
        raise BuildError(ExitCode.CHART_SYNTAX, "notes must be a list")
    notes: list[dict] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise BuildError(ExitCode.CHART_SYNTAX,
                             f"notes[{index}] must be an object")
        beat = item.get("beat")
        layer_id = item.get("layer")
        velocity = item.get("velocity", 1.0)
        if not isinstance(beat, (int, float)) or isinstance(beat, bool) or not np.isfinite(beat):
            raise BuildError(ExitCode.CHART_SYNTAX,
                             f"notes[{index}].beat must be a finite number")
        if layer_id not in known_layers:
            raise BuildError(ExitCode.CHART_SYNTAX,
                             f"notes[{index}] references unknown layer '{layer_id}'",
                             {"layer": layer_id})
        if not isinstance(velocity, (int, float)) or not 0.0 <= float(velocity) <= 1.0:
            raise BuildError(ExitCode.CHART_SYNTAX,
                             f"notes[{index}].velocity must be within [0, 1]")
        notes.append({"beat": float(beat), "layer": layer_id,
                      "velocity": float(velocity)})
    return notes


def load_sources(layers: list[LayerSpec],
                 internal_rate: int = INTERNAL_SAMPLE_RATE) -> dict[str, LoadedSource]:
    """Read, resample and stereo-normalize every declared source."""
    loaded: dict[str, LoadedSource] = {}
    for spec in layers:
        audio, info = read_wav(spec.path)
        resampled = resample_to_rate(audio, info.sample_rate, internal_rate)
        stereo = normalize_channels(resampled)
        stereo = apply_gain_db(stereo, spec.gain_db)
        # The lead-in is preserved through resampling: convert the *native*
        # onset frame into internal frames using the exact rate ratio.
        onset_frame = int(round(spec.lead_in_seconds * internal_rate))
        loaded[spec.layer_id] = LoadedSource(
            spec=spec,
            audio=np.ascontiguousarray(stereo, dtype=np.float32),
            source_rate=info.sample_rate,
            channels=info.channels,
            sample_width=info.sample_width,
            onset_frame_internal=onset_frame,
            native_frames=info.frames,
        )
    return loaded


def _place(buffer: np.ndarray, source: np.ndarray, position: int,
           velocity: float) -> None:
    """Add ``source`` into ``buffer`` at frame ``position`` (with clipping bounds)."""
    if position >= buffer.shape[0]:
        return
    start = max(0, position)
    source_start = start - position
    available = min(buffer.shape[0] - start, source.shape[0] - source_start)
    if available <= 0:
        return
    buffer[start:start + available] += source[source_start:source_start + available] * velocity


def render_section(section: Section, time_map: TimeMap,
                   sources: dict[str, LoadedSource], notes: list[dict],
                   internal_rate: int = INTERNAL_SAMPLE_RATE) -> dict[str, np.ndarray]:
    """Render one section and return a stem array per layer.

    Notes whose beat falls into the half-open section interval are scheduled
    on the un-stretched beat grid, then each layer is time-stretched by the
    section factor and forced to the section's exact frame count.
    """
    section_frames = time_map.section_frames(section.start_beat, section.end_beat)
    if section_frames <= 0:
        raise BuildError(ExitCode.DURATION_MISMATCH,
                         "section maps to zero or negative frames",
                         {"section": section.section_id, "frames": section_frames})

    section_bpm = time_map.bpm_at(section.start_beat + 1e-12)
    stretch_factor = section.native_bpm / section_bpm

    # Pre-stretch every layer source once.  Scheduling then happens on the
    # final section grid, so onset offsets and beat positions stay coherent.
    prepared: dict[str, tuple[np.ndarray, int]] = {}
    for layer_id, source in sources.items():
        stretched_source = time_stretch(source.audio, stretch_factor,
                                        internal_rate, exact_frames=None)
        stretched_onset = int(round(source.onset_frame_internal * stretch_factor))
        prepared[layer_id] = (stretched_source, stretched_onset)

    stems: dict[str, np.ndarray] = {}
    for layer_id in sources:
        stems[layer_id] = np.zeros((section_frames, 2), dtype=np.float32)

    origin_frame = int(time_map.beat_to_frame(section.start_beat))
    for note in notes:
        if not (section.start_beat <= note["beat"] < section.end_beat):
            continue
        source_audio, onset_frame = prepared[note["layer"]]
        note_frame = int(time_map.beat_to_frame(note["beat"]))
        position = note_frame - origin_frame - onset_frame
        _place(stems[note["layer"]], source_audio, position, note["velocity"])

    for layer_id, stem in stems.items():
        if stem.shape[0] != section_frames:
            raise BuildError(
                ExitCode.DURATION_MISMATCH,
                "rendered section length does not match the declared timeline",
                {"section": section.section_id,
                 "expected_frames": section_frames,
                 "actual_frames": int(stem.shape[0])},
            )
    return stems
