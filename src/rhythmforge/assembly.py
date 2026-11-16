"""Section concatenation with equal-power crossfades and frame accounting."""

from __future__ import annotations

import numpy as np

from .dsp import equal_power_crossfade
from .errors import BuildError, ExitCode


def crossfade_frames(time_map, crossfade_beats: float,
                     sample_rate: int) -> int:
    """Convert the chart's crossfade length in beats at the first tempo."""
    if crossfade_beats < 0:
        raise BuildError(ExitCode.CHART_SYNTAX,
                         "crossfade_beats must be non-negative")
    first_bpm = float(time_map.bpms[0])
    seconds = crossfade_beats * 60.0 / first_bpm
    return int(round(seconds * sample_rate))


def assemble_sections(section_stems: list[dict[str, np.ndarray]],
                      layer_order: list[str], fade_frames: int) -> np.ndarray:
    """Concatenate per-section stems into full-length per-layer arrays.

    Each section keeps its exact declared frame slot.  At each internal
    boundary an equal-power crossfade blends the tail of the previous slot
    with the head of the next; the total length stays exactly equal to the
    sum of section frame lengths, matching marker positions.
    """
    total_frames = sum(stems[layer_order[0]].shape[0]
                       for stems in section_stems)
    assembled = {
        layer_id: np.zeros((total_frames, 2), dtype=np.float32)
        for layer_id in layer_order
    }

    offset = 0
    for section_index, stems in enumerate(section_stems):
        length = stems[layer_order[0]].shape[0]
        for layer_id in layer_order:
            if stems[layer_id].shape[0] != length:
                raise BuildError(ExitCode.DURATION_MISMATCH,
                                 "layer stems within a section disagree on length",
                                 {"section_index": section_index,
                                  "layer": layer_id})
        if section_index == 0 or fade_frames <= 0:
            for layer_id in layer_order:
                assembled[layer_id][offset:offset + length] = stems[layer_id]
        else:
            usable_fade = min(fade_frames, offset, length)
            fade_in = equal_power_crossfade(usable_fade)
            fade_out = np.sqrt(np.clip(1.0 - fade_in ** 2, 0.0, 1.0))
            boundary = offset
            for layer_id in layer_order:
                previous = assembled[layer_id][boundary - usable_fade:boundary]
                current = stems[layer_id][:usable_fade]
                blended = previous * fade_out[:, None] + current * fade_in[:, None]
                assembled[layer_id][boundary - usable_fade:boundary] = blended.astype(np.float32)
                assembled[layer_id][boundary:boundary + length - usable_fade] = \
                    stems[layer_id][usable_fade:]
        offset += length

    if offset != total_frames:
        raise BuildError(ExitCode.DURATION_MISMATCH,
                         "assembled timeline length drifted from section slots",
                         {"expected_frames": total_frames, "actual_frames": offset})
    return np.stack([assembled[layer_id] for layer_id in layer_order], axis=0)
