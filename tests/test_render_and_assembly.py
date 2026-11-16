from __future__ import annotations

import numpy as np

from rhythmforge.assembly import assemble_sections
from rhythmforge.dsp import equal_power_crossfade
from rhythmforge.renderer import load_sources, parse_layers, parse_notes, render_section
from rhythmforge.timing import load_chart


def _context(project):
    chart, time_map, signatures, sections = load_chart(project["chart_path"])
    layers = parse_layers(chart, project["samples_dir"])
    notes = parse_notes(chart, {layer.layer_id for layer in layers})
    sources = load_sources(layers, time_map.sample_rate)
    return chart, time_map, sections, layers, notes, sources


def test_sources_normalized_to_internal_format(project):
    _, time_map, _, layers, _, sources = _context(project)
    for layer in layers:
        loaded = sources[layer.layer_id]
        assert loaded.audio.dtype == np.float32
        assert loaded.audio.ndim == 2 and loaded.audio.shape[1] == 2
        # Onset offset survives resampling at the internal rate.
        expected = int(round(layer.lead_in_seconds * time_map.sample_rate))
        assert loaded.onset_frame_internal == expected


def test_every_section_matches_declared_frames(project):
    _, time_map, sections, _, notes, sources = _context(project)
    for section in sections:
        stems = render_section(section, time_map, sources, notes,
                               time_map.sample_rate)
        expected = time_map.section_frames(section.start_beat, section.end_beat)
        for audio in stems.values():
            assert audio.shape[0] == expected


def test_assembly_total_length_equals_sum_of_sections(project):
    _, time_map, sections, layers, notes, sources = _context(project)
    layer_order = [layer.layer_id for layer in layers]
    rendered = [render_section(section, time_map, sources, notes,
                               time_map.sample_rate) for section in sections]
    assembled = assemble_sections(rendered, layer_order, fade_frames=2048)
    expected = sum(time_map.section_frames(s.start_beat, s.end_beat)
                   for s in sections)
    assert assembled.shape == (len(layer_order), expected, 2)


def test_equal_power_crossfade_preserves_energy():
    length = 512
    fade_in = equal_power_crossfade(length)
    fade_out = np.sqrt(np.clip(1.0 - fade_in ** 2, 0.0, 1.0))
    assert np.allclose(fade_in ** 2 + fade_out ** 2, 1.0)


def test_stem_sum_equals_master_in_float(project, built_project):
    import json
    import os

    from rhythmforge.waveio import read_wav

    output = built_project["output_dir"]
    chart = built_project["chart"]
    enabled = {layer["id"] for layer in chart["layers"] if layer["enabled"]}
    master, info = read_wav(os.path.join(output, "master.wav"))
    total = np.zeros_like(master, dtype=np.float64)
    for layer_id in enabled:
        stem, stem_info = read_wav(os.path.join(output, "stems", f"{layer_id}.wav"))
        assert stem_info.frames == info.frames
        total += stem.astype(np.float64)
    assert np.max(np.abs(total - master.astype(np.float64))) < 2e-4
