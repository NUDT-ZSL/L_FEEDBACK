from __future__ import annotations

import hashlib
import os
import wave

import numpy as np

from rhythmforge.generator import SOURCE_SPECS, generate_project


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_covers_required_format_combinations(project):
    rates = {spec.sample_rate for spec in SOURCE_SPECS}
    channels = {spec.channels for spec in SOURCE_SPECS}
    widths = {spec.sample_width for spec in SOURCE_SPECS}
    assert {44100, 48000}.issubset(rates)
    assert {1, 2}.issubset(channels)
    assert {16, 24}.issubset(widths)

    for spec in SOURCE_SPECS:
        path = os.path.join(project["samples_dir"], spec.file_name)
        with wave.open(path, "rb") as handle:
            assert handle.getframerate() == spec.sample_rate
            assert handle.getnchannels() == spec.channels
            assert handle.getsampwidth() * 8 == spec.sample_width
            assert handle.getcomptype() == "NONE"
            assert handle.getnframes() == int(
                round(spec.duration_seconds * spec.sample_rate))


def test_lead_in_silence_present(project):
    from rhythmforge.waveio import read_wav

    for spec in SOURCE_SPECS:
        data, _ = read_wav(os.path.join(project["samples_dir"], spec.file_name))
        lead = spec.onset_frame
        # Everything strictly before the declared onset frame is silence.
        assert lead > 0
        assert float((data[:lead] == 0).mean()) == 1.0
        # Audible content begins within a short window after the onset frame
        # (the generator applies a millisecond attack envelope at the onset).
        attack_window = max(2, int(0.01 * spec.sample_rate))
        assert float(np.abs(data[lead:lead + attack_window]).max()) > 0.0


def test_generation_is_byte_deterministic(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    generate_project(str(first), str(first / "chart.json"))
    generate_project(str(second), str(second / "chart.json"))

    files_a = sorted(f for f in os.listdir(first) if f.endswith(".wav"))
    files_b = sorted(f for f in os.listdir(second) if f.endswith(".wav"))
    assert files_a == files_b
    for name in files_a:
        assert _sha256(str(first / name)) == _sha256(str(second / name))
