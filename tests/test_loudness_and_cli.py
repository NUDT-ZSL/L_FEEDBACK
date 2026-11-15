"""Gain smoothing behavior and CLI error contract."""

from __future__ import annotations
import json
import os

import numpy as np
import pytest

from audiopack.cli import main as cli_main
from audiopack.loudness import gain_envelope, linear_gain
from tests.samples import sine_wave, write_wav


def test_gain_envelope_minimum_ramp_length():
    sr = 48000
    length = sr  # 1 second
    # Automation points only 10 ms apart must be expanded to >= 100 ms.
    env = gain_envelope(length, sr, [(1000, 0.0), (1480, -12.0)])
    min_ramp = int(round(0.1 * sr))

    # The first differing sample cannot appear before the 100 ms ramp budget.
    start_value = linear_gain(0.0)
    first_change = int(np.argmax(env != np.float32(start_value)))
    assert first_change >= 1000
    # Ramp reaches its target no earlier than min_ramp samples later.
    target = linear_gain(-12.0)
    reach = int(np.argmax(np.isclose(env, target, atol=1e-6)))
    assert reach - first_change + 1 >= min_ramp
    # Monotonic decrease and bounded, finite values throughout.
    diffs = np.diff(env.astype(np.float64))
    assert np.all(diffs <= 1e-9)
    assert np.isfinite(env).all()


def test_constant_envelope_is_uniform_gain():
    env = gain_envelope(48000, 48000, [(0, -6.0)])
    assert np.allclose(env, linear_gain(-6.0), atol=1e-7)


def test_cli_emits_json_error_and_nonzero_exit(tmp_path, capsys):
    # Directory with a WAV but no timeline -> E_TIMELINE_FORMAT on stderr.
    root = tmp_path / "bad"
    root.mkdir()
    write_wav(str(root / "a.wav"), sine_wave(0.1, 440, 48000), 48000, "PCM_16")
    out = str(tmp_path / "out.wav")

    exit_code = cli_main([str(root), out])
    captured = capsys.readouterr()
    assert exit_code != 0
    payload = json.loads(captured.err.strip().splitlines()[-1])
    assert payload["error"]["code"] == "E_TIMELINE_FORMAT"
    assert not os.path.exists(out)


def test_cli_range_error_exit_code(tmp_path, capsys):
    root = tmp_path / "range"
    root.mkdir()
    write_wav(str(root / "a.wav"), sine_wave(0.1, 440, 48000), 48000, "PCM_16")
    timeline = {
        "sample_rate": 48000,
        "events": [
            {
                "start_ms": -5, "end_ms": 100, "clip": "a.wav",
                "fade_in_ms": 0, "fade_out_ms": 0, "gain_db": 0,
            }
        ],
    }
    json.dump(timeline, open(root / "timeline.json", "w"))
    exit_code = cli_main([str(root), str(tmp_path / "o.wav")])
    payload = json.loads(capsys.readouterr().err.strip())
    assert exit_code == 1
    assert payload["error"]["code"] == "E_RANGE"
