from __future__ import annotations
import json
import os
import subprocess
import sys

import pytest

from rhythmforge.errors import BuildError, ExitCode
from rhythmforge.locking import OutputLock


def _write_chart(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _base_chart(project):
    return json.loads(open(project["chart_path"]).read())


def test_missing_chart_file(project):
    from rhythmforge.timing import load_chart

    with pytest.raises(BuildError) as exc:
        load_chart(os.path.join(project["base"], "does_not_exist.json"))
    assert exc.value.code == ExitCode.MISSING_SOURCE


def test_chart_syntax_error(project):
    bad = os.path.join(project["base"], "syntax.json")
    with open(bad, "w", encoding="utf-8") as handle:
        handle.write("{ not valid json")
    from rhythmforge.timing import load_chart

    with pytest.raises(BuildError) as exc:
        load_chart(bad)
    assert exc.value.code == ExitCode.CHART_SYNTAX


def test_missing_source_wav(project):
    chart = _base_chart(project)
    chart["layers"][0]["file"] = "ghost.wav"
    path = os.path.join(project["base"], "missing_source.json")
    _write_chart(path, chart)
    from rhythmforge.pipeline import build

    with pytest.raises(BuildError) as exc:
        build(path, project["samples_dir"],
              os.path.join(project["base"], "out_missing"))
    assert exc.value.code == ExitCode.MISSING_SOURCE


def test_unsupported_format_rejected(project):
    import wave

    bad_wav = os.path.join(project["samples_dir"], "eightbit.wav")
    with wave.open(bad_wav, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(8000)
        handle.writeframes(bytes(32))
    chart = _base_chart(project)
    chart["layers"][0]["file"] = "eightbit.wav"
    path = os.path.join(project["base"], "bad_format.json")
    _write_chart(path, chart)
    from rhythmforge.pipeline import build

    with pytest.raises(BuildError) as exc:
        build(path, project["samples_dir"],
              os.path.join(project["base"], "out_format"))
    assert exc.value.code == ExitCode.UNSUPPORTED_FORMAT


def test_mapping_conflict_negative_bpm(project):
    chart = _base_chart(project)
    chart["bpm_points"][1]["bpm"] = -10.0
    path = os.path.join(project["base"], "neg_bpm.json")
    _write_chart(path, chart)
    from rhythmforge.timing import load_chart

    with pytest.raises(BuildError) as exc:
        load_chart(path)
    assert exc.value.code == ExitCode.MAPPING_CONFLICT


def test_mapping_conflict_section_gap(project):
    chart = _base_chart(project)
    chart["sections"][1]["start_beat"] = 9.0
    path = os.path.join(project["base"], "gap.json")
    _write_chart(path, chart)
    from rhythmforge.timing import load_chart

    with pytest.raises(BuildError) as exc:
        load_chart(path)
    assert exc.value.code == ExitCode.MAPPING_CONFLICT


def test_unknown_layer_note_rejected(project):
    chart = _base_chart(project)
    chart["notes"].append({"beat": 1.0, "layer": "nonexistent", "velocity": 1.0})
    path = os.path.join(project["base"], "unknown_note.json")
    _write_chart(path, chart)
    from rhythmforge.renderer import parse_layers, parse_notes

    layers = parse_layers(chart, project["samples_dir"])
    with pytest.raises(BuildError) as exc:
        parse_notes(chart, {layer.layer_id for layer in layers})
    assert exc.value.code == ExitCode.CHART_SYNTAX


def test_output_lock_blocks_second_holder(tmp_path):
    with OutputLock(str(tmp_path)):
        with pytest.raises(BuildError) as exc:
            with OutputLock(str(tmp_path)):
                pass
    assert exc.value.code == ExitCode.LOCKED_OUTPUT


def test_cli_returns_json_error_and_exit_code(project):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(repo_root, "src")
    completed = subprocess.run(
        [sys.executable, "-m", "rhythmforge.cli", "build",
         "--chart", os.path.join(project["base"], "nope.json"),
         "--samples-dir", project["samples_dir"],
         "--output", os.path.join(project["base"], "cli_out")],
        capture_output=True, text=True, env=env,
    )
    assert completed.returncode == ExitCode.MISSING_SOURCE
    payload = json.loads(completed.stderr)
    assert payload["ok"] is False
    assert payload["error"]["code"] == ExitCode.MISSING_SOURCE


def test_cli_successful_build_exit_zero(built_project):
    project = built_project
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(repo_root, "src")
    completed = subprocess.run(
        [sys.executable, "-m", "rhythmforge.cli", "verify",
         "--chart", project["chart_path"],
         "--samples-dir", project["samples_dir"],
         "--output", project["output_dir"]],
        capture_output=True, text=True, env=env,
    )
    assert completed.returncode == ExitCode.OK
    assert json.loads(completed.stdout)["ok"] is True
