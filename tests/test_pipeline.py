"""End-to-end tests for the spatmix pipeline (run with pytest)."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run_cli(*argv):
    proc = subprocess.run(
        [PY, "-m", "spatmix", *argv],
        cwd=REPO, capture_output=True, text=True,
    )
    return proc


def write_scene(tmp_path, scene):
    path = os.path.join(tmp_path, "scene.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(scene, fh)
    return path


def base_scene():
    from spatmix.example import build_example_scene
    return build_example_scene()


def read_events(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def event_types(log):
    return [e["type"] for e in log["events"]]


# ---------------------------------------------------------------- basic render
def test_render_example_51(tmp_path):
    scene_path = write_scene(tmp_path, base_scene())
    out = os.path.join(tmp_path, "out")
    proc = run_cli("render", scene_path, "--out", out)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    result = json.loads(proc.stdout)
    assert result["status"] == "ok"

    data, sr = sf.read(os.path.join(out, "mix.wav"), always_2d=True)
    assert sr == 48000
    assert data.shape[1] == 6                      # 5.1 layout
    assert data.shape[0] == 375 * 256              # ceil(2.0*48000/256) frames
    assert np.max(np.abs(data)) > 0.01             # audible content
    assert np.all(np.abs(data[:, 3]) == 0.0)       # LFE silent

    log = read_events(os.path.join(out, "events.json"))
    assert log["log_version"] == 1
    types = set(event_types(log))
    assert "unknown_material" in types             # unobtanium fallback
    assert "reflection_added" in types             # floor reflection
    assert "reflection_skipped" in types           # far_wall too weak
    assert "occlusion_start" in types              # wall crossing
    assert "occlusion_end" in types
    assert "occlusion_degenerate_start" in types   # source inside box
    assert "doppler_clamp_start" in types          # 40 m/s listener
    assert "source_audible" in types
    assert "source_inaudible" in types
    frames = [e["frame"] for e in log["events"]]
    assert frames == sorted(frames)                # chronological order


def test_render_layout_20(tmp_path):
    scene = base_scene()
    scene["layout"] = "2.0"
    scene_path = write_scene(tmp_path, scene)
    out = os.path.join(tmp_path, "out")
    proc = run_cli("render", scene_path, "--out", out)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    data, _ = sf.read(os.path.join(out, "mix.wav"), always_2d=True)
    assert data.shape[1] == 2


# ------------------------------------------------------------- determinism
def test_determinism_across_processes(tmp_path):
    scene_path = write_scene(tmp_path, base_scene())
    outs = []
    for name in ("run_a", "run_b"):
        out = os.path.join(tmp_path, name)
        proc = run_cli("render", scene_path, "--out", out)
        assert proc.returncode == 0, proc.stderr + proc.stdout
        outs.append(out)
    a, _ = sf.read(os.path.join(outs[0], "mix.wav"), always_2d=True)
    b, _ = sf.read(os.path.join(outs[1], "mix.wav"), always_2d=True)
    np.testing.assert_array_equal(a, b)            # bit-identical samples
    with open(os.path.join(outs[0], "events.json"), "rb") as fh:
        log_a = fh.read()
    with open(os.path.join(outs[1], "events.json"), "rb") as fh:
        log_b = fh.read()
    assert log_a == log_b                          # identical event logs


# ------------------------------------------------------ snapshot and resume
def test_snapshot_resume_bit_identical(tmp_path):
    scene_path = write_scene(tmp_path, base_scene())

    full = os.path.join(tmp_path, "full")
    proc = run_cli("render", scene_path, "--out", full)
    assert proc.returncode == 0, proc.stderr + proc.stdout

    part = os.path.join(tmp_path, "part")
    proc = run_cli("render", scene_path, "--out", part, "--max-frames", "200")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    result = json.loads(proc.stdout)
    assert result["status"] == "partial"
    assert result["frames"] == 200
    snapshot = os.path.join(part, "snapshot.json")
    assert os.path.exists(snapshot)

    proc = run_cli("resume", "--snapshot", snapshot)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    result = json.loads(proc.stdout)
    assert result["status"] == "ok"
    assert result["resumed_from"] == 200

    a, _ = sf.read(os.path.join(full, "mix.wav"), always_2d=True)
    b, _ = sf.read(os.path.join(part, "mix.wav"), always_2d=True)
    np.testing.assert_array_equal(a, b)            # sample-identical
    with open(os.path.join(full, "events.json"), "rb") as fh:
        log_a = fh.read()
    with open(os.path.join(part, "events.json"), "rb") as fh:
        log_b = fh.read()
    assert log_a == log_b                          # event-identical


def test_snapshot_version_mismatch(tmp_path):
    scene_path = write_scene(tmp_path, base_scene())
    part = os.path.join(tmp_path, "part")
    proc = run_cli("render", scene_path, "--out", part, "--max-frames", "100")
    assert proc.returncode == 0
    snapshot = os.path.join(part, "snapshot.json")
    with open(snapshot, "r", encoding="utf-8") as fh:
        state = json.load(fh)
    state["version"] = 999
    with open(snapshot, "w", encoding="utf-8") as fh:
        json.dump(state, fh)
    proc = run_cli("resume", "--snapshot", snapshot)
    assert proc.returncode == 5
    assert json.loads(proc.stdout)["error_code"] == "E_SNAPSHOT_VERSION"


def test_snapshot_scene_mismatch(tmp_path):
    scene_path = write_scene(tmp_path, base_scene())
    part = os.path.join(tmp_path, "part")
    proc = run_cli("render", scene_path, "--out", part, "--max-frames", "100")
    assert proc.returncode == 0
    other = base_scene()
    other["duration"] = 1.5
    other_path = write_scene(tmp_path, other)
    proc = run_cli("resume", "--snapshot", os.path.join(part, "snapshot.json"),
                   "--scene", other_path)
    assert proc.returncode == 6
    assert json.loads(proc.stdout)["error_code"] == "E_SNAPSHOT_MISMATCH"


# ------------------------------------------------------------- input errors
def test_error_track_order(tmp_path):
    scene = base_scene()
    scene["listener"]["keyframes"][1]["t"] = 0.5
    scene["listener"]["keyframes"][2]["t"] = 0.5   # not strictly increasing
    scene_path = write_scene(tmp_path, scene)
    proc = run_cli("render", scene_path, "--out", os.path.join(tmp_path, "o"))
    assert proc.returncode == 3
    assert json.loads(proc.stdout)["error_code"] == "E_TRACK_ORDER"


def test_error_range_nan(tmp_path):
    scene = base_scene()
    path = os.path.join(tmp_path, "scene.json")
    text = json.dumps(scene).replace(
        '"pos": [-40.0, 0.0, 1.6]', '"pos": [-40.0, NaN, 1.6]')
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    proc = run_cli("render", path, "--out", os.path.join(tmp_path, "o"))
    assert proc.returncode == 4
    assert json.loads(proc.stdout)["error_code"] == "E_RANGE"


def test_error_range_frequency(tmp_path):
    scene = base_scene()
    scene["sources"][0]["frequency"] = 30000.0     # above Nyquist
    scene_path = write_scene(tmp_path, scene)
    proc = run_cli("render", scene_path, "--out", os.path.join(tmp_path, "o"))
    assert proc.returncode == 4
    assert json.loads(proc.stdout)["error_code"] == "E_RANGE"


def test_error_schema_missing_field(tmp_path):
    scene = base_scene()
    del scene["sources"]
    scene_path = write_scene(tmp_path, scene)
    proc = run_cli("render", scene_path, "--out", os.path.join(tmp_path, "o"))
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["error_code"] == "E_SCHEMA"


def test_error_schema_bad_json(tmp_path):
    path = os.path.join(tmp_path, "scene.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{not json")
    proc = run_cli("render", path, "--out", os.path.join(tmp_path, "o"))
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["error_code"] == "E_SCHEMA"


# ----------------------------------------------------------- dsp unit checks
def test_noise_seed_deterministic():
    from spatmix.synth import DrySource
    spec = {"id": "n1", "waveform": "noise", "frequency": 1000.0,
            "duration": 0.1, "pos": [0, 0, 0], "amplitude": 1.0,
            "material": "default"}
    a = DrySource(spec, 48000)
    b = DrySource(spec, 48000)
    pos = np.arange(100, dtype=np.float64)
    np.testing.assert_array_equal(a.read(pos), b.read(pos))


def test_occlusion_degenerate_rule():
    from spatmix.geometry import point_in_aabb, segment_intersects_aabb
    omin, omax = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    assert point_in_aabb([0.5, 0.5, 0.5], omin, omax)
    assert not point_in_aabb([2.0, 0.5, 0.5], omin, omax)
    assert segment_intersects_aabb([-1.0, 0.5, 0.5], [2.0, 0.5, 0.5], omin, omax)
    assert not segment_intersects_aabb([-1.0, 5.0, 0.5], [2.0, 5.0, 0.5],
                                       omin, omax)


def test_vbap_front_center_51():
    from spatmix.panning import vbap_gains
    gains = vbap_gains(0.0, "5.1")
    assert gains[2] == pytest.approx(1.0)          # all energy in C
    assert gains[3] == 0.0                         # LFE always silent
    gains = vbap_gains(-30.0, "2.0")
    assert gains[0] == pytest.approx(1.0)
    assert gains[1] == pytest.approx(0.0)
    rear = vbap_gains(180.0, "2.0")                # defined fallback
    assert rear[0] == pytest.approx(rear[1])
