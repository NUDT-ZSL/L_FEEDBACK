from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

from rhythmforge.cli import verify_build
from rhythmforge.pipeline import build
from rhythmforge.waveio import read_wav


def _walk_hashes(directory):
    result = {}
    for root, _, files in os.walk(directory):
        for name in files:
            if name == ".build.lock":
                continue
            path = os.path.join(root, name)
            result[os.path.relpath(path, directory)] = hashlib.sha256(
                open(path, "rb").read()).hexdigest()
    return result


def test_full_build_and_verify(built_project):
    result = verify_build(built_project["chart_path"],
                          built_project["samples_dir"],
                          built_project["output_dir"])
    assert result["ok"] is True
    assert result["frames"] == result["expected_frames"]
    assert result["stem_max_abs_error"] < result["tolerance"]
    assert result["sample_peak"] <= result["peak_limit"] + 1e-6


def test_noop_rebuild_is_bit_identical(built_project):
    output = built_project["output_dir"]
    before = _walk_hashes(output)
    summary = build(built_project["chart_path"], built_project["samples_dir"],
                    output)
    after = _walk_hashes(output)
    assert summary["rebuilt_sections"] == []
    assert set(summary["reused_sections"]) == {
        section["id"] for section in built_project["chart"]["sections"]}
    assert before == after


def test_only_affected_section_rebuilds(built_project, tmp_path):
    output = built_project["output_dir"]
    build(built_project["chart_path"], built_project["samples_dir"], output)
    cache_dir = os.path.join(output, "cache")
    before = {name: hashlib.sha256(open(os.path.join(cache_dir, name), "rb").read())
              .hexdigest() for name in os.listdir(cache_dir)}

    chart = json.loads(open(built_project["chart_path"]).read())
    chart["sections"][1]["native_bpm"] = 130.0
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(chart, indent=2, sort_keys=True))
    summary = build(str(altered), built_project["samples_dir"], output)

    assert summary["rebuilt_sections"] == [chart["sections"][1]["id"]]
    after = {name: hashlib.sha256(open(os.path.join(cache_dir, name), "rb").read())
             .hexdigest() for name in os.listdir(cache_dir)}
    preserved = [name for name in before if name in after
                 and before[name] == after[name]]
    assert len(preserved) == len(chart["sections"]) - 1


def test_markers_align_to_master_frames(built_project):
    output = built_project["output_dir"]
    markers = json.load(open(os.path.join(output, "markers.json")))
    master, info = read_wav(os.path.join(output, "master.wav"))
    assert markers["total_frames"] == info.frames
    frames = [marker["frame"] for marker in markers["markers"]]
    assert frames == sorted(frames)
    assert all(0 <= frame <= info.frames for frame in frames)
    bpm_frames = [marker["frame"] for marker in markers["markers"]
                  if marker["type"] == "bpm"]
    section_frames = [marker["frame"] for marker in markers["markers"]
                      if marker["type"] == "section"]
    assert len(bpm_frames) == len(built_project["chart"]["bpm_points"])
    assert len(section_frames) == len(built_project["chart"]["sections"]) + 1


def test_normalization_report_exists_and_consistent(built_project):
    report = json.load(open(os.path.join(built_project["output_dir"],
                                         "normalization.json")))
    assert "applied_gain_db" in report
    assert report["frames"] == built_project["manifest"]["total_frames"]
    assert set(report["enabled_layers"]).issubset(set(report["all_layers"]))


def test_force_rebuild_reproduces_identical_audio(project):
    first_output = os.path.join(project["base"], "force_a")
    second_output = os.path.join(project["base"], "force_b")
    build(project["chart_path"], project["samples_dir"], first_output)
    build(project["chart_path"], project["samples_dir"], second_output,
          force=True)
    for name in ("master.wav", "markers.json"):
        left = hashlib.sha256(open(os.path.join(first_output, name), "rb").read())
        right = hashlib.sha256(open(os.path.join(second_output, name), "rb").read())
        assert left.hexdigest() == right.hexdigest()
