from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_cli(arguments, cwd=ROOT):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "render_terrain.py"), *arguments],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )


def test_cli_repeated_runs_are_byte_identical(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    common = [
        "--height-size",
        "17",
        "--resolution",
        "64",
        "48",
        "--cascades",
        "2",
        "--shadow-size",
        "64",
        "--tile-size",
        "16",
        "--unit-scale",
        "0.4",
        "--height-scale",
        "2.0",
    ]
    first = _run_cli([*common, "--output-dir", str(first_dir)])
    second = _run_cli([*common, "--output-dir", str(second_dir)])
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    for name in ("color.png", "depth.png", "shadow_visibility.png"):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
    first_manifest = json.loads((first_dir / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second_dir / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["summary_sha256"] == second_manifest["summary_sha256"]
    assert first_manifest["warning_codes"] == []
    assert first_manifest["cascades"]["count"] == 2


def test_cli_rejects_invalid_parameters_before_writing(tmp_path):
    mismatch = tmp_path / "mismatch.npy"
    np.save(mismatch, np.ones((4, 5, 5), dtype=np.float64) / 4.0)
    cases = [
        [
            "--height-size",
            "9",
            "--resolution",
            "-8",
            "24",
            "--cascades",
            "1",
        ],
        [
            "--height-size",
            "9",
            "--resolution",
            "32",
            "24",
            "--cascades",
            "0",
        ],
        [
            "--height-size",
            "9",
            "--resolution",
            "32",
            "24",
            "--cascades",
            "1",
            "--shadow-size",
            "32",
            "--dirty-rect",
            "0",
            "0",
            "99",
            "4",
        ],
        [
            "--height-size",
            "9",
            "--resolution",
            "32",
            "24",
            "--cascades",
            "1",
            "--shadow-size",
            "32",
            "--height-file",
            str(tmp_path / "missing-height.npy"),
        ],
        [
            "--height-size",
            "9",
            "--resolution",
            "32",
            "24",
            "--cascades",
            "1",
            "--shadow-size",
            "32",
            "--weights-file",
            str(mismatch),
        ],
    ]
    for index, arguments in enumerate(cases):
        output = tmp_path / f"invalid-{index}"
        result = _run_cli([*arguments, "--output-dir", str(output)])
        assert result.returncode != 0
        assert result.stderr.strip()
        assert "ERROR" in result.stderr
        assert not output.exists()


def test_cli_cache_incremental_update(tmp_path):
    cache = tmp_path / "render.cache.npz"
    base_height = tmp_path / "height-base.npy"
    changed_height = tmp_path / "height-changed.npy"
    height = np.zeros((17, 17), dtype=np.float64)
    height[:, 7:10] = 1.0
    np.save(base_height, height)
    changed = height.copy()
    changed[4:6, 4:6] += 0.2
    np.save(changed_height, changed)
    common = [
        "--resolution",
        "160",
        "120",
        "--cascades",
        "2",
        "--shadow-size",
        "64",
        "--tile-size",
        "16",
        "--unit-scale",
        "0.4",
        "--height-scale",
        "2.0",
        "--light-direction",
        "-1.0,-0.35,-0.1",
    ]
    output = tmp_path / "out"
    first = _run_cli(
        [
            *common,
            "--height-file",
            str(base_height),
            "--cache-file",
            str(cache),
            "--output-dir",
            str(output),
        ]
    )
    assert first.returncode == 0, first.stderr
    second = _run_cli(
        [
            *common,
            "--height-file",
            str(changed_height),
            "--cache-file",
            str(cache),
            "--dirty-rect",
            "3",
            "3",
            "7",
            "7",
            "--output-dir",
            str(output),
        ]
    )
    assert second.returncode == 0, second.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["incremental"]["mode"] == "incremental"
    assert manifest["incremental"]["full_rebuild"] is False
    assert manifest["incremental"]["rendered_tiles"] < manifest["incremental"]["total_tiles"]
