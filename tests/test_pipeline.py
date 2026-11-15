"""Plain-assert test suite for skeletal_baker (no pytest required).

Run: python tests/test_pipeline.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skeletal_baker.animator import SkeletalAnimator
from skeletal_baker.clip import OrthoFrustum, clip_frame
from skeletal_baker.errors import (
    ErrorLog,
    E_DEGENERATE_TRIANGLE,
    E_NON_INVERTIBLE_MATRIX,
)
from skeletal_baker.normals import recompute_normals
from skeletal_baker.pipeline import bake
from skeletal_baker.skinning import skin_frame


def test_weights_normalized():
    anim = SkeletalAnimator(seed=7)
    sums = anim.weights.sum(axis=1)
    assert np.allclose(sums, 1.0), "skin weights must sum to 1"
    assert (anim.weights >= 0).all()


def test_fk_finite_and_deterministic():
    anim_a = SkeletalAnimator(seed=3)
    anim_b = SkeletalAnimator(seed=3)
    for frame in range(5):
        mats_a = anim_a.frame_world_matrices(frame)
        mats_b = anim_b.frame_world_matrices(frame)
        assert mats_a is not None and mats_b is not None
        for mat_a, mat_b in zip(mats_a, mats_b):
            assert np.all(np.isfinite(mat_a))
            assert np.array_equal(mat_a, mat_b), "FK must be deterministic"


def test_non_invertible_matrix():
    anim = SkeletalAnimator(seed=0)
    log = ErrorLog()
    singular = np.eye(4)
    singular[2, 2] = 0.0
    result = skin_frame(anim.vertices, anim.weights,
                        [singular, np.eye(4)], 0, log)
    assert result is None
    assert log.entries[0]["code"] == E_NON_INVERTIBLE_MATRIX


def test_degenerate_triangle_capped():
    positions = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [2.0, 2.0, 0.0], [2.0, 2.0, 0.0], [2.0, 2.0, 0.0],
    ])
    triangles = np.array([[0, 1, 2]] + [[3, 4, 5]] * 15, dtype=np.int32)
    log = ErrorLog()
    normals, face_normals = recompute_normals(positions, triangles, 0, log)
    deg = [e for e in log.entries if e["code"] == E_DEGENERATE_TRIANGLE]
    assert len(deg) == 10, "degenerate reports must be capped at 10"
    assert np.allclose(normals[0], [0, 0, 1])
    assert np.allclose(face_normals[0], [0, 0, 1])


def test_full_cull():
    positions = np.array([[5.0, 5.0, 0.0], [6.0, 5.0, 0.0],
                          [5.0, 6.0, 0.0]])
    normals = np.tile([0.0, 0.0, 1.0], (3, 1))
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    ndc = OrthoFrustum().to_ndc(positions)
    pos, nrm, face_idx, stats = clip_frame(ndc, normals, triangles)
    assert stats.triangles_out == 0 and stats.triangles_culled == 1
    assert len(pos) == 0 and len(nrm) == 0


def test_determinism_and_outputs():
    with tempfile.TemporaryDirectory() as dir1, \
            tempfile.TemporaryDirectory() as dir2:
        assert bake(dir1, frames=24, seed=42, fps=24.0) == 0
        assert bake(dir2, frames=24, seed=42, fps=24.0) == 0
        path1, path2 = Path(dir1), Path(dir2)
        names1 = sorted(p.name for p in path1.iterdir())
        names2 = sorted(p.name for p in path2.iterdir())
        assert names1 == names2
        for name in names1:
            bytes1 = (path1 / name).read_bytes()
            bytes2 = (path2 / name).read_bytes()
            assert bytes1 == bytes2, f"{name} differs between runs"
        # errors.log is valid JSON Lines with expected schema.
        for line in (path1 / "errors.log").read_text().splitlines():
            entry = json.loads(line)
            assert {"frame", "code", "message"} <= set(entry)
            assert entry["code"].startswith("E_")
        # Fully culled frames keep empty vertex buffers.
        culled = {json.loads(line)["frame"]
                  for line in (path1 / "errors.log").read_text().splitlines()
                  if json.loads(line)["code"] == "E_FRAME_FULLY_CULLED"}
        for frame in culled:
            assert (path1 / f"frame_{frame:04d}.vert.bin").stat().st_size == 0
            assert (path1 / f"frame_{frame:04d}.png").exists()
        # Overview grid: 10 columns, rows auto-computed.
        overview = Image.open(path1 / "overview.png")
        frame_png = Image.open(path1 / "frame_0000.png")
        width, height = frame_png.size
        assert overview.size == (10 * width, 3 * height)


def test_cli_invalid_args_exit_2():
    root = Path(__file__).resolve().parent.parent
    bad_argsets = [
        ["--output-dir", "/tmp/x", "--frames", "0"],
        ["--output-dir", "/tmp/x", "--frames", "abc"],
        ["--output-dir", "/tmp/x", "--fps", "-1"],
        ["--output-dir", "/tmp/x", "--seed", "-3"],
        ["--frames", "4"],  # missing required --output-dir
    ]
    for extra in bad_argsets:
        proc = subprocess.run(
            [sys.executable, "-m", "skeletal_baker", *extra],
            capture_output=True, text=True, cwd=root,
        )
        assert proc.returncode == 2, (extra, proc.returncode)
        assert "usage" in proc.stderr.lower()


def test_cli_valid_exit_0():
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as out:
        proc = subprocess.run(
            [sys.executable, "-m", "skeletal_baker",
             "--output-dir", out, "--frames", "8", "--seed", "1",
             "--fps", "30"],
            capture_output=True, text=True, cwd=root,
        )
        assert proc.returncode == 0, proc.stderr
        assert (Path(out) / "overview.png").exists()


if __name__ == "__main__":
    all_tests = [v for k, v in sorted(globals().items())
                 if k.startswith("test_")]
    for test_fn in all_tests:
        test_fn()
        print(f"PASS {test_fn.__name__}")
    print(f"{len(all_tests)} tests passed")
