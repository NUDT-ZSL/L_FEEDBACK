"""End-to-end and unit tests for the terrain baking pipeline.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from terrain_baker.config import load_config
from terrain_baker.consistency import geometric_vertex_normals, validate_chunk
from terrain_baker.errors import ErrorCode
from terrain_baker.heightfield import HeightField, generate_heightfield
from terrain_baker.lod_mesh import (build_chunk_mesh, build_stitch_strip,
                                    select_lod_for_distance, stitch_meshes)
from terrain_baker.normalmap import (bake_normal_map, decode_rgb8,
                                     normals_from_heights)

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.example.json"


def small_config(path: Path, seed: int = 42) -> Path:
    cfg = {
        "seed": seed,
        "heightfield": {
            "resolution": 65, "chunk_size": 33, "world_size": 256.0,
            "min_height": 0.0, "max_height": 64.0,
            "noise": {"octaves": 4, "base_frequency": 4.0,
                      "gain": 0.5, "lacunarity": 2.0},
        },
        "lods": {"levels": 3, "normal_threshold_deg": 1.0},
        "view": {"position": [128.0, 128.0, 200.0]},
    }
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


class TestHeightfield(unittest.TestCase):
    def test_deterministic(self):
        cfg = load_config(CONFIG)
        a = generate_heightfield(cfg)
        b = generate_heightfield(cfg)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.dtype, np.float64)
        self.assertGreater(a.max() - a.min(), 0.0)

    def test_seed_changes_output(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = small_config(Path(td) / "a.json", seed=1)
            p2 = small_config(Path(td) / "b.json", seed=2)
            ha = generate_heightfield(load_config(p1))
            hb = generate_heightfield(load_config(p2))
            self.assertFalse(np.array_equal(ha, hb))

    def test_chunk_query_and_clamp(self):
        cfg = load_config(CONFIG)
        field = HeightField(cfg)
        xs, ys, h = field.chunk_grid(0, 0, step=1)
        n = cfg.heightfield.chunk_size
        self.assertEqual(h.shape, (n, n))
        # neighbouring chunk shares the border vertices exactly
        _, _, h_east = field.chunk_grid(1, 0, step=1)
        np.testing.assert_array_equal(h[:, -1], h_east[:, 0])
        # clamped sampling
        self.assertEqual(field.sample_grid(np.array([-5]), np.array([0]))[0],
                         field.heights[0, 0])
        v = field.sample_world(np.array([-10.0]), np.array([1e9]))
        self.assertAlmostEqual(v[0], field.heights[-1, 0])


class TestNormalMap(unittest.TestCase):
    def test_flat_field_gives_up_normal(self):
        cfg = load_config(CONFIG)
        field = HeightField(cfg, heights=np.full((65, 65), 10.0))
        img = bake_normal_map(field, 0, 0)
        rgb = np.asarray(img)
        # up normal encodes to (128, 128, 255)
        self.assertTrue(np.all(rgb[:, :, 2] == 255))
        self.assertTrue(np.all(rgb[:, :, 0] == 128))
        self.assertTrue(np.all(rgb[:, :, 1] == 128))

    def test_tilt_direction(self):
        # height increases towards +x -> normal tilts towards -x
        cfg = load_config(CONFIG)
        ramp = np.tile(np.linspace(0, 10, 65), (65, 1))
        field = HeightField(cfg, heights=ramp)
        img = bake_normal_map(field, 0, 0)
        n = decode_rgb8(np.asarray(img))
        self.assertLess(n[32, 32, 0], 0.0)
        self.assertAlmostEqual(n[32, 32, 1], 0.0, places=2)
        self.assertGreater(n[32, 32, 2], 0.9)

    def test_border_uses_clamped_sampling(self):
        # ramp continues past the border; clamped border derivative must be
        # half the interior derivative, not the full one-sided difference.
        ramp = np.tile(np.arange(8, dtype=np.float64), (8, 1))
        normals = normals_from_heights(ramp, 1.0)
        interior = normals[4, 4]
        border = normals[4, 0]
        ratio = (-border[0] / border[2]) / (-interior[0] / interior[2])
        self.assertAlmostEqual(ratio, 0.5, places=6)


class TestLodMesh(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(CONFIG)
        self.field = HeightField(self.cfg)

    def test_grid_sizes_and_alignment(self):
        m0 = build_chunk_mesh(self.field, 0, 0, 0)
        m2 = build_chunk_mesh(self.field, 0, 0, 2)
        cells = self.cfg.heightfield.chunk_size - 1
        self.assertEqual(m0.grid_size, cells + 1)
        self.assertEqual(m2.grid_size, cells // 4 + 1)
        # coarse vertices coincide with every 4th fine vertex
        fine = m0.vertices.reshape(m0.grid_size, m0.grid_size, 3)
        coarse = m2.vertices.reshape(m2.grid_size, m2.grid_size, 3)
        np.testing.assert_allclose(fine[::4, ::4], coarse)

    def test_neighbour_border_positions_match(self):
        a = build_chunk_mesh(self.field, 0, 0, 1)
        b = build_chunk_mesh(self.field, 1, 0, 1)
        ea = a.vertices[a.edges["east"]]
        wb = b.vertices[b.edges["west"]]
        np.testing.assert_allclose(ea, wb)

    def test_stitch_strip_covers_fine_edge(self):
        fine = build_chunk_mesh(self.field, 0, 0, 0)
        coarse = build_chunk_mesh(self.field, 1, 0, 1)
        strip = stitch_meshes(fine, coarse, "east")
        referenced = set(strip.ravel().tolist())
        fine_edge = set(fine.edges["east"].tolist())
        # every fine edge vertex is referenced -> no T-junction gap
        self.assertTrue(fine_edge <= referenced)
        coarse_edge = set(coarse.edges["west"].tolist())
        self.assertTrue(coarse_edge <= referenced)
        # fine edge vertices at even positions coincide with coarse vertices
        fe = fine.vertices[fine.edges["east"]]
        ce = coarse.vertices[coarse.edges["west"]]
        np.testing.assert_allclose(fe[::2], ce)

    def test_stitch_rejects_incompatible_edges(self):
        with self.assertRaises(ValueError):
            build_stitch_strip(np.array([0, 1, 2]), np.array([0, 1, 2, 3]))

    def test_lod_selection_monotonic(self):
        near = select_lod_for_distance(50.0, 64.0, 3)
        far = select_lod_for_distance(5000.0, 64.0, 3)
        self.assertLessEqual(near, far)
        self.assertEqual(far, 3)


class TestConsistency(unittest.TestCase):
    def test_mesh_matches_baked_normalmap(self):
        cfg = load_config(CONFIG)
        field = HeightField(cfg)
        for lod in range(cfg.lod_levels):
            mesh = build_chunk_mesh(field, 1, 2, lod)
            img = bake_normal_map(field, 1, 2, step=1 << lod)
            result = validate_chunk(mesh, np.asarray(img), threshold_deg=1.0)
            # RGB8 quantisation limits precision to well under 1 degree
            self.assertLess(result.max_angle_deg, 0.5)
            self.assertTrue(result.ok)

    def test_geometric_normals_match_stencil(self):
        cfg = load_config(CONFIG)
        field = HeightField(cfg)
        mesh = build_chunk_mesh(field, 0, 0, 0)
        geo = geometric_vertex_normals(mesh)
        n = mesh.grid_size
        _, _, h = field.chunk_grid(0, 0, 1)
        ref = normals_from_heights(h, field.spacing)
        dot = np.clip(np.sum(geo.reshape(n, n, 3) * ref, axis=-1), -1, 1)
        # exact stencil match up to float64 accumulation ordering
        self.assertLess(np.degrees(np.arccos(dot)).max(), 1e-4)

    def test_mismatch_is_flagged(self):
        cfg = load_config(CONFIG)
        field = HeightField(cfg)
        mesh = build_chunk_mesh(field, 0, 0, 0)
        img = np.asarray(bake_normal_map(field, 0, 0)).copy()
        img[:, :, 2] = 0  # corrupt: normals point sideways
        result = validate_chunk(mesh, img, threshold_deg=1.0)
        self.assertFalse(result.ok)
        self.assertGreater(result.bad_vertices, 0)


class TestCliAndRecovery(unittest.TestCase):
    def run_cli(self, args, env_extra=None, cwd=ROOT):
        env = dict(os.environ)
        if env_extra:
            env.update(env_extra)
        return subprocess.run(
            [sys.executable, "-m", "terrain_baker", *args],
            cwd=cwd, env=env, capture_output=True, text=True, timeout=300)

    def test_full_bake_and_report(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = small_config(Path(td) / "cfg.json")
            out = Path(td) / "out"
            proc = self.run_cli(["--config", str(cfg_path), "--output", str(out),
                                 "--jobs", "4"])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["summary"]["failed"], 0)
            self.assertEqual(report["summary"]["consistency_errors"], 0)
            # 2x2 chunks x 3 lods
            self.assertEqual(report["summary"]["total_tasks"], 12)
            self.assertEqual(report["summary"]["done"], 12)
            chunk = report["chunks"][0]
            self.assertEqual(len(chunk["lods"]), 3)
            for lod in chunk["lods"]:
                self.assertTrue(Path(lod["mesh"]).exists())
                self.assertTrue(Path(lod["normal_map"]).exists())
                img = Image.open(lod["normal_map"])
                self.assertEqual(img.mode, "RGB")
                self.assertIsNone(lod["error"])
            # error codes are string business codes
            self.assertIsInstance(chunk["status"], str)

    def test_crash_resume_skips_done_tasks(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = small_config(Path(td) / "cfg.json")
            out = Path(td) / "out"
            # hard crash (os._exit) inside task 1,1,2
            proc = self.run_cli(
                ["--config", str(cfg_path), "--output", str(out), "--jobs", "2"],
                env_extra={"TERRAIN_BAKER_CRASH_AT": "1,1,2"})
            self.assertEqual(proc.returncode, 3)
            # state file survives the crash as valid JSON
            state = json.loads((out / "_state" / "bake_state.json").read_text())
            self.assertTrue(state["tasks"], "expected persisted task state")
            # resume: interrupted (running) tasks are re-queued and finish
            proc = self.run_cli(["--config", str(cfg_path), "--output", str(out),
                                 "--resume", "--jobs", "2"])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["summary"]["done"], 12)
            self.assertEqual(report["summary"]["failed"], 0)

    def test_failed_task_retry_on_resume(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = small_config(Path(td) / "cfg.json")
            out = Path(td) / "out"
            proc = self.run_cli(
                ["--config", str(cfg_path), "--output", str(out), "--jobs", "3"],
                env_extra={"TERRAIN_BAKER_FAIL_TASKS": "0,1,1"})
            self.assertEqual(proc.returncode, 3)
            state = json.loads((out / "_state" / "bake_state.json").read_text())
            self.assertEqual(state["tasks"]["0,1,1"]["status"], "failed")
            self.assertEqual(state["tasks"]["0,1,1"]["error"]["code"],
                             ErrorCode.E_TASK_EXECUTION)
            # record outputs of tasks that completed before the failure
            done_before = {k for k, v in state["tasks"].items()
                           if v.get("status") == "done"}
            self.assertTrue(done_before)
            mtimes = {}
            for key in done_before:
                cx, cy, lod = key.split(",")
                p = out / f"chunk_{int(cx):03d}_{int(cy):03d}" / f"mesh_lod{lod}.npz"
                mtimes[str(p)] = p.stat().st_mtime_ns
            # resume without the fault injection recovers
            proc = self.run_cli(["--config", str(cfg_path), "--output", str(out),
                                 "--resume", "--jobs", "3"])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["summary"]["failed"], 0)
            # completed tasks were not re-baked
            for path, mtime in mtimes.items():
                self.assertEqual(Path(path).stat().st_mtime_ns, mtime,
                                 f"done task was re-baked: {path}")

    def test_bad_config_exit_code(self):
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "bad.json"
            bad.write_text('{"seed": 1, "heightfield": {"resolution": 65,'
                           ' "chunk_size": 30}}', encoding="utf-8")
            proc = self.run_cli(["--config", str(bad), "--output",
                                 str(Path(td) / "out")])
            self.assertEqual(proc.returncode, 1)
            self.assertIn(ErrorCode.E_CONFIG, proc.stderr)

    def test_missing_config_exit_code(self):
        proc = self.run_cli(["--config", "no_such_file.json", "--output", "x"])
        self.assertEqual(proc.returncode, 1)


if __name__ == "__main__":
    unittest.main()
