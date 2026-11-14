"""端到端与单元测试：合成、缓存、反射、错误码、批量与确定性。"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from matrelight import blend
from matrelight.basis import reconstruct_tangent_basis
from matrelight.cache import read_cache
from matrelight.deterministic import DeterministicContext
from matrelight.errors import ERROR_CODES, MaterialError
from matrelight.material import parse_material
from matrelight.pipeline import process_material

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = REPO_ROOT / "samples"


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (SAMPLES / "material_mixed.yaml").exists():
            subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" / "make_samples.py"), "-o", str(SAMPLES)],
                check=True,
                cwd=REPO_ROOT,
            )
        cls.tmp_root = REPO_ROOT / ".test_work"
        cls.tmp_root.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_root, ignore_errors=True)

    def test_01_mixed_material_outputs(self):
        out = self.tmp_root / "run_mixed"
        result = process_material(SAMPLES / "material_mixed.yaml", out, seed=777)
        material_dir = Path(result.output_dir)
        for name in (
            "normal_composite.png",
            "reflection_light_00.npy",
            "reflection_light_00.png",
            "reflection_light_01.npy",
            "reflection_light_02.npy",
            "material_cache.mzc",
            "result.json",
        ):
            self.assertTrue((material_dir / name).is_file(), name)

        metadata, arrays = read_cache(material_dir / "material_cache.mzc")
        self.assertEqual(metadata["seed"], 777)
        self.assertEqual(metadata["normal_encoding"], "opengl_plus_y")
        self.assertEqual(metadata["brdf"], "ggx_trowbridge_reitz")

        # 合成法线必须是单位向量
        norms = np.linalg.norm(arrays["normal"], axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

        # 缓存归一化权重之和为 1，且与每层贡献中记录的一致
        weights = np.asarray(metadata["normalized_weights"])
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
        contrib_weights = [c["normalized_weight"] for c in metadata["layer_contributions"]]
        np.testing.assert_allclose(weights, contrib_weights, rtol=0, atol=1e-15)

        # 切线基正交且为右手系
        t = arrays["tangent"]
        b = arrays["bitangent"]
        n = arrays["normal"]
        np.testing.assert_allclose(np.sum(t * n, axis=-1), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.sum(t * b, axis=-1), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.cross(n, t), b, atol=1e-6)

        # 修正标记形状与阈值被写入元数据
        self.assertEqual(
            metadata["tangent_basis"]["correction_marker_shape"], [64, 64]
        )
        self.assertEqual(metadata["tangent_basis"]["threshold_deg"], 30.0)
        self.assertEqual(arrays["correction_marker"].shape, (64, 64))

        # 最终 BRDF 参数存在且粗糙度在合法范围
        params = metadata["final_brdf_params"]
        self.assertGreaterEqual(params["roughness_min"], 0.0)
        self.assertLessEqual(params["roughness_max"], 1.0)

        # 反射图非负且三盏灯都有输出
        for i in range(3):
            hdr = np.load(material_dir / f"reflection_light_{i:02d}.npy")
            self.assertEqual(hdr.shape, (64, 64, 3))
            self.assertTrue(np.isfinite(hdr).all())
            self.assertGreater(float(hdr.sum()), 0.0)

    def test_02_weighted_duplicate_layers(self):
        out = self.tmp_root / "run_weighted"
        result = process_material(SAMPLES / "material_weighted.yaml", out, seed=42)
        metadata, arrays = read_cache(Path(result.output_dir) / "material_cache.mzc")
        names = [c["name"] for c in metadata["layer_contributions"]]
        self.assertEqual(names, ["base", "detail", "detail_duplicate"])
        # 权重 2:1:1 归一化
        np.testing.assert_allclose(
            metadata["normalized_weights"], [0.5, 0.25, 0.25], rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            np.linalg.norm(arrays["normal"], axis=-1), 1.0, atol=1e-6
        )

    def test_03_determinism_byte_identical(self):
        out_a = self.tmp_root / "det_a"
        out_b = self.tmp_root / "det_b"
        process_material(SAMPLES / "material_mixed.yaml", out_a, seed=909)
        process_material(SAMPLES / "material_mixed.yaml", out_b, seed=909)
        files_a = sorted(p.relative_to(out_a) for p in out_a.rglob("*") if p.is_file())
        files_b = sorted(p.relative_to(out_b) for p in out_b.rglob("*") if p.is_file())
        self.assertEqual(files_a, files_b)
        for rel in files_a:
            self.assertEqual(
                (out_a / rel).read_bytes(),
                (out_b / rel).read_bytes(),
                f"同种子输出不一致: {rel}",
            )

    def test_04_different_seed_changes_cache(self):
        out_a = self.tmp_root / "seed_a"
        out_b = self.tmp_root / "seed_b"
        process_material(SAMPLES / "material_weighted.yaml", out_a, seed=1)
        process_material(SAMPLES / "material_weighted.yaml", out_b, seed=2)
        cache_a = (out_a / "weighted_material" / "material_cache.mzc").read_bytes()
        cache_b = (out_b / "weighted_material" / "material_cache.mzc").read_bytes()
        self.assertNotEqual(cache_a, cache_b)

    def test_05_error_mix_mode(self):
        with self.assertRaises(MaterialError) as caught:
            parse_material(SAMPLES / "case_bad_mode.yaml")
        self.assertEqual(caught.exception.code, "E_MIX_MODE")
        self.assertEqual(caught.exception.exit_code, ERROR_CODES["E_MIX_MODE"])

    def test_06_error_uv_out_of_bounds(self):
        with self.assertRaises(MaterialError) as caught:
            process_material(SAMPLES / "case_uv_oob.yaml", self.tmp_root / "oob", seed=1)
        self.assertEqual(caught.exception.code, "E_UV_OUT_OF_BOUNDS")
        # 失败项不得留下部分输出
        self.assertFalse((self.tmp_root / "oob").exists())

    def test_07_error_missing_input(self):
        with self.assertRaises(MaterialError) as caught:
            parse_material(SAMPLES / "not_here.yaml")
        self.assertEqual(caught.exception.code, "E_INPUT_MISSING")
        with self.assertRaises(MaterialError) as caught2:
            process_material(
                SAMPLES / "case_missing_texture.yaml",
                self.tmp_root / "missing_tex",
                seed=1,
            )
        self.assertEqual(caught2.exception.code, "E_INPUT_MISSING")
        self.assertFalse((self.tmp_root / "missing_tex").exists())

    def test_08_error_invalid_layer(self):
        with self.assertRaises(MaterialError) as caught:
            process_material(
                SAMPLES / "case_invalid_layer.yaml",
                self.tmp_root / "invalid",
                seed=1,
            )
        self.assertEqual(caught.exception.code, "E_LAYER_INVALID")
        self.assertFalse((self.tmp_root / "invalid").exists())

    def test_09_seeded_normalize_order(self):
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        vectors = np.random.default_rng(9).normal(size=(20, 3))
        a = DeterministicContext.seeded_normalize(rng1, vectors)
        b = DeterministicContext.seeded_normalize(rng2, vectors)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0)

    def test_09b_tangent_discontinuity_correction(self):
        # 右半法线方向符号翻转，接缝处切线夹角达到 180 度
        size = 32
        xs = np.linspace(0.2, 1.0, size)
        nx = np.tile(xs, (size, 1))
        nx[:, size // 2:] = -nx[:, size // 2:]
        nx *= 0.6
        ny = np.zeros((size, size))
        nz = np.sqrt(np.clip(1.0 - nx * nx, 1e-6, 1.0))
        normals = np.stack([nx, ny, nz], axis=-1)
        normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
        rng = np.random.default_rng(7)
        tangents, bitangents, meta, marker = reconstruct_tangent_basis(
            normals, threshold_deg=5.0, rng=rng
        )
        self.assertGreater(meta["correction_count"], 0)
        self.assertGreater(int(marker.sum()), 0)
        self.assertGreaterEqual(meta["max_adjacent_angle_deg"], 179.0)
        # 修正只改切线基：法线长度与方向保持
        np.testing.assert_allclose(
            np.linalg.norm(normals, axis=-1), 1.0, atol=1e-9
        )
        np.testing.assert_allclose((tangents * normals).sum(axis=-1), 0.0, atol=1e-9)
        np.testing.assert_allclose(np.cross(normals, tangents), bitangents, atol=1e-9)

    def test_10_weight_normalization_matches_render(self):
        spec = parse_material(SAMPLES / "material_weighted.yaml")
        weights = blend.normalized_weights(spec.layers)
        out = self.tmp_root / "weight_consistency"
        result = process_material(SAMPLES / "material_weighted.yaml", out, seed=55)
        metadata, _ = read_cache(Path(result.output_dir) / "material_cache.mzc")
        np.testing.assert_allclose(
            weights, metadata["normalized_weights"], rtol=0, atol=1e-15
        )

    def test_11_seam_material_triggers_basis_correction(self):
        out = self.tmp_root / "run_seam"
        result = process_material(SAMPLES / "material_seam.yaml", out, seed=7)
        metadata, arrays = read_cache(Path(result.output_dir) / "material_cache.mzc")
        self.assertEqual(metadata["tangent_basis"]["threshold_deg"], 5.0)
        self.assertGreater(metadata["tangent_basis"]["correction_count"], 0)
        self.assertGreater(float(arrays["correction_marker"].sum()), 0.0)
        # 修正不改变法线长度
        np.testing.assert_allclose(
            np.linalg.norm(arrays["normal"], axis=-1), 1.0, atol=1e-6
        )


class CliBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (SAMPLES / "material_mixed.yaml").exists():
            subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" / "make_samples.py"), "-o", str(SAMPLES)],
                check=True,
                cwd=REPO_ROOT,
            )
        cls.tmp_root = REPO_ROOT / ".test_batch"
        cls.tmp_root.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_root, ignore_errors=True)

    def test_batch_summary_order_and_isolation(self):
        inputs = [
            SAMPLES / "material_mixed.yaml",
            SAMPLES / "case_bad_mode.yaml",
            SAMPLES / "material_weighted.yaml",
            SAMPLES / "case_uv_oob.yaml",
        ]
        cmd = [
            sys.executable,
            "-m",
            "matrelight",
            "batch",
            *map(str, inputs),
            "-o",
            str(self.tmp_root / "out"),
            "--seed",
            "31337",
            "-j",
            "2",
        ]
        env = dict(PYTHONPATH=str(REPO_ROOT))
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        summary = json.loads(
            (self.tmp_root / "out" / "batch_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["failure_count"], 2)
        # 按输入顺序记录
        self.assertEqual(
            [entry["input"] for entry in summary["entries"]],
            [str(p.resolve()) for p in inputs],
        )
        statuses = [entry["status"] for entry in summary["entries"]]
        self.assertEqual(
            statuses, ["OK", "E_MIX_MODE", "OK", "E_UV_OUT_OF_BOUNDS"]
        )
        for entry in summary["entries"]:
            self.assertEqual(entry["exit_code"], 0 if entry["ok"] else ERROR_CODES[entry["error_code"]])
        # 失败项没有输出目录
        self.assertFalse((self.tmp_root / "out" / "bad_mode").exists())
        self.assertFalse((self.tmp_root / "out" / "uv_oob").exists())
        self.assertTrue((self.tmp_root / "out" / "mixed_material").is_dir())

    def test_render_cli_structured_error(self):
        cmd = [
            sys.executable,
            "-m",
            "matrelight",
            "render",
            str(SAMPLES / "case_bad_mode.yaml"),
            "-o",
            str(self.tmp_root / "cli_fail"),
        ]
        env = dict(PYTHONPATH=str(REPO_ROOT))
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(proc.returncode, ERROR_CODES["E_MIX_MODE"])
        payload = json.loads(proc.stderr.strip())
        self.assertEqual(payload["error_code"], "E_MIX_MODE")
        self.assertEqual(payload["exit_code"], ERROR_CODES["E_MIX_MODE"])
        self.assertFalse(payload["ok"])

    def test_batch_deterministic_summary(self):
        inputs = [SAMPLES / "material_mixed.yaml", SAMPLES / "material_weighted.yaml"]
        env = dict(PYTHONPATH=str(REPO_ROOT))
        out1 = self.tmp_root / "b1"
        out2 = self.tmp_root / "b2"
        common = [
            sys.executable,
            "-m",
            "matrelight",
            "batch",
            *map(str, inputs),
            "--seed",
            "8080",
            "-j",
            "2",
        ]
        subprocess.run(common + ["-o", str(out1)], cwd=REPO_ROOT, env=env, check=True)
        subprocess.run(common + ["-o", str(out2)], cwd=REPO_ROOT, env=env, check=True)
        s1 = json.loads((out1 / "batch_summary.json").read_text())
        s2 = json.loads((out2 / "batch_summary.json").read_text())
        # output_dir 是绝对路径，跨目录比较时剔除；其余字段必须完全一致
        for summary in (s1, s2):
            for entry in summary["entries"]:
                entry.pop("output_dir", None)
        self.assertEqual(s1, s2)
        for rel in ("mixed_material/material_cache.mzc", "weighted_material/material_cache.mzc"):
            self.assertEqual((out1 / rel).read_bytes(), (out2 / rel).read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
