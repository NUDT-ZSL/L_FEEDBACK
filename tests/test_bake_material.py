"""End-to-end tests for the deterministic material baker."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from generate_samples import generate  # noqa: E402
from material_baker.assets import _decode_image  # noqa: E402
from material_baker.scene import load_scene  # noqa: E402
from material_baker.tangent import build_tangent_field  # noqa: E402


def run_bake(scene: Path, out: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ROOT / "bake_material.py"),
        "--scene",
        str(scene),
        "--out",
        str(out),
        *extra,
    ]
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def tree_hashes(root: Path) -> dict[str, str]:
    import hashlib

    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def png_bit_depth(path: Path) -> int:
    payload = path.read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"not a PNG: {path}")
    return int(payload[24])


class MaterialBakerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory(prefix="material-baker-tests-")
        cls.work = Path(cls._temp.name)
        cls.clean_scene = generate(
            cls.work / "clean",
            width=128,
            height=96,
            variant="clean",
            seed=12345,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def test_end_to_end_outputs_and_determinism(self) -> None:
        first = self.work / "out-a"
        second = self.work / "out-b"
        proc_a = run_bake(self.clean_scene, first, "--force")
        proc_b = run_bake(self.clean_scene, second, "--force")
        self.assertEqual(proc_a.returncode, 0, proc_a.stderr)
        self.assertEqual(proc_b.returncode, 0, proc_b.stderr)
        self.assertEqual(tree_hashes(first), tree_hashes(second))

        required = {
            "baked_color.png",
            "baked_normal.png",
            "material_scalar.png",
            "lighting_diffuse.png",
            "lighting_specular.png",
            "lighting_indirect.png",
            "preview_main.png",
            "preview_normal.png",
            "preview_single_light.png",
            "validation_mask.png",
            "asset_manifest.json",
            "validation_report.json",
            "manifest.json",
            "lighting_components.npy",
            "baked_linear.npy",
            "single_light_linear.npy",
        }
        self.assertTrue(required.issubset({path.name for path in first.iterdir()}))

        report = json.loads((first / "validation_report.json").read_text())
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["normal_space"]["pixels_with_anomaly"], 0)
        self.assertGreater(report["seams"]["mirror_consistent_pairs"], 0)

        assets = json.loads((first / "asset_manifest.json").read_text())
        self.assertEqual(assets["textures"]["normal"]["format"], "tga")
        self.assertIn("linearization", assets["textures"]["base_color"])
        self.assertIn("linearization", assets["textures"]["normal"])
        self.assertIn("linearization", assets["textures"]["roughness"])

        components = np.load(first / "lighting_components.npy")
        self.assertEqual(components.shape, (96, 128, 9))
        self.assertEqual(components.dtype, np.float32)
        baked = np.load(first / "baked_linear.npy")
        self.assertEqual(baked.shape, (96, 128, 3))
        preview = cv2.imread(str(first / "preview_main.png"), cv2.IMREAD_UNCHANGED)
        self.assertEqual(preview.shape[:2], (288, 384))

    def test_chunk_boundaries_are_byte_neutral(self) -> None:
        small = self.work / "chunk-small"
        large = self.work / "chunk-large"
        proc_small = run_bake(
            self.clean_scene, small, "--force", "--chunk-rows", "7"
        )
        proc_large = run_bake(
            self.clean_scene, large, "--force", "--chunk-rows", "37"
        )
        self.assertEqual(proc_small.returncode, 0, proc_small.stderr)
        self.assertEqual(proc_large.returncode, 0, proc_large.stderr)
        for name in (
            "baked_linear.npy",
            "lighting_components.npy",
            "single_light_linear.npy",
            "validation_report.json",
            "baked_color.png",
            "preview_main.png",
        ):
            self.assertEqual(
                (small / name).read_bytes(),
                (large / name).read_bytes(),
                f"chunk boundary changed {name}",
            )
        states = json.loads((small / ".work" / "state.json").read_text())
        self.assertEqual(states["chunk_rows"], 7)
        self.assertNotEqual(
            len(states["completed"]),
            len(
                json.loads((large / ".work" / "state.json").read_text())[
                    "completed"
                ]
            ),
        )

    def test_interrupted_resume_matches_full_run(self) -> None:
        full = self.work / "resume-full"
        recovered = self.work / "resume-recovered"
        proc_full = run_bake(self.clean_scene, full, "--force")
        self.assertEqual(proc_full.returncode, 0, proc_full.stderr)

        proc_crash = run_bake(
            self.clean_scene,
            recovered,
            "--force",
            "--fail-after-chunks",
            "2",
        )
        self.assertEqual(proc_crash.returncode, 70, proc_crash.stderr)
        self.assertIn("checkpoints retained", proc_crash.stderr)
        state = json.loads((recovered / ".work" / "state.json").read_text())
        self.assertEqual(len(state["completed"]), 2)
        stale_checkpoint = (
            recovered / ".work" / "chunks" / ".bake_00000.npy.tmp-999"
        )
        stale_checkpoint.write_bytes(b"orphan")
        stale_output = recovered / ".asset_manifest.json.tmp-999"
        stale_output.write_bytes(b"orphan")

        proc_resume = run_bake(self.clean_scene, recovered)
        self.assertEqual(proc_resume.returncode, 0, proc_resume.stderr)
        self.assertIn("2/4 checkpoint chunks already complete", proc_resume.stderr)
        self.assertIn("[bake] complete chunks=4 new=2 resumed=2", proc_resume.stderr)
        self.assertEqual(tree_hashes(full), tree_hashes(recovered))

    def test_business_error_codes_and_reports(self) -> None:
        original = json.loads(self.clean_scene.read_text(encoding="utf-8"))

        missing_scene = self.clean_scene.parent / "scene_missing.json"
        payload = json.loads(json.dumps(original))
        payload["assets"]["base_color"]["path"] = "textures/missing.png"
        missing_scene.write_text(json.dumps(payload), encoding="utf-8")
        proc = run_bake(missing_scene, self.work / "err-missing", "--force")
        self.assertEqual(proc.returncode, 11, proc.stderr)
        report = json.loads(
            (self.work / "err-missing" / "validation_report.json").read_text()
        )
        self.assertEqual(report["error"]["code"], "E001")

        invalid_scene = self.clean_scene.parent / "scene_invalid.json"
        payload = json.loads(json.dumps(original))
        payload["lights"][0]["direction"] = [0, 0, 0]
        invalid_scene.write_text(json.dumps(payload), encoding="utf-8")
        proc = run_bake(invalid_scene, self.work / "err-invalid", "--force")
        self.assertEqual(proc.returncode, 14, proc.stderr)
        report = json.loads(
            (self.work / "err-invalid" / "validation_report.json").read_text()
        )
        self.assertEqual(report["error"]["code"], "E004")

        rgba_normal = self.clean_scene.parent / "textures" / "normal_rgba.png"
        source = cv2.imread(
            str(self.clean_scene.parent / "textures" / "basecolor.png"),
            cv2.IMREAD_UNCHANGED,
        )
        rgba = np.concatenate(
            [source, np.full((*source.shape[:2], 1), 255, dtype=np.uint8)], axis=2
        )
        self.assertTrue(cv2.imwrite(str(rgba_normal), rgba))
        channels_scene = self.clean_scene.parent / "scene_channels.json"
        payload = json.loads(json.dumps(original))
        payload["assets"]["normal"]["path"] = "textures/normal_rgba.png"
        channels_scene.write_text(json.dumps(payload), encoding="utf-8")
        proc = run_bake(channels_scene, self.work / "err-channels", "--force")
        self.assertEqual(proc.returncode, 13, proc.stderr)
        report = json.loads(
            (self.work / "err-channels" / "validation_report.json").read_text()
        )
        self.assertEqual(report["error"]["code"], "E003")

        nonunit_scene = generate(
            self.work / "nonunit",
            width=64,
            height=64,
            variant="non-unit",
            seed=7,
        )
        proc = run_bake(nonunit_scene, self.work / "err-nonunit", "--force")
        self.assertEqual(proc.returncode, 15, proc.stderr)
        report = json.loads(
            (self.work / "err-nonunit" / "validation_report.json").read_text()
        )
        self.assertEqual(report["error"]["code"], "E005")
        self.assertIn("x", report["error"]["location"])

    def test_anomaly_localization_and_seam_classes(self) -> None:
        seam_scene = generate(
            self.work / "seam-error",
            width=128,
            height=96,
            variant="seam-flip",
            seed=19,
        )
        proc = run_bake(seam_scene, self.work / "seam-out", "--force")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(
            (self.work / "seam-out" / "validation_report.json").read_text()
        )
        self.assertEqual(report["status"], "warning")
        self.assertGreater(report["normal_space"]["pixels_with_anomaly"], 0)
        self.assertGreater(report["seams"]["anomalous_pairs"], 0)
        sample = report["normal_space"]["samples"][0]
        self.assertIn("x", sample)
        self.assertIn("y", sample)
        self.assertIn("tile", sample)
        self.assertIn(sample["type"], {"uv_mismatch", "tspace_flip"})
        seam = report["seams"]["samples"][0]
        self.assertIn("tile", seam)
        self.assertEqual(len(seam["boundary"]), 2)
        self.assertEqual(seam["classification"], "mirror_seam")

        hard_scene = generate(
            self.work / "hard-edge",
            width=128,
            height=96,
            variant="hard-edge",
            seed=23,
        )
        proc = run_bake(hard_scene, self.work / "hard-out", "--force")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        hard_report = json.loads(
            (self.work / "hard-out" / "validation_report.json").read_text()
        )
        self.assertGreater(
            hard_report["seams"]["classification_counts"].get(
                "hard_edge_seam", 0
            ),
            0,
        )

    def test_scalar_defaults_light_order_and_output_controls(self) -> None:
        payload = json.loads(self.clean_scene.read_text(encoding="utf-8"))
        payload["assets"].pop("roughness")
        payload["assets"].pop("metallic")
        payload["assets"]["defaults"] = {"roughness": 0.37, "metallic": 0.11}
        payload["lights"] = list(reversed(payload["lights"]))
        default_scene = self.clean_scene.parent / "scene_defaults.json"
        default_scene.write_text(json.dumps(payload), encoding="utf-8")
        out = self.work / "controls"
        proc = run_bake(
            default_scene,
            out,
            "--force",
            "--bit-depth",
            "16",
            "--resolution",
            "200x150",
            "--name-pattern",
            "asset_{kind}.png",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        assets = json.loads((out / "asset_manifest.json").read_text())
        self.assertEqual(
            assets["textures"]["roughness"]["source"], "scene_default"
        )
        self.assertEqual(assets["textures"]["roughness"]["value"], 0.37)
        self.assertEqual(assets["textures"]["metallic"]["value"], 0.11)
        self.assertEqual(png_bit_depth(out / "asset_baked_color.png"), 16)
        self.assertEqual(png_bit_depth(out / "asset_preview_main.png"), 16)
        preview = cv2.imread(
            str(out / "asset_preview_main.png"), cv2.IMREAD_UNCHANGED
        )
        self.assertEqual(preview.shape[:2], (150, 200))

        original_payload = json.loads(
            self.clean_scene.read_text(encoding="utf-8")
        )
        reversed_payload = json.loads(json.dumps(original_payload))
        reversed_payload["lights"] = list(reversed(original_payload["lights"]))
        reversed_scene = self.clean_scene.parent / "scene_lights_reversed.json"
        reversed_scene.write_text(
            json.dumps(reversed_payload), encoding="utf-8"
        )
        ordered_out = self.work / "light-order-a"
        reversed_out = self.work / "light-order-b"
        ordered_proc = run_bake(self.clean_scene, ordered_out, "--force")
        reversed_proc = run_bake(reversed_scene, reversed_out, "--force")
        self.assertEqual(ordered_proc.returncode, 0, ordered_proc.stderr)
        self.assertEqual(reversed_proc.returncode, 0, reversed_proc.stderr)
        self.assertEqual(
            (ordered_out / "baked_linear.npy").read_bytes(),
            (reversed_out / "baked_linear.npy").read_bytes(),
        )
        self.assertEqual(
            (ordered_out / "preview_single_light.png").read_bytes(),
            (reversed_out / "preview_single_light.png").read_bytes(),
        )

    def test_tga_rle_and_bottom_origin_decode(self) -> None:
        # 2x2 RGB, bottom-left origin. Bottom row: blue, white; top: red, green.
        header = bytearray(18)
        header[2] = 10  # RLE true-color
        header[12:14] = (2).to_bytes(2, "little")
        header[14:16] = (2).to_bytes(2, "little")
        header[16] = 24
        header[17] = 0
        pixels = bytes(
            [
                # BGR blue, BGR white, BGR red, BGR green
                255, 0, 0,
                255, 255, 255,
                0, 0, 255,
                0, 255, 0,
            ]
        )
        payload = bytes(header) + bytes([3]) + pixels
        path = self.work / "rle.tga"
        path.write_bytes(payload)
        image, metadata = _decode_image(path, "rle")
        self.assertEqual(metadata["source_channels"], 3)
        self.assertEqual(image[0, 0].tolist(), [255, 0, 0])
        self.assertEqual(image[0, 1].tolist(), [0, 255, 0])
        self.assertEqual(image[1, 0].tolist(), [0, 0, 255])

    def test_uv_mirror_handedness_rebasing_round_trips(self) -> None:
        scene = load_scene(self.clean_scene)
        object_normal = np.array([0.31, 0.42, 0.85], dtype=np.float32)
        object_normal /= np.linalg.norm(object_normal)
        handedness: list[int] = []
        for flip_u, flip_v in ((False, False), (True, False), (False, True), (True, True)):
            scene.surface["islands"] = [
                {
                    "id": 0,
                    "bounds": [0, 0, 128, 96],
                    "flip_u": flip_u,
                    "flip_v": flip_v,
                }
            ]
            tangent = build_tangent_field(scene, (96, 128))
            handedness.append(int(tangent.handedness[40, 50]))
            tangent_space = np.array(
                [
                    np.dot(object_normal, tangent.tangent[40, 50]),
                    np.dot(object_normal, tangent.bitangent[40, 50]),
                    np.dot(object_normal, tangent.geometry_normal),
                ],
                dtype=np.float32,
            )
            tangent_normal_map = np.zeros((96, 128, 3), dtype=np.float32)
            tangent_normal_map[40, 50] = tangent_space
            decoded = tangent.object_normal_pixels(
                tangent_normal_map,
                np.array([50]),
                np.array([40]),
            )[0]
            np.testing.assert_allclose(decoded, object_normal, rtol=1e-5, atol=1e-6)
        self.assertEqual(handedness, [-1, 1, 1, -1])


if __name__ == "__main__":
    unittest.main()
