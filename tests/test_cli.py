from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from immersive_audio.container import ContainerReader

from tests.fixtures import (
    audio_object,
    keyframe,
    run_cli,
    tone,
    write_samples,
    write_scene,
)


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="iapano-cli-")
        self.root = Path(self.temp.name)
        write_samples(self.root / "tone.wav", tone(8_000, 0.2), 8_000)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _simple_scene(self, name: str = "scene.json") -> Path:
        return write_scene(
            self.root / name,
            [
                audio_object(
                    "a",
                    [keyframe(0.0, 0.0), keyframe(0.2, 90.0)],
                    [{"path": "tone.wav", "start": 0.0, "loops": 1}],
                )
            ],
            sample_rate=8_000,
            frame_length=400,
            duration=0.2,
        )

    def test_pack_and_render_with_options_and_stats(self) -> None:
        container = self.root / "scene.iap"
        wav = self.root / "scene.wav"
        pack_stats = self.root / "pack-stats.json"
        render_stats = self.root / "render-stats.json"
        packed = run_cli(
            "pack",
            "--scene",
            str(self._simple_scene()),
            "--output",
            str(container),
            "--stats",
            str(pack_stats),
        )
        self.assertEqual(packed.returncode, 0, msg=packed.stderr)
        pack_data = json.loads(packed.stdout)
        self.assertEqual(pack_data, json.loads(pack_stats.read_text(encoding="utf-8")))
        self.assertEqual(pack_data["command"], "pack")
        self.assertGreater(pack_data["peak_memory_mb"], 0.0)
        self.assertLessEqual(pack_data["peak_memory_mb"], 512)
        self.assertTrue(pack_data["within_memory_limit"])
        self.assertIn("build_object_buses", pack_data["stages_seconds"])

        rendered = run_cli(
            "render",
            "--input",
            str(container),
            "--output",
            str(wav),
            "--stats",
            str(render_stats),
        )
        self.assertEqual(rendered.returncode, 0, msg=rendered.stderr)
        render_data = json.loads(rendered.stdout)
        self.assertEqual(render_data, json.loads(render_stats.read_text(encoding="utf-8")))
        self.assertEqual(render_data["counts"]["output_channels"], 8)
        self.assertGreater(render_data["peak_memory_mb"], 0.0)
        self.assertLessEqual(render_data["peak_memory_mb"], 512)
        self.assertTrue(render_data["within_memory_limit"])

    def test_invalid_scene_reports_stable_code(self) -> None:
        bad = self.root / "bad.json"
        bad.write_text("{", encoding="utf-8")
        result = run_cli("pack", str(bad), str(self.root / "out.iap"))
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.stderr.startswith("E1002:"), result.stderr)

    def test_missing_audio_reports_stable_code(self) -> None:
        scene_path = self._simple_scene()
        (self.root / "tone.wav").unlink()
        result = run_cli("pack", str(scene_path), str(self.root / "out.iap"))
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stderr.startswith("E2001:"), result.stderr)

    def test_corrupt_container_render_reports_stable_code(self) -> None:
        container = self.root / "scene.iap"
        self.assertEqual(
            run_cli("pack", str(self._simple_scene()), str(container)).returncode,
            0,
        )
        raw = bytearray(container.read_bytes())
        raw[0] ^= 0xFF
        container.write_bytes(raw)
        result = run_cli("render", str(container), str(self.root / "out.wav"))
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stderr.startswith("E3001:"), result.stderr)
        self.assertFalse((self.root / "out.wav").exists())

    def test_checksum_error_is_reported_before_wav_write(self) -> None:
        container = self.root / "scene.iap"
        self.assertEqual(
            run_cli("pack", str(self._simple_scene()), str(container)).returncode,
            0,
        )
        reader = ContainerReader(container)
        raw = bytearray(container.read_bytes())
        raw[reader.header.audio_offset] ^= 0x04
        container.write_bytes(raw)
        wav = self.root / "out.wav"
        result = run_cli("render", str(container), str(wav))
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stderr.startswith("E3005:"), result.stderr)
        self.assertFalse(wav.exists())

    def test_64_objects_and_100_clips_per_object(self) -> None:
        write_samples(
            self.root / "tiny.wav",
            tone(8_000, 0.002, frequency=1_000.0, amplitude=0.01),
            8_000,
        )
        objects = []
        for obj_index in range(64):
            clips = [
                {
                    "path": "tiny.wav",
                    "start": (clip_index % 10) * 0.002,
                    "loops": 1,
                }
                for clip_index in range(100)
            ]
            objects.append(
                audio_object(
                    f"o{obj_index}",
                    [keyframe(0.0, float(obj_index * 5.625))],
                    clips,
                )
            )
        scene_path = write_scene(
            self.root / "max.json",
            objects,
            sample_rate=8_000,
            frame_length=64,
            duration=0.02,
        )
        container = self.root / "max.iap"
        wav = self.root / "max.wav"
        packed = run_cli("pack", str(scene_path), str(container))
        self.assertEqual(packed.returncode, 0, msg=packed.stderr)
        stats = json.loads(packed.stdout)
        self.assertEqual(stats["counts"]["objects"], 64)
        self.assertEqual(stats["counts"]["clips"], 6_400)
        self.assertLessEqual(stats["peak_memory_mb"], 512)
        reader = ContainerReader(container)
        self.assertEqual(reader.header.channel_count, 64)
        self.assertEqual(reader.header.frame_count, (160 + 63) // 64)
        rendered = run_cli("render", str(container), str(wav))
        self.assertEqual(rendered.returncode, 0, msg=rendered.stderr)
        self.assertLessEqual(json.loads(rendered.stdout)["peak_memory_mb"], 512)
