from __future__ import annotations

import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from immersive_audio.container import ContainerReader, HEADER_SIZE
from immersive_audio.errors import PipelineError

from tests.fixtures import (
    audio_object,
    keyframe,
    run_cli,
    sha256_file,
    tone,
    write_samples,
    write_scene,
)


class ContainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="iapano-container-")
        self.root = Path(self.temp.name)
        write_samples(self.root / "a.wav", tone(48_000, 0.25, 440.0))
        write_samples(self.root / "b.wav", tone(48_000, 0.35, 880.0))
        scene_path = write_scene(
            self.root / "scene.json",
            [
                audio_object(
                    "front",
                    [keyframe(0.0, 0.0), keyframe(0.5, 45.0)],
                    [{"path": "a.wav", "start": 0.0, "loops": 2}],
                ),
                audio_object(
                    "back",
                    [keyframe(0.0, 180.0), keyframe(0.5, 270.0)],
                    [{"path": "b.wav", "start": 0.1, "loops": 1}],
                ),
            ],
            sample_rate=48_000,
            frame_length=1_024,
            duration=0.5,
        )
        result = run_cli(
            "pack", str(scene_path), str(self.root / "scene.iap"), "--stats", str(self.root / "stats.json")
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.container = self.root / "scene.iap"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_header_alignment_and_readback(self) -> None:
        reader = ContainerReader(self.container)
        header = reader.header
        self.assertEqual(header.magic, b"IAPANOR1")
        self.assertEqual(header.version, 1)
        self.assertEqual(header.header_size, HEADER_SIZE)
        self.assertEqual(header.sample_rate, 48_000)
        self.assertEqual(header.frame_length, 1_024)
        self.assertEqual(header.channel_count, 2)
        self.assertEqual(header.object_count, 2)
        self.assertEqual(header.sample_count, 24_000)
        self.assertEqual(header.frame_count, (24_000 + 1_023) // 1_024)
        self.assertEqual(
            header.metadata_size,
            header.frame_count * header.object_count * 20,
        )
        metadata, audio = reader.read_all()
        self.assertEqual(metadata.shape, (header.frame_count, 2, 5))
        self.assertEqual(audio.shape, (24_000, 2))
        self.assertEqual(reader.verify_payload(), header.payload_sha256.hex())

    def test_comments_contain_all_clip_hashes(self) -> None:
        reader = ContainerReader(self.container)
        objects = reader.comments["objects"]
        self.assertEqual(len(objects), 2)
        self.assertEqual(
            objects[0]["clips"][0]["sha256"], sha256_file(self.root / "a.wav")
        )
        self.assertEqual(
            objects[1]["clips"][0]["sha256"], sha256_file(self.root / "b.wav")
        )

    def test_pack_is_byte_deterministic(self) -> None:
        second = self.root / "second.iap"
        result = run_cli(
            "pack",
            str(self.root / "scene.json"),
            str(second),
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(self.container.read_bytes(), second.read_bytes())

    def test_bad_magic(self) -> None:
        damaged = self.root / "magic.iap"
        shutil.copyfile(self.container, damaged)
        raw = bytearray(damaged.read_bytes())
        raw[0] ^= 0xFF
        damaged.write_bytes(raw)
        with self.assertRaises(PipelineError) as raised:
            ContainerReader(damaged)
        self.assertEqual(raised.exception.code, "E3001")

    def test_bad_version(self) -> None:
        damaged = self.root / "version.iap"
        shutil.copyfile(self.container, damaged)
        raw = bytearray(damaged.read_bytes())
        struct.pack_into("<H", raw, 8, 99)
        damaged.write_bytes(raw)
        with self.assertRaises(PipelineError) as raised:
            ContainerReader(damaged)
        self.assertEqual(raised.exception.code, "E3002")

    def test_truncated_payload(self) -> None:
        damaged = self.root / "truncated.iap"
        raw = self.container.read_bytes()[:-17]
        damaged.write_bytes(raw)
        with self.assertRaises(PipelineError) as raised:
            ContainerReader(damaged)
        self.assertEqual(raised.exception.code, "E3003")

    def test_checksum_detects_audio_corruption(self) -> None:
        damaged = self.root / "checksum.iap"
        shutil.copyfile(self.container, damaged)
        reader = ContainerReader(damaged)
        raw = bytearray(damaged.read_bytes())
        raw[reader.header.audio_offset + 5] ^= 0x01
        damaged.write_bytes(raw)
        damaged_reader = ContainerReader(damaged)
        with self.assertRaises(PipelineError) as raised:
            damaged_reader.verify_payload()
        self.assertEqual(raised.exception.code, "E3005")

    def test_invalid_comment_json(self) -> None:
        damaged = self.root / "comment.iap"
        shutil.copyfile(self.container, damaged)
        reader = ContainerReader(damaged)
        raw = bytearray(damaged.read_bytes())
        raw[reader.header.comment_offset] = ord("[")
        damaged.write_bytes(raw)
        with self.assertRaises(PipelineError) as raised:
            ContainerReader(damaged)
        self.assertEqual(raised.exception.code, "E3006")

    def test_metadata_matches_timeline(self) -> None:
        reader = ContainerReader(self.container)
        metadata = reader.read_metadata()
        self.assertAlmostEqual(float(metadata[0, 0, 0]), 0.0)
        last_block_start = (reader.header.frame_count - 1) * reader.header.frame_length
        expected_last_azimuth = 45.0 * last_block_start / reader.header.sample_count
        self.assertAlmostEqual(
            float(metadata[-1, 0, 0]), expected_last_azimuth, delta=1e-4
        )
        self.assertAlmostEqual(float(metadata[0, 1, 0]), 180.0)
        self.assertAlmostEqual(
            float(metadata[-1, 1, 0]),
            180.0 + 90.0 * last_block_start / reader.header.sample_count,
            delta=1e-4,
        )
        np.testing.assert_allclose(metadata[:, 0, 2], 1.0)
        np.testing.assert_allclose(metadata[:, 1, 3], 1.0)
