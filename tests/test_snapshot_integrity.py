from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from ruin_bake.bake import BakeParams, bake_full
from ruin_bake.snapshot import (
    FOOTER_SIZE,
    MAGIC,
    PREFIX,
    SnapshotError,
    load_snapshot,
    write_snapshot_atomic,
)

from .helpers import TEST_TMP, make_layout, make_scene


class SnapshotIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory(dir=TEST_TMP)
        cls.directory = Path(cls._temp.name)
        cls.scene = make_scene()
        cls.layout = make_layout(cls.scene)
        cls.output = bake_full(cls.scene, cls.layout, BakeParams(samples=8, workers=1, chunk_size=11))
        cls.path = cls.directory / "valid.snapshot"
        cls.snapshot = write_snapshot_atomic(
            cls.path,
            scene=cls.scene,
            layout=cls.layout,
            irradiances=cls.output.irradiances,
            evaluation=cls.output.evaluation,
            revision=3,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def test_round_trip(self) -> None:
        loaded = load_snapshot(self.path)
        self.assertEqual(loaded.scene_hash, self.scene.scene_hash)
        self.assertEqual(loaded.layout_hash, self.layout.layout_hash)
        self.assertEqual(tuple(record.irradiance for record in loaded.probes), self.output.irradiances)
        self.assertEqual(loaded.revision, 3)
        self.assertEqual(loaded.probe_count, len(self.layout.probes))

    def test_single_flipped_byte_is_rejected(self) -> None:
        data = bytearray(self.path.read_bytes())
        data[PREFIX.size + 20] ^= 0x40
        bad_path = self.directory / "flipped.snapshot"
        bad_path.write_bytes(data)
        with self.assertRaises(SnapshotError):
            load_snapshot(bad_path)

    def test_truncated_snapshot_is_rejected(self) -> None:
        data = self.path.read_bytes()[:-7]
        bad_path = self.directory / "truncated.snapshot"
        bad_path.write_bytes(data)
        with self.assertRaises(SnapshotError):
            load_snapshot(bad_path)

    def test_bad_version_is_rejected_before_reuse(self) -> None:
        data = bytearray(self.path.read_bytes())
        struct.pack_into("<I", data, 8, 99)
        bad_path = self.directory / "version.snapshot"
        bad_path.write_bytes(data)
        with self.assertRaises(SnapshotError):
            load_snapshot(bad_path)

    def test_tampered_offsets_with_valid_checksums_are_rejected(self) -> None:
        original = self.path.read_bytes()
        magic, version, header_len, payload_len = PREFIX.unpack_from(original, 0)
        body = original[: PREFIX.size + header_len + payload_len]
        header_start = PREFIX.size
        header_end = header_start + header_len
        payload = original[header_end : header_end + payload_len]
        header = json.loads(original[header_start:header_end].decode("utf-8"))
        header["probe_directory"]["records_offset"] = 8
        header_bytes = json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        new_prefix = PREFIX.pack(magic, version, len(header_bytes), payload_len)
        new_body = new_prefix + header_bytes + payload
        tampered = new_body + hashlib.sha256(new_body).digest()
        bad_path = self.directory / "offset.snapshot"
        bad_path.write_bytes(tampered)
        with self.assertRaisesRegex(SnapshotError, "probe directory"):
            load_snapshot(bad_path)


if __name__ == "__main__":
    unittest.main()

