from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from pe_audit.binary import parse_image
from pe_audit.manifests import (
    align_manifests,
    load_json,
    parse_manifest,
)
from pe_audit.errors import AuditError
from tests.support import generate, read_json, run_cli, write_json


class AlignmentValidationTest(unittest.TestCase):
    def _documents(self, directory: str) -> tuple[dict, dict, object, object]:
        output = generate(directory, seed=4)
        before_data = (Path(directory) / "patch_before.bin").read_bytes()
        after_data = (Path(directory) / "patch_after.bin").read_bytes()
        before_image = parse_image(before_data)
        after_image = parse_image(after_data)
        before_doc = read_json(output / "before_manifest.json")
        after_doc = read_json(output / "after_manifest.json")
        return before_doc, after_doc, before_image, after_image

    def test_valid_manifest_aligns_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            before_doc, after_doc, before_image, after_image = self._documents(
                directory
            )
            before = parse_manifest(
                before_doc, image=before_image, version="before"
            )
            after = parse_manifest(
                after_doc, image=after_image, version="after"
            )
            alignment = align_manifests(before, after)
            self.assertEqual(len(alignment.common), 8)
            self.assertEqual(alignment.removed, ())
            self.assertEqual(alignment.added, ())
            for name in alignment.common:
                self.assertIn(name, before.regions)
                self.assertIn(name, after.regions)

    def test_duplicate_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            before_doc, _after_doc, before_image, _after_image = self._documents(
                directory
            )
            duplicate = copy.deepcopy(before_doc)
            duplicate["functions"].append(dict(duplicate["functions"][0]))
            with self.assertRaises(AuditError) as caught:
                parse_manifest(duplicate, image=before_image, version="before")
            self.assertEqual(caught.exception.code, "MANIFEST_INVALID")

    def test_missing_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            before_doc, _after_doc, before_image, _after_image = self._documents(
                directory
            )
            broken = copy.deepcopy(before_doc)
            broken["functions"][0].pop("name")
            with self.assertRaises(AuditError) as caught:
                parse_manifest(broken, image=before_image, version="before")
            self.assertEqual(caught.exception.code, "MANIFEST_INVALID")

    def test_overlapping_intervals_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            before_doc, _after_doc, before_image, _after_image = self._documents(
                directory
            )
            overlap = copy.deepcopy(before_doc)
            overlap["functions"].sort(key=lambda item: item["rva"])
            overlap["functions"][1]["rva"] = overlap["functions"][0]["rva"]
            with self.assertRaises(AuditError) as caught:
                parse_manifest(overlap, image=before_image, version="before")
            self.assertEqual(caught.exception.code, "MANIFEST_INVALID")

    def test_out_of_bounds_interval_uses_bounds_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            before_doc, _after_doc, before_image, _after_image = self._documents(
                directory
            )
            bad = copy.deepcopy(before_doc)
            bad["functions"][0]["length"] = 10021
            with self.assertRaises(AuditError) as caught:
                parse_manifest(bad, image=before_image, version="before")
            self.assertEqual(caught.exception.code, "BOUNDS_OVERFLOW")
            self.assertEqual(caught.exception.exit_code, 4)

    def test_bounds_overflow_cli_emits_stderr_json_and_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = generate(directory, seed=4)
            bad = read_json(output / "before_manifest.json")
            bad["functions"][0]["length"] = 10021
            bad_path = Path(directory) / "bad_manifest.json"
            write_json(bad_path, bad)
            report_path = Path(directory) / "must_not_exist.json"
            result = run_cli(
                "--before",
                str(output / "patch_before.bin"),
                "--after",
                str(output / "patch_after.bin"),
                "--before-manifest",
                str(bad_path),
                "--after-manifest",
                str(output / "after_manifest.json"),
                "--defects",
                str(output / "defects.json"),
                "--entries",
                str(output / "entries.json"),
                "--output",
                str(report_path),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.returncode, 4)
            self.assertFalse(report_path.exists())
            self.assertIn("BOUNDS_OVERFLOW", result.stderr)
            import json

            parsed = json.loads(result.stderr)
            self.assertEqual(parsed["error"]["code"], "BOUNDS_OVERFLOW")

    def test_deleted_function_is_a_valid_alignment_difference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            before_doc, after_doc, before_image, after_image = self._documents(
                directory
            )
            before = parse_manifest(
                before_doc, image=before_image, version="before"
            )
            after_doc = copy.deepcopy(after_doc)
            after_doc["functions"] = [
                item
                for item in after_doc["functions"]
                if item["name"] != "orphan_copy"
            ]
            after = parse_manifest(
                after_doc, image=after_image, version="after"
            )
            alignment = align_manifests(before, after)
            self.assertIn("orphan_copy", alignment.removed)
            self.assertNotIn("orphan_copy", alignment.common)


if __name__ == "__main__":
    unittest.main()
