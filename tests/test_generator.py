from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from pe_audit.binary import parse_image
from pe_audit.callgraph import analyze_calls
from pe_audit.manifests import load_json, parse_manifest
from tests.support import generate, read_json, run_generator


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GeneratorTest(unittest.TestCase):
    def test_same_seed_is_byte_for_byte_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            generate(first, seed=42)
            generate(second, seed=42)
            for name in (
                "patch_before.bin",
                "patch_after.bin",
                "before_manifest.json",
                "after_manifest.json",
                "entries.json",
                "defects.json",
            ):
                self.assertEqual(
                    (Path(first) / name).read_bytes(),
                    (Path(second) / name).read_bytes(),
                    name,
                )

    def test_different_seeds_change_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            generate(first, seed=1)
            generate(second, seed=2)
            for name in ("patch_before.bin", "patch_after.bin"):
                self.assertNotEqual(
                    _hash(Path(first) / name),
                    _hash(Path(second) / name),
                )

    def test_identities_match_while_layout_differs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = generate(directory, seed=7)
            before = read_json(output / "before_manifest.json")
            after = read_json(output / "after_manifest.json")
            before_names = {item["name"] for item in before["functions"]}
            after_names = {item["name"] for item in after["functions"]}
            self.assertEqual(before_names, after_names)
            before_order = [
                item["name"]
                for item in sorted(before["functions"], key=lambda value: value["rva"])
            ]
            after_order = [
                item["name"]
                for item in sorted(after["functions"], key=lambda value: value["rva"])
            ]
            self.assertNotEqual(before_order, after_order)
            before_rvas = {item["name"]: item["rva"] for item in before["functions"]}
            after_rvas = {item["name"]: item["rva"] for item in after["functions"]}
            self.assertNotEqual(before_rvas, after_rvas)

    def test_manifest_intervals_are_in_bounds_and_nonoverlapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = generate(directory, seed=13)
            for version in ("before", "after"):
                data = (output / f"patch_{version}.bin").read_bytes()
                image = parse_image(data)
                document = load_json(
                    output / f"{version}_manifest.json", "manifest"
                )
                manifest = parse_manifest(
                    document, image=image, version=version
                )
                intervals = sorted(
                    (region.rva, region.end_rva)
                    for region in manifest.regions.values()
                )
                for index, (start, end) in enumerate(intervals):
                    self.assertLessEqual(end, 0x1000 + len(image.data[0x200:]))
                    if index:
                        self.assertLessEqual(intervals[index - 1][1], start)

    def test_generated_calls_resolve_only_by_stable_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = generate(directory, seed=9)
            for version in ("before", "after"):
                image = parse_image(
                    (output / f"patch_{version}.bin").read_bytes()
                )
                document = load_json(
                    output / f"{version}_manifest.json", "manifest"
                )
                manifest = parse_manifest(
                    document, image=image, version=version
                )
                analysis = analyze_calls(image, manifest)
                self.assertFalse(
                    any(
                        function.broken_calls
                        for function in analysis.functions.values()
                    )
                )
                calls = [
                    call
                    for function in analysis.functions.values()
                    for call in function.calls
                    if call.kind == "local"
                ]
                self.assertTrue(calls)
                self.assertTrue(all(call.callee in manifest.regions for call in calls))

    def test_cli_writes_both_binaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_generator("--seed", "3", "--out", directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(directory) / "patch_before.bin").is_file())
            self.assertTrue((Path(directory) / "patch_after.bin").is_file())


if __name__ == "__main__":
    unittest.main()
