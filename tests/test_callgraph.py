from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pe_audit.binary import ImportSymbol, call_import, call_local, jcc, parse_image, ret
from pe_audit.callgraph import analyze_calls, parse_function
from pe_audit.manifests import load_json, parse_manifest
from tests.support import generate


IMPORTS = [
    ImportSymbol("msvcrt.dll", "strcpy"),
    ImportSymbol("msvcrt.dll", "strncpy"),
]


def _parse(body: bytes, name: str = "f", names: list[str] | None = None):
    local_names = names or [name]
    return parse_function(
        body=body,
        function=name,
        local_names=local_names,
        imports=IMPORTS,
        known_functions=set(local_names),
    )


class CallGraphTest(unittest.TestCase):
    def test_import_and_local_calls_resolve_by_name(self) -> None:
        parsed = _parse(
            call_import(0).encode() + call_local(1).encode() + ret().encode(),
            names=["f", "callee"],
        )
        self.assertEqual(parsed.calls[0].kind, "import")
        self.assertEqual(parsed.calls[0].import_symbol, IMPORTS[0])
        self.assertEqual(parsed.calls[1].kind, "local")
        self.assertEqual(parsed.calls[1].callee, "callee")
        self.assertEqual(parsed.calls[1].offset, 4)

    def test_branch_number_is_attached_to_guarded_call(self) -> None:
        parsed = _parse(
            jcc(0).encode() + call_import(0).encode() + ret().encode()
        )
        self.assertEqual(parsed.branch_count, 1)
        self.assertEqual(parsed.calls[0].branch_number, 0)
        self.assertEqual(parsed.calls[0].branch_offset, 0)

    def test_branch_resets_after_ret(self) -> None:
        parsed = _parse(
            jcc(0).encode()
            + ret().encode()
            + call_import(0).encode()
            + ret().encode()
        )
        self.assertIsNone(parsed.calls[0].branch_number)
        self.assertIsNone(parsed.calls[0].branch_offset)

    def test_dangling_name_index_becomes_broken_edge(self) -> None:
        parsed = _parse(
            call_local(999).encode() + ret().encode(),
            names=["f"],
        )
        self.assertEqual(parsed.calls, [])
        self.assertEqual(parsed.broken_calls[0]["reason"], "local_name_index_out_of_range")
        self.assertFalse(parsed.complete)

    def test_illegal_bytes_are_counted_but_calls_still_resolve(self) -> None:
        parsed = _parse(
            b"\xfe\xff\xff\xff" + call_import(1).encode() + ret().encode()
        )
        self.assertEqual(parsed.invalid_bytes, 4)
        self.assertEqual(parsed.calls[0].import_symbol, IMPORTS[1])
        self.assertFalse(parsed.complete)

    def test_generated_evidence_contains_offsets_and_branch_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = generate(directory, seed=6)
            image = parse_image(
                (Path(directory) / "patch_after.bin").read_bytes()
            )
            document = load_json(
                output / "after_manifest.json", "manifest"
            )
            manifest = parse_manifest(document, image=image, version="after")
            analysis = analyze_calls(image, manifest)
            guarded = analysis.functions["guarded_copy"]
            self.assertEqual(guarded.calls[0].branch_number, 0)
            self.assertGreaterEqual(guarded.calls[0].offset, 0)
            self.assertEqual(guarded.calls[0].import_symbol.name, "strncpy")


if __name__ == "__main__":
    unittest.main()
