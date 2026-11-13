from __future__ import annotations

import unittest

from pe_audit.binary import OP_CALL_LOCAL, call_import, cmp_imm, jcc, pad, push_imm, ret
from pe_audit.diffing import diff_aligned_functions, diff_function
from pe_audit.manifests import Alignment
from tests.support import make_analysis


class DiffingTest(unittest.TestCase):
    def test_padding_only_change_is_unchanged(self) -> None:
        before = make_analysis({"f": call_import(0).encode() + ret().encode()})
        after = make_analysis(
            {"f": pad().encode() * 3 + call_import(0).encode() + ret().encode()}
        )
        result = diff_function(before=before, after=after, function="f")
        self.assertEqual(result.status, "unchanged")
        self.assertEqual(result.counts, {"added": 0, "removed": 0, "modified": 0})

    def test_relocation_does_not_create_false_positive(self) -> None:
        body = call_import(0).encode() + ret().encode()
        before = make_analysis({"f": body})
        after = make_analysis({"f": body})
        result = diff_function(before=before, after=after, function="f")
        self.assertEqual(result.status, "unchanged")

    def test_replaced_import_call_is_reported_as_modified(self) -> None:
        before = make_analysis({"f": call_import(0).encode() + ret().encode()})
        after = make_analysis({"f": call_import(1).encode() + ret().encode()})
        result = diff_function(before=before, after=after, function="f")
        self.assertEqual(result.status, "modified")
        self.assertEqual(len(result.modified), 1)
        self.assertIn("strcpy", result.modified[0]["before"]["signature"])
        self.assertIn("strncpy", result.modified[0]["after"]["signature"])

    def test_added_guard_instructions_are_reported(self) -> None:
        before = make_analysis(
            {"f": push_imm(8).encode() + call_import(0).encode() + ret().encode()}
        )
        after = make_analysis(
            {
                "f": push_imm(8).encode()
                + cmp_imm(4).encode()
                + jcc(0).encode()
                + call_import(0).encode()
                + ret().encode()
            }
        )
        result = diff_function(before=before, after=after, function="f")
        self.assertEqual(result.status, "modified")
        self.assertEqual(len(result.added), 2)

    def test_function_order_does_not_affect_aligned_diff(self) -> None:
        before = make_analysis(
            {
                "a": call_import(0).encode() + ret().encode(),
                "b": ret().encode(),
            }
        )
        after = make_analysis(
            {
                "b": pad().encode() + ret().encode(),
                "a": call_import(0).encode() + ret().encode(),
            }
        )
        alignment = Alignment(
            before=before.manifest,
            after=after.manifest,
            common=("a", "b"),
            removed=(),
            added=(),
        )
        report = diff_aligned_functions(
            before=before, after=after, alignment=alignment
        )
        self.assertEqual(report["unchanged_functions"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
