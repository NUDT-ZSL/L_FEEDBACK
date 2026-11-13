from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pe_audit.binary import ImportSymbol, call_import, jcc, ret
from pe_audit.callgraph import ParsedFunction
from pe_audit.errors import AuditError
from pe_audit.rules import evaluate_function, load_defects, validate_defects
from tests.support import make_analysis, write_json


STRCPY = ImportSymbol("msvcrt.dll", "strcpy")
STRNCPY = ImportSymbol("msvcrt.dll", "strncpy")


def _document() -> dict:
    return {
        "rules": [
            {
                "id": "D1",
                "target": "f",
                "dangerous_imports": [
                    {"module": "msvcrt.dll", "name": "strcpy"}
                ],
                "rule": {"type": "forbid_call"},
            }
        ]
    }


def _parsed(body: bytes) -> ParsedFunction:
    analysis = make_analysis({"f": body}, imports=[STRCPY, STRNCPY])
    return analysis.functions["f"]


class RulesValidationTest(unittest.TestCase):
    def test_valid_document_passes(self) -> None:
        defects = validate_defects(_document(), known_functions={"f"})
        self.assertEqual(defects.rules[0].id, "D1")

    def test_missing_field_is_rule_invalid(self) -> None:
        broken = _document()
        broken["rules"][0].pop("target")
        with self.assertRaises(AuditError) as caught:
            validate_defects(broken)
        self.assertEqual(caught.exception.code, "RULE_INVALID")
        self.assertEqual(caught.exception.exit_code, 5)

    def test_duplicate_rule_id_rejected(self) -> None:
        document = _document()
        document["rules"].append(dict(document["rules"][0]))
        with self.assertRaises(AuditError) as caught:
            validate_defects(document)
        self.assertEqual(caught.exception.code, "RULE_INVALID")

    def test_guarded_call_requires_branch_number(self) -> None:
        document = _document()
        document["rules"][0]["rule"] = {"type": "guarded_call"}
        with self.assertRaises(AuditError) as caught:
            validate_defects(document)
        self.assertEqual(caught.exception.code, "RULE_INVALID")

    def test_missing_import_module_is_rule_invalid(self) -> None:
        document = _document()
        document["rules"][0]["dangerous_imports"] = [{"name": "strcpy"}]
        with self.assertRaises(AuditError) as caught:
            validate_defects(document)
        self.assertEqual(caught.exception.code, "RULE_INVALID")

    def test_unknown_rule_target_is_rule_invalid(self) -> None:
        with self.assertRaises(AuditError) as caught:
            validate_defects(_document(), known_functions={"other"})
        self.assertEqual(caught.exception.code, "RULE_INVALID")

    def test_load_defects_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "defects.json"
            write_json(path, _document())
            defects = load_defects(path, known_functions={"f"})
            self.assertEqual(len(defects.rules), 1)


class VerdictTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = validate_defects(_document()).rules[0]

    def test_forbid_call_vulnerable_then_clean(self) -> None:
        vulnerable = evaluate_function(
            self.rule, _parsed(call_import(0).encode() + ret().encode())
        )
        self.assertEqual(vulnerable["verdict"], "VULNERABLE")
        self.assertEqual(vulnerable["evidence"][0]["call_offset"], 0)
        clean = evaluate_function(
            self.rule, _parsed(call_import(1).encode() + ret().encode())
        )
        self.assertEqual(clean["verdict"], "CLEAN")
        self.assertEqual(clean["evidence"], [])

    def test_guarded_call_clean_only_with_matching_branch(self) -> None:
        document = _document()
        document["rules"][0]["rule"] = {
            "type": "guarded_call",
            "branch_number": 0,
        }
        rule = validate_defects(document).rules[0]
        clean = evaluate_function(
            rule,
            _parsed(jcc(0).encode() + call_import(0).encode() + ret().encode()),
        )
        self.assertEqual(clean["verdict"], "CLEAN")
        self.assertTrue(clean["evidence"][0]["guarded"])

    def test_wrong_branch_number_stays_vulnerable(self) -> None:
        document = _document()
        document["rules"][0]["rule"] = {
            "type": "guarded_call",
            "branch_number": 0,
        }
        rule = validate_defects(document).rules[0]
        state = evaluate_function(
            rule,
            _parsed(jcc(1).encode() + call_import(0).encode() + ret().encode()),
        )
        self.assertEqual(state["verdict"], "VULNERABLE")
        self.assertFalse(state["evidence"][0]["guarded"])

    def test_illegal_bytes_mark_evidence_incomplete(self) -> None:
        state = evaluate_function(
            self.rule,
            _parsed(
                b"\xff\xff\xff\xff" + call_import(1).encode() + ret().encode()
            ),
        )
        self.assertEqual(state["verdict"], "CLEAN")
        self.assertFalse(state["evidence_complete"])
        self.assertEqual(len(state["parse_issues"]), 1)

    def test_removed_dangerous_import_is_clean_with_evidence_complete(self) -> None:
        state = evaluate_function(
            self.rule, _parsed(call_import(1).encode() + ret().encode())
        )
        self.assertEqual(state["verdict"], "CLEAN")
        self.assertTrue(state["evidence_complete"])


if __name__ == "__main__":
    unittest.main()
