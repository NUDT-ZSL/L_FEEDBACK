from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.support import generate, read_json, run_cli, write_json


def _audit_args(directory: Path, report: Path) -> tuple[str, ...]:
    return (
        "--before",
        str(directory / "patch_before.bin"),
        "--after",
        str(directory / "patch_after.bin"),
        "--before-manifest",
        str(directory / "before_manifest.json"),
        "--after-manifest",
        str(directory / "after_manifest.json"),
        "--defects",
        str(directory / "defects.json"),
        "--entries",
        str(directory / "entries.json"),
        "--output",
        str(report),
    )


class EndToEndTest(unittest.TestCase):
    def test_full_report_and_three_state_conclusions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = generate(directory, seed=0)
            report_path = output / "report.json"
            result = run_cli(*_audit_args(output, report_path))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            report = read_json(report_path)
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(len(report["targets"]), 3)
            by_id = {item["id"]: item for item in report["targets"]}
            self.assertEqual(by_id["D1"]["conclusion"], "已封堵")
            self.assertEqual(by_id["D1"]["basis"], "patched_clean")
            self.assertEqual(by_id["D2"]["conclusion"], "证据不足")
            self.assertEqual(by_id["D2"]["basis"], "vulnerable_unreachable")
            self.assertEqual(by_id["D3"]["conclusion"], "仍可达")
            self.assertEqual(by_id["D3"]["basis"], "vulnerable_and_reachable")
            summary = report["summary"]
            self.assertEqual(summary["target_count"], 3)
            self.assertEqual(summary["sealed"], 1)
            self.assertEqual(summary["still_reachable"], 1)
            self.assertEqual(summary["insufficient_evidence"], 1)

    def test_still_reachable_includes_path_offsets_and_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = generate(directory, seed=0)
            report_path = output / "report.json"
            result = run_cli(*_audit_args(output, report_path))
            self.assertEqual(result.returncode, 0, result.stderr)
            report = read_json(report_path)
            target = next(item for item in report["targets"] if item["id"] == "D3")
            self.assertEqual(
                target["path"], ["entry_main", "dispatch", "guarded_copy"]
            )
            self.assertTrue(target["path_edges"])
            self.assertTrue(all(edge["offset"] >= 0 for edge in target["path_edges"]))
            evidence = target["call_evidence"]
            self.assertTrue(evidence)
            self.assertIsInstance(evidence[0]["call_offset"], int)
            self.assertIsInstance(evidence[0]["branch_number"], int)

    def test_report_is_deterministic_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = generate(directory, seed=0)
            first_path = output / "first.json"
            second_path = output / "second.json"
            first = run_cli(*_audit_args(output, first_path))
            second = run_cli(*_audit_args(output, second_path))
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(first.stdout, second.stdout)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())

    def test_manifest_order_independence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = generate(directory, seed=0)
            first_path = output / "report_a.json"
            second_path = output / "report_b.json"
            first = run_cli(*_audit_args(output, first_path))
            self.assertEqual(first.returncode, 0, first.stderr)

            shuffled_dir = output / "shuffled"
            shuffled_dir.mkdir()
            for name in ("before_manifest.json", "after_manifest.json"):
                document = read_json(output / name)
                document["functions"].reverse()
                write_json(shuffled_dir / name, document)
            second = run_cli(
                "--before",
                str(output / "patch_before.bin"),
                "--after",
                str(output / "patch_after.bin"),
                "--before-manifest",
                str(shuffled_dir / "before_manifest.json"),
                "--after-manifest",
                str(shuffled_dir / "after_manifest.json"),
                "--defects",
                str(output / "defects.json"),
                "--entries",
                str(output / "entries.json"),
                "--output",
                str(second_path),
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(first.stdout, second.stdout)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())

    def test_deleted_target_is_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = generate(directory, seed=0)
            after_doc = read_json(output / "after_manifest.json")
            after_doc["functions"] = [
                item
                for item in after_doc["functions"]
                if item["name"] != "unsafe_copy"
            ]
            deleted_manifest = output / "after_deleted.json"
            write_json(deleted_manifest, after_doc)
            report_path = output / "deleted_report.json"
            result = run_cli(
                "--before",
                str(output / "patch_before.bin"),
                "--after",
                str(output / "patch_after.bin"),
                "--before-manifest",
                str(output / "before_manifest.json"),
                "--after-manifest",
                str(deleted_manifest),
                "--defects",
                str(output / "defects.json"),
                "--entries",
                str(output / "entries.json"),
                "--output",
                str(report_path),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = read_json(report_path)
            target = next(item for item in report["targets"] if item["id"] == "D1")
            self.assertEqual(target["after"]["deleted"], True)
            self.assertFalse(target["after"]["present"])
            self.assertEqual(target["conclusion"], "已封堵")
            self.assertEqual(target["basis"], "target_deleted")
            self.assertIn("unsafe_copy", report["functions"]["alignment"]["removed"])

    def test_rule_invalid_does_not_write_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = generate(directory, seed=0)
            bad_rules = output / "bad_rules.json"
            write_json(bad_rules, {"rules": [{"id": "broken"}]})
            report_path = output / "must_not_exist.json"
            result = run_cli(
                "--before",
                str(output / "patch_before.bin"),
                "--after",
                str(output / "patch_after.bin"),
                "--before-manifest",
                str(output / "before_manifest.json"),
                "--after-manifest",
                str(output / "after_manifest.json"),
                "--defects",
                str(bad_rules),
                "--entries",
                str(output / "entries.json"),
                "--output",
                str(report_path),
            )
            self.assertEqual(result.returncode, 5)
            self.assertFalse(report_path.exists())
            error = json.loads(result.stderr)
            self.assertEqual(error["error"]["code"], "RULE_INVALID")

    def test_unknown_rule_target_is_rule_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = generate(directory, seed=0)
            bad_rules = output / "bad_target.json"
            document = read_json(output / "defects.json")
            document["rules"][0]["target"] = "no_such_function"
            write_json(bad_rules, document)
            report_path = output / "must_not_exist.json"
            result = run_cli(
                "--before",
                str(output / "patch_before.bin"),
                "--after",
                str(output / "patch_after.bin"),
                "--before-manifest",
                str(output / "before_manifest.json"),
                "--after-manifest",
                str(output / "after_manifest.json"),
                "--defects",
                str(bad_rules),
                "--entries",
                str(output / "entries.json"),
                "--output",
                str(report_path),
            )
            self.assertEqual(result.returncode, 5)
            self.assertFalse(report_path.exists())

    def test_report_has_no_paths_or_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = generate(directory, seed=0)
            report_path = output / "report.json"
            result = run_cli(*_audit_args(output, report_path))
            self.assertEqual(result.returncode, 0, result.stderr)
            raw = report_path.read_text(encoding="utf-8")
            self.assertNotIn(str(output), raw)
            self.assertNotIn(str(Path(directory)), raw)
            self.assertNotIn("timestamp", raw.lower())
            self.assertNotIn("generated_at", raw.lower())

    def test_stdout_is_valid_summary_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = generate(directory, seed=0)
            report_path = output / "report.json"
            result = run_cli(*_audit_args(output, report_path))
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            self.assertEqual(summary["target_count"], 3)


if __name__ == "__main__":
    unittest.main()
