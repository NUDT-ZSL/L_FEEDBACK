#!/usr/bin/env python3
"""End-to-end acceptance harness for audit_workbench.

Run from the project root:
    py -3.11 tests/acceptance.py --work-dir artifacts/acceptance-work

The harness only uses generated local fixtures.  It records each check in
artifacts/acceptance/results.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_workbench.config import load_config_file, load_config_from_json  # noqa: E402
from audit_workbench.errors import EXIT_CODES  # noqa: E402


class Acceptance:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.checks: list[dict[str, Any]] = []
        self.stdout: list[str] = []

    def log(self, message: str) -> None:
        line = f"[acceptance] {message}"
        print(line, flush=True)
        self.stdout.append(line)

    def check(self, name: str, condition: bool, detail: Any = None) -> None:
        if not condition:
            raise AssertionError(f"{name}: {detail!r}")
        self.checks.append({"name": name, "passed": True, "detail": detail})
        self.log(f"PASS {name}")

    def run(
        self,
        args: list[str],
        *,
        expected_exit: int | None = None,
        timeout: float = 60,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [sys.executable, "-m", "audit_workbench", *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        if expected_exit is not None and completed.returncode != expected_exit:
            raise AssertionError(
                f"command exited {completed.returncode}, expected {expected_exit}: "
                f"{args}\nstdout={completed.stdout[:2000]}\nstderr={completed.stderr[:2000]}"
            )
        return completed

    def load(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    def loads(self, text: str) -> Any:
        return json.loads(text)


def tree_digest(root: Path) -> tuple[list[str], dict[str, str]]:
    files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    )
    hashes = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in files
    }
    return files, hashes


def fixture_paths(manifest: dict[str, Any], name: str) -> tuple[Path, Path]:
    info = manifest["fixtures"][name]
    return Path(info["root"]), Path(info["config"])


def call_map(report: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (call["root"], call["config"], call["id"]): call
        for call in report["calls"]
    }


def parse_text_calls(text: str) -> list[dict[str, str]]:
    result = []
    for line in text.splitlines():
        if not line.startswith("CALL|"):
            continue
        values: dict[str, str] = {}
        for field in line.split("|")[1:]:
            key, value = field.split("=", 1)
            values[key] = value
        result.append(values)
    return result


def assert_error_contract(error: dict[str, Any]) -> None:
    assert set(error) == {"code", "message", "context"}, error
    assert error["message"], error
    assert isinstance(error["context"], dict), error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-dir",
        default=str(ROOT / "artifacts" / "acceptance-work"),
    )
    parser.add_argument(
        "--results",
        default=str(ROOT / "artifacts" / "acceptance" / "results.json"),
    )
    args = parser.parse_args()
    started = time.time()
    work = Path(args.work_dir).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results).resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)

    a = Acceptance(work)
    samples = work / "samples"
    samples_dup = work / "samples-dup"
    a.run(["gen", "--seed", "20240501", "--out", str(samples)], expected_exit=0)
    a.run(["gen", "--seed", "20240501", "--out", str(samples_dup)], expected_exit=0)
    manifest = a.load(samples / "roots.json")

    # 1. Deterministic generation, layout, and conditional visibility.
    files_a, hashes_a = tree_digest(samples)
    files_b, hashes_b = tree_digest(samples_dup)
    a.check("gen.deterministic_file_list", files_a == files_b, len(files_a))
    a.check("gen.deterministic_hashes", hashes_a == hashes_b, len(hashes_a))
    a.check("gen.three_roots", len(manifest["roots"]) >= 3, len(manifest["roots"]))
    for root in manifest["roots"]:
        root_path = samples / root["path"]
        source_files = [
            path
            for path in root_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".c", ".h"}
        ]
        max_depth = max(
            len(path.relative_to(root_path).parts) - 1 for path in source_files
        )
        a.check(
            f"gen.layout.{root['name']}",
            len(source_files) >= 6 and max_depth >= 3,
            {"files": len(source_files), "max_depth": max_depth},
        )

    state_main = work / "state-main"
    report_ab_path = work / "report-ab.json"
    text_ab_path = work / "report-ab.txt"
    a.run(
        [
            "scan",
            "--root",
            str(samples),
            "--configs",
            "cfgA,cfgB",
            "--jobs",
            "2",
            "--state-dir",
            str(state_main),
            "--json",
            str(report_ab_path),
            "--text",
            str(text_ab_path),
        ],
        expected_exit=0,
    )
    report_ab = a.load(report_ab_path)
    calls_ab = call_map(report_ab)
    a.check("scan.ab.clean", report_ab["code"] == "OK", report_ab["code"])
    a.check(
        "scan.ab.macro_expansions",
        report_ab["summary"]["macro_expansions"] > 0,
        report_ab["summary"]["macro_expansions"],
    )

    for root in manifest["roots"]:
        expected = manifest["expected"][root["name"]]
        cond = expected["conditional_call"]
        b_only = expected["b_only_call"]
        a_call = calls_ab.get((root["name"], "cfgA", cond["id"]))
        b_call = calls_ab.get((root["name"], "cfgB", cond["id"]))
        b_only_call = calls_ab.get((root["name"], "cfgB", b_only["id"]))
        a_only_absent = calls_ab.get((root["name"], "cfgA", b_only["id"]))
        a.check(
            f"config.visibility_a.{root['name']}",
            a_call is not None and b_call is None,
            {"a": a_call is not None, "b": b_call is not None},
        )
        a.check(
            f"config.visibility_b.{root['name']}",
            b_only_call is not None and a_only_absent is None,
            {"b": b_only_call is not None, "a": a_only_absent is not None},
        )
        assert a_call is not None
        a.check(
            f"config.location.{root['name']}",
            a_call["location"]["file"] == cond["file"]
            and
            a_call["location"]["line"] == cond["line"]
            and a_call["location"]["column"] == cond["column"],
            {"actual": a_call["location"], "expected": cond},
        )
        cfg_a_calls = [
            call
            for call in report_ab["calls"]
            if call["root"] == root["name"] and call["config"] == "cfgA"
        ]
        a.check(
            f"calls.minimum.{root['name']}",
            len(cfg_a_calls) >= 6,
            len(cfg_a_calls),
        )

    # 2. Config round-trip and equivalence.
    cfg_a = load_config_file(samples / "configs" / "cfgA.json")
    cfg_a_round = load_config_from_json(cfg_a.serialized())
    a.check("config.roundtrip_equivalent", cfg_a.equivalent(cfg_a_round), None)
    a.check(
        "config.roundtrip_hash",
        cfg_a.content_hash() == cfg_a_round.content_hash(),
        cfg_a.content_hash(),
    )
    compare = a.run(
        [
            "config",
            "compare",
            "--left",
            str(samples / "configs" / "cfgA.json"),
            "--right",
            str(samples / "configs" / "cfgA.json"),
        ],
        expected_exit=0,
    )
    a.check(
        "config.cli_compare",
        a.loads(compare.stdout)["equivalent"] is True,
        compare.stdout.strip(),
    )

    # 3. Reachability, taint, and high-risk algebra.
    for root in manifest["roots"]:
        expected = manifest["expected"][root["name"]]
        orphan = calls_ab[(root["name"], "cfgA", expected["unreachable"]["id"])]
        fp = calls_ab[(root["name"], "cfgA", expected["fp_reachable"]["id"])]
        unknown = calls_ab[(root["name"], "cfgA", expected["indirect_unknown"]["id"])]
        literal = calls_ab[(root["name"], "cfgA", expected["literal_untainted"]["id"])]
        param = calls_ab[(root["name"], "cfgA", expected["param_tainted"]["id"])]
        unresolved = calls_ab[(root["name"], "cfgA", expected["unresolved_taint"]["id"])]
        a.check(f"reachability.unreachable.{root['name']}", orphan["reachability"] == "unreachable", orphan["reachability"])
        a.check(f"reachability.fp_table.{root['name']}", fp["reachability"] == "reachable", fp["path"])
        a.check(
            f"reachability.external_unknown.{root['name']}",
            unknown["reachability"] == "unknown" and bool(unknown["reachability_reason"]),
            unknown["reachability_reason"],
        )
        a.check(f"taint.literal.{root['name']}", literal["taint"] == "untainted" and not literal["high_risk"], literal)
        a.check(f"taint.param.{root['name']}", param["taint"] == "tainted" and param["high_risk"], param["taint_reason"])
        a.check(
            f"taint.unresolved.{root['name']}",
            unresolved["taint"] == "unknown" and not unresolved["high_risk"],
            unresolved["taint_reason"],
        )
        a.check(
            f"parse_failure_visible.{root['name']}",
            any(
                failure["root"] == root["name"]
                and failure["file"] == unresolved["location"]["file"]
                for failure in report_ab["parse_failures"]
            ),
            None,
        )
    detail_reach = {
        state: sum(1 for call in report_ab["calls"] if call["reachability"] == state)
        for state in ("reachable", "unreachable", "unknown")
    }
    detail_taint = {
        state: sum(1 for call in report_ab["calls"] if call["taint"] == state)
        for state in ("tainted", "untainted", "unknown")
    }
    detail_high = [call for call in report_ab["calls"] if call["high_risk"]]
    a.check("summary.reachability_counts", detail_reach == report_ab["summary"]["reachability"], detail_reach)
    a.check("summary.taint_counts", detail_taint == report_ab["summary"]["taint"], detail_taint)
    a.check(
        "summary.high_exact",
        {call["id"] for call in detail_high}
        == {
            call["id"]
            for call in report_ab["calls"]
            if call["reachability"] == "reachable" and call["taint"] == "tainted"
        },
        len(detail_high),
    )

    # 4. Business errors: cycle, macro conflict, parse limit, parse continuation.
    def fixture_scan(name: str, extra: list[str] | None = None) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
        root, config = fixture_paths(manifest, name)
        json_path = work / f"fixture-{name}.json"
        completed = a.run(
            [
                "scan",
                "--root",
                str(samples / root),
                "--config-file",
                str(samples / config),
                "--state-dir",
                str(work / "fixture-state"),
                "--json",
                str(json_path),
                *(extra or []),
            ]
        )
        return completed, a.load(json_path)

    cycle_default, cycle_report = fixture_scan("include_cycle")
    a.check("error.include_cycle.exit", cycle_default.returncode == EXIT_CODES["audit_complete_with_errors"], cycle_default.returncode)
    cycle_error = cycle_report["errors"][0]
    assert_error_contract(cycle_error)
    a.check(
        "error.include_cycle.code_context",
        cycle_report["code"] == "E_INCLUDE_CYCLE"
        and "file" in cycle_error["context"]
        and "line" in cycle_error["context"],
        cycle_error["context"],
    )
    cycle_strict, cycle_strict_report = fixture_scan("include_cycle", ["--strict"])
    a.check("error.strict.exit", cycle_strict.returncode == EXIT_CODES["strict_failure"], cycle_strict.returncode)
    a.check("error.strict.same_code", cycle_strict_report["code"] == "E_INCLUDE_CYCLE", cycle_strict_report["code"])

    conflict_default, conflict_report = fixture_scan("macro_conflict")
    conflict_error = conflict_report["errors"][0]
    assert_error_contract(conflict_error)
    a.check(
        "error.macro_conflict",
        conflict_default.returncode == 1
        and conflict_report["code"] == "E_MACRO_CONFLICT"
        and conflict_error["context"].get("line", 0) > 0,
        conflict_error["context"],
    )

    limit_default, limit_report = fixture_scan("parse_limit")
    limit_error = limit_report["errors"][0]
    assert_error_contract(limit_error)
    a.check(
        "error.parse_limit",
        limit_default.returncode == 1
        and limit_report["code"] == "E_PARSE_LIMIT"
        and "file" in limit_error["context"]
        and "line" in limit_error["context"],
        limit_error["context"],
    )

    parse_default, parse_report = fixture_scan("parse_failure")
    a.check(
        "parse_failure.continue",
        parse_default.returncode == 0
        and parse_report["summary"]["parse_failures"] > 0
        and {call["id"] for call in parse_report["calls"]}
        == {"parse_failure.unresolved", "parse_failure.visible"},
        {
            "failures": parse_report["summary"]["parse_failures"],
            "calls": [call["id"] for call in parse_report["calls"]],
        },
    )

    # 5. Report reproducibility, text/JSON parity, and path truncation.
    rep1 = work / "repro-1.json"
    rep2 = work / "repro-2.json"
    for path in (rep1, rep2):
        a.run(
            [
                "scan",
                "--root",
                str(samples),
                "--configs",
                "cfgA,cfgB",
                "--jobs",
                "3",
                "--state-dir",
                str(state_main),
                "--json",
                str(path),
            ],
            expected_exit=0,
        )
    sha1 = hashlib.sha256(rep1.read_bytes()).hexdigest()
    sha2 = hashlib.sha256(rep2.read_bytes()).hexdigest()
    a.check("report.reproducible", sha1 == sha2, {"sha256": sha1})
    repro = a.load(rep1)
    text_calls = parse_text_calls((work / "report-ab.txt").read_text(encoding="utf-8"))
    a.check("report.text_count", len(text_calls) == len(report_ab["calls"]), len(text_calls))
    parity = True
    json_calls = call_map(report_ab)
    for text_call in text_calls:
        key = (text_call["root"], text_call["config"], text_call["id"])
        json_call = json_calls.get(key)
        if json_call is None:
            parity = False
            break
        if (
            text_call["reachability"] != json_call["reachability"]
            or text_call["taint"] != json_call["taint"]
            or text_call["high_risk"] != str(json_call["high_risk"]).lower()
        ):
            parity = False
            break
    a.check("report.text_json_parity", parity, None)

    deep_root, deep_config = fixture_paths(manifest, "deep_path")
    deep_json = work / "deep.json"
    a.run(
        [
            "scan",
            "--root",
            str(samples / deep_root),
            "--config-file",
            str(samples / deep_config),
            "--max-path-hops",
            "2",
            "--state-dir",
            str(work / "deep-state"),
            "--json",
            str(deep_json),
        ],
        expected_exit=0,
    )
    deep_report = a.load(deep_json)
    deep_call = deep_report["calls"][0]
    a.check(
        "report.path_truncated",
        deep_call["path_truncated"] is True
        and deep_call["truncated"] is True
        and len(deep_call["path"]) == 2,
        {
            "truncated": deep_call["path_truncated"],
            "path": deep_call["path"],
        },
    )

    # 6. Snapshot equality, transition diff, and gate pass.
    clean_report_path = work / "clean.json"
    a.run(
        [
            "scan",
            "--root",
            str(samples),
            "--configs",
            "cfgA",
            "--jobs",
            "2",
            "--state-dir",
            str(state_main),
            "--json",
            str(clean_report_path),
        ],
        expected_exit=0,
    )
    clean_report = a.load(clean_report_path)
    baseline_id = clean_report["snapshot_id"]
    same_report_path = work / "same.json"
    a.run(
        [
            "scan",
            "--root",
            str(samples),
            "--configs",
            "cfgA",
            "--jobs",
            "2",
            "--state-dir",
            str(state_main),
            "--baseline",
            baseline_id,
            "--json",
            str(same_report_path),
        ],
        expected_exit=0,
    )
    same_report = a.load(same_report_path)
    diff_same = same_report["baseline"]
    a.check(
        "snapshot.same_diff_empty",
        not diff_same["added"]
        and not diff_same["disappeared"]
        and not diff_same["transitions"]
        and diff_same["content_match"],
        diff_same,
    )
    gate_pass = a.run(
        [
            "scan",
            "--root",
            str(samples),
            "--configs",
            "cfgA",
            "--jobs",
            "2",
            "--state-dir",
            str(state_main),
            "--baseline",
            baseline_id,
            "--fail-on",
            "new-high>0",
            "--json",
            str(work / "gate-pass.json"),
        ],
        expected_exit=EXIT_CODES["audit_clean"],
    )
    gate_pass_report = a.load(work / "gate-pass.json")
    a.check(
        "gate.clean_pass",
        gate_pass.returncode == 0 and gate_pass_report["gate"]["status"] == "passed",
        gate_pass_report["gate"],
    )

    root_a = manifest["roots"][0]
    main_rel = [
        item
        for item in root_a["files"]
        if item.endswith(".c") and "/src_" in item and "main_" in item
    ][0]
    main_path = samples / main_rel
    original_main = main_path.read_text(encoding="utf-8")
    orphan_rel = [
        item
        for item in root_a["files"]
        if item.endswith(".c") and "worker_" in item
    ][0]
    orphan_id = manifest["expected"][root_a["name"]]["unreachable"]["id"]
    marker = "    return argc + payload[0];"
    transitioned = original_main.replace(
        marker,
        "    orphan_copy(buffer, buffer);\n" + marker,
    )
    main_path.write_text(transitioned, encoding="utf-8", newline="\n")
    trans_report_path = work / "transition.json"
    a.run(
        [
            "scan",
            "--root",
            str(samples),
            "--configs",
            "cfgA",
            "--jobs",
            "1",
            "--state-dir",
            str(state_main),
            "--baseline",
            baseline_id,
            "--json",
            str(trans_report_path),
        ],
        expected_exit=0,
    )
    trans_report = a.load(trans_report_path)
    trans_diff = trans_report["baseline"]
    a.check(
        "snapshot.transition_exact",
        not trans_diff["added"]
        and not trans_diff["disappeared"]
        and len(trans_diff["transitions"]) == 1
        and trans_diff["transitions"][0]["id"] == orphan_id
        and trans_diff["transitions"][0]["to"]["reachability"] == "reachable",
        trans_diff,
    )
    trans_gate = a.run(
        [
            "scan",
            "--root",
            str(samples),
            "--configs",
            "cfgA",
            "--jobs",
            "1",
            "--state-dir",
            str(state_main),
            "--baseline",
            baseline_id,
            "--fail-on",
            "transition=reachable",
            "--json",
            str(work / "transition-gate.json"),
        ],
        expected_exit=EXIT_CODES["guard_failure"],
    )
    a.check("gate.transition_failure_exit", trans_gate.returncode == 4, trans_gate.returncode)

    main_path.write_text(
        original_main.replace(
            marker,
            '    system(getenv("INJECT_NEW_HIGH"));\n' + marker,
        ),
        encoding="utf-8",
        newline="\n",
    )
    newhigh_report_path = work / "newhigh.json"
    newhigh_gate = a.run(
        [
            "scan",
            "--root",
            str(samples),
            "--configs",
            "cfgA",
            "--jobs",
            "1",
            "--state-dir",
            str(state_main),
            "--baseline",
            baseline_id,
            "--fail-on",
            "new-high>0",
            "--json",
            str(newhigh_report_path),
        ],
        expected_exit=EXIT_CODES["guard_failure"],
    )
    newhigh_report = a.load(newhigh_report_path)
    newhigh_diff = newhigh_report["baseline"]
    a.check(
        "gate.new_high_failure",
        newhigh_gate.returncode == 4
        and newhigh_diff["new_high_count"] == 1
        and len(newhigh_diff["added"]) == 1
        and newhigh_diff["added"][0]["callee"] == "system"
        and newhigh_diff["added"][0]["high_risk"],
        {
            "new_high_count": newhigh_diff["new_high_count"],
            "added": newhigh_diff["added"],
        },
    )
    main_path.write_text(original_main, encoding="utf-8", newline="\n")

    # 7. Corrupt/truncated snapshot rejection before replacement.
    baseline_file = state_main / "snapshots" / f"{baseline_id}.json"
    corrupt_state = work / "corrupt-state"
    (corrupt_state / "snapshots").mkdir(parents=True, exist_ok=True)
    corrupt_file = corrupt_state / "snapshots" / f"{baseline_id}.json"
    data = baseline_file.read_bytes()
    corrupt_file.write_bytes(data[: len(data) // 2])
    corrupt_before = hashlib.sha256(corrupt_file.read_bytes()).hexdigest()
    corrupt_report_path = work / "corrupt.json"
    corrupt = a.run(
        [
            "scan",
            "--root",
            str(samples),
            "--configs",
            "cfgA",
            "--state-dir",
            str(corrupt_state),
            "--baseline",
            baseline_id,
            "--fail-on",
            "new-high>0",
            "--json",
            str(corrupt_report_path),
        ],
        expected_exit=1,
    )
    corrupt_report = a.load(corrupt_report_path)
    corrupt_error = corrupt_report["errors"][0]
    assert_error_contract(corrupt_error)
    a.check(
        "snapshot.corrupt_rejected",
        corrupt.returncode == 1
        and corrupt_report["code"] == "E_SNAPSHOT_CORRUPT"
        and corrupt_report["baseline"] is None
        and corrupt_report["gate"] is None
        and corrupt_report["calls"] == []
        and hashlib.sha256(corrupt_file.read_bytes()).hexdigest() == corrupt_before,
        corrupt_error,
    )
    tmp_file = corrupt_state / "snapshots" / ".tmp-pending.json"
    tmp_file.write_text('{"truncated":', encoding="utf-8")
    tmp_report_path = work / "tmp-corrupt.json"
    tmp_corrupt = a.run(
        [
            "scan",
            "--root",
            str(samples),
            "--configs",
            "cfgA",
            "--state-dir",
            str(corrupt_state),
            "--baseline",
            str(tmp_file),
            "--json",
            str(tmp_report_path),
        ],
        expected_exit=1,
    )
    tmp_report = a.load(tmp_report_path)
    a.check(
        "snapshot.temp_rejected",
        tmp_corrupt.returncode == 1
        and tmp_report["code"] == "E_SNAPSHOT_CORRUPT"
        and tmp_report["baseline"] is None,
        tmp_report["errors"],
    )

    # 8. Baseline mismatch rejects before threshold comparison.
    other_samples = work / "samples-other-seed"
    a.run(["gen", "--seed", "20240502", "--out", str(other_samples)], expected_exit=0)
    mismatch_path = work / "mismatch.json"
    mismatch = a.run(
        [
            "scan",
            "--root",
            str(other_samples),
            "--configs",
            "cfgA",
            "--state-dir",
            str(work / "mismatch-state"),
            "--baseline",
            str(baseline_file),
            "--fail-on",
            "new-high>0",
            "--json",
            str(mismatch_path),
        ],
        expected_exit=1,
    )
    mismatch_report = a.load(mismatch_path)
    mismatch_error = mismatch_report["errors"][0]
    assert_error_contract(mismatch_error)
    a.check(
        "baseline.mismatch_rejected",
        mismatch.returncode == 1
        and mismatch_report["code"] == "E_BASELINE_MISMATCH"
        and mismatch_report["gate"]["status"] == "rejected"
        and mismatch_report["baseline"]["content_match"] is False,
        {"error": mismatch_error, "gate": mismatch_report["gate"]},
    )

    # 9. Real subprocess timeout, concurrency equivalence, and no residual workers.
    jobs1_path = work / "jobs1.json"
    jobs4_path = work / "jobs4.json"
    a.run(
        [
            "scan",
            "--root",
            str(samples),
            "--configs",
            "cfgA,cfgB,cfgC",
            "--jobs",
            "1",
            "--timeout",
            "1.2",
            "--state-dir",
            str(work / "state-jobs1"),
            "--json",
            str(jobs1_path),
        ],
        expected_exit=1,
        timeout=30,
    )
    a.run(
        [
            "scan",
            "--root",
            str(samples),
            "--configs",
            "cfgA,cfgB,cfgC",
            "--jobs",
            "4",
            "--timeout",
            "1.2",
            "--state-dir",
            str(work / "state-jobs4"),
            "--json",
            str(jobs4_path),
        ],
        expected_exit=1,
        timeout=30,
    )
    jobs1 = a.load(jobs1_path)
    jobs4 = a.load(jobs4_path)
    conclusions_1 = {
        (call["root"], call["config"], call["id"]): (
            call["reachability"],
            call["taint"],
            call["high_risk"],
        )
        for call in jobs1["calls"]
    }
    conclusions_4 = {
        (call["root"], call["config"], call["id"]): (
            call["reachability"],
            call["taint"],
            call["high_risk"],
        )
        for call in jobs4["calls"]
    }
    statuses = [
        (item["root"], item["config"], item["status"])
        for item in jobs4["combinations"]
    ]
    timeout_error = jobs4["errors"][0]
    assert_error_contract(timeout_error)
    residual: list[str] = []
    if os.name != "nt":
        process_list = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,args="],
            text=True,
            capture_output=True,
            check=False,
        )
        residual = [
            line
            for line in process_list.stdout.splitlines()
            if "audit_workbench _worker" in line
        ]
    a.check(
        "concurrency.timeout_isolated",
        len(jobs4["combinations"]) == 9
        and sum(1 for item in jobs4["combinations"] if item["status"] == "ok") == 8
        and ("unit_c", "cfgC", "timed_out") in statuses
        and timeout_error["code"] == "E_TIMEOUT"
        and "combination" in timeout_error["context"],
        statuses,
    )
    a.check("concurrency.jobs_equivalent", conclusions_1 == conclusions_4, len(conclusions_1))
    a.check("concurrency.byte_reproducible", jobs1_path.read_bytes() == jobs4_path.read_bytes(), None)
    a.check(
        "concurrency.no_residual_workers",
        not residual and jobs4["summary"]["active_children"] == 0,
        residual,
    )

    # 10. Help/metadata exit contract and tool-failure separation.
    help_run = a.run(["--help"], expected_exit=0)
    help_values = {
        name: int(value)
        for name, value in re.findall(r"(\w+)=(\d+)", help_run.stdout)
    }
    metadata_run = a.run(["metadata", "--json"], expected_exit=0)
    metadata_values = a.loads(metadata_run.stdout)
    a.check(
        "exit.help_metadata_consistency",
        help_values.get("audit_clean") == EXIT_CODES["audit_clean"]
        and help_values.get("audit_complete_with_errors")
        == EXIT_CODES["audit_complete_with_errors"]
        and help_values.get("tool_failure") == EXIT_CODES["tool_failure"]
        and help_values.get("strict_failure") == EXIT_CODES["strict_failure"]
        and help_values.get("guard_failure") == EXIT_CODES["guard_failure"]
        and metadata_values["exit_codes"] == EXIT_CODES,
        {"help": help_values, "metadata": metadata_values["exit_codes"]},
    )
    invalid = a.run(
        [
            "scan",
            "--root",
            str(samples),
            "--jobs",
            "0",
            "--state-dir",
            str(work / "invalid-state"),
        ],
        expected_exit=EXIT_CODES["tool_failure"],
    )
    invalid_error = a.loads(invalid.stderr)
    assert_error_contract(invalid_error)
    a.check(
        "exit.invalid_args",
        invalid.returncode == EXIT_CODES["tool_failure"]
        and invalid_error["code"] == "E_ARGS"
        and invalid.returncode
        != EXIT_CODES["audit_complete_with_errors"],
        invalid_error,
    )

    elapsed = round(time.time() - started, 3)
    result = {
        "format": "acceptance-results-v1",
        "status": "passed",
        "python": sys.version,
        "platform": sys.platform,
        "work_dir": str(work),
        "elapsed_seconds": elapsed,
        "checks_total": len(a.checks),
        "checks": a.checks,
        "exit_codes": EXIT_CODES,
        "notes": [
            "The target Windows command is: py -3.11 -m audit_workbench ...",
            "This run used the available Linux Python because no py launcher is installed here.",
            "All source packages are generated locally from a fixed seed.",
        ],
    }
    results_path.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    a.log(f"ALL {len(a.checks)} CHECKS PASSED in {elapsed}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

