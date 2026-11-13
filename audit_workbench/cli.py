"""Command-line interface for the offline audit workbench."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

from . import RULES_VERSION, __version__
from .config import load_config_file, resolve_config
from .engine import (
    apply_baseline_and_gate,
    build_report,
    discover_roots,
    load_scan_configs,
    minimal_error_report,
    report_json,
    report_text,
    run_audit,
)
from .errors import (
    AUDIT_ERROR_CODES,
    EXIT_AUDIT_CLEAN,
    EXIT_AUDIT_COMPLETE_WITH_ERRORS,
    EXIT_CODES,
    EXIT_GUARD_FAILURE,
    EXIT_STRICT_FAILURE,
    EXIT_TOOL_FAILURE,
    AuditError,
    WorkbenchError,
)
from .generator import generate
from .preprocessor import (
    MAX_EXPANSION_DEPTH,
    MAX_INCLUDE_DEPTH,
    MAX_MACRO_EXPANSIONS,
    MAX_VIRTUAL_LINES,
)
from .snapshot import ensure_state_dir, list_snapshots, load_snapshot
from .worker import worker_main


EXIT_HELP = """\
EXIT CONTRACT (also exposed as machine-readable metadata):
  audit_clean={clean}
  guard_pass={clean}
  audit_complete_with_errors={completed}
  tool_failure={tool}
  strict_failure={strict}
  guard_failure={guard}

--strict changes a completed audit containing error objects from the
"audit_complete_with_errors" code to "strict_failure". --fail-on evaluates
only baseline differences and uses "guard_failure" when a gate is triggered.
""".format(
    clean=EXIT_AUDIT_CLEAN,
    completed=EXIT_AUDIT_COMPLETE_WITH_ERRORS,
    tool=EXIT_TOOL_FAILURE,
    strict=EXIT_STRICT_FAILURE,
    guard=EXIT_GUARD_FAILURE,
)


BUSINESS_ERROR_CODES = {
    "E_INCLUDE_CYCLE",
    "E_MACRO_CONFLICT",
    "E_SNAPSHOT_CORRUPT",
    "E_TIMEOUT",
    "E_PARSE_LIMIT",
    "E_BASELINE_MISMATCH",
    "E_GUARD_FAILED",
    "E_CONFIG",
}


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        error = AuditError(
            "E_ARGS",
            message or "invalid arguments",
            {"line": 1, "parser": self.prog},
        ).to_dict()
        payload = json.dumps(error, ensure_ascii=False, sort_keys=True) + "\n"
        sys.stdout.write(payload)
        sys.stderr.write(payload)
        raise SystemExit(EXIT_TOOL_FAILURE)


def metadata() -> dict[str, Any]:
    return {
        "tool_version": __version__,
        "rules_version": RULES_VERSION,
        "exit_codes": dict(EXIT_CODES),
        "error_codes": sorted(AUDIT_ERROR_CODES),
        "limits": {
            "include_depth": MAX_INCLUDE_DEPTH,
            "virtual_lines": MAX_VIRTUAL_LINES,
            "macro_expansions": MAX_MACRO_EXPANSIONS,
            "expansion_depth": MAX_EXPANSION_DEPTH,
        },
        "report_formats": ["json", "text"],
        "runtime": "Python 3.11+; no compiler or network required",
    }


def _write_text(path_value: str, content: str) -> None:
    path = Path(path_value)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise WorkbenchError(
            "E_IO",
            f"cannot write output file: {exc}",
            {"file": str(path), "line": 1},
        ) from exc


def _emit_report(
    report: dict[str, Any],
    *,
    json_path: str | None = None,
    text_path: str | None = None,
    print_json: bool = True,
) -> None:
    json_content = report_json(report)
    if json_path and json_path != "-":
        _write_text(json_path, json_content)
    if text_path:
        _write_text(text_path, report_text(report))
    if print_json:
        sys.stdout.write(json_content)


def _completed_exit(report: dict[str, Any], strict: bool) -> int:
    if report.get("gate") and report["gate"].get("status") == "failed":
        return EXIT_GUARD_FAILURE
    if report.get("errors"):
        return EXIT_STRICT_FAILURE if strict else EXIT_AUDIT_COMPLETE_WITH_ERRORS
    return EXIT_AUDIT_CLEAN


def command_gen(args: argparse.Namespace) -> int:
    try:
        manifest = generate(args.seed, Path(args.out), force=args.force)
    except WorkbenchError:
        raise
    summary = {
        "status": "generated",
        "seed": args.seed,
        "out": str(Path(args.out)),
        "roots": [
            {
                "name": root["name"],
                "path": root["path"],
                "files": len(root["files"]),
            }
            for root in manifest["roots"]
        ],
        "configs": manifest["configs"],
        "fixtures": sorted(manifest["fixtures"]),
    }
    sys.stdout.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")
    return EXIT_AUDIT_CLEAN


def command_scan(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir)
    try:
        ensure_state_dir(state_dir)
        baseline_payload = None
        baseline_id = args.baseline
        if args.fail_on and not baseline_id:
            raise WorkbenchError(
                "E_ARGS",
                "--fail-on requires --baseline",
                {"option": "--fail-on", "line": 1},
            )
        if baseline_id:
            try:
                baseline_payload, baseline_path = load_snapshot(state_dir, baseline_id)
                baseline_id = baseline_path.stem
            except WorkbenchError as exc:
                report = minimal_error_report(exc.to_dict())
                _emit_report(
                    report,
                    json_path=args.json,
                    text_path=args.text,
                    print_json=True,
                )
                return EXIT_STRICT_FAILURE if args.strict else EXIT_AUDIT_COMPLETE_WITH_ERRORS
        roots, config_dirs = discover_roots(args.root)
        configs = load_scan_configs(
            args.configs.split(",") if args.configs else [],
            args.config_file or [],
            config_dirs,
            state_dir,
        )
        payload, snapshot_id = run_audit(
            roots,
            configs,
            state_dir=state_dir,
            jobs=args.jobs,
            timeout=args.timeout,
            max_hops=args.max_path_hops,
        )
        gate_specs = [
            item
            for value in (args.fail_on or [])
            for item in value.split(",")
            if item.strip()
        ]
        diff, errors, gate = apply_baseline_and_gate(
            payload,
            state_dir=state_dir,
            baseline_id=baseline_id,
            baseline_payload=baseline_payload,
            gate_specs=gate_specs,
            strict_content_lock=args.strict_content_lock,
        )
        report = build_report(
            payload,
            snapshot_id,
            diff=diff,
            errors=errors,
            gate=gate,
        )
        _emit_report(
            report,
            json_path=args.json,
            text_path=args.text,
            print_json=True,
        )
        return _completed_exit(report, args.strict)
    except WorkbenchError as exc:
        if exc.code in BUSINESS_ERROR_CODES:
            report = minimal_error_report(exc.to_dict())
            _emit_report(
                report,
                json_path=args.json,
                text_path=args.text,
                print_json=True,
            )
            return EXIT_STRICT_FAILURE if args.strict else EXIT_AUDIT_COMPLETE_WITH_ERRORS
        raise


def command_metadata(args: argparse.Namespace) -> int:
    data = metadata()
    if args.json:
        sys.stdout.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")
    else:
        sys.stdout.write(
            f"tool_version={data['tool_version']}\n"
            f"rules_version={data['rules_version']}\n"
        )
        for name, value in sorted(data["exit_codes"].items()):
            sys.stdout.write(f"audit_exit_code {name}={value}\n")
        for code in data["error_codes"]:
            sys.stdout.write(f"business_error_code={code}\n")
        for name, value in sorted(data["limits"].items()):
            sys.stdout.write(f"limit {name}={value}\n")
    return EXIT_AUDIT_CLEAN


def command_config_normalize(args: argparse.Namespace) -> int:
    config = load_config_file(Path(args.input))
    serialized = config.serialized()
    if args.output:
        _write_text(args.output, serialized)
    sys.stdout.write(serialized)
    return EXIT_AUDIT_CLEAN


def command_config_compare(args: argparse.Namespace) -> int:
    left = resolve_config(args.left)
    right = resolve_config(args.right)
    equivalent = left.equivalent(right)
    result = {
        "equivalent": equivalent,
        "left": {"name": left.name, "config_hash": left.content_hash()},
        "right": {"name": right.name, "config_hash": right.content_hash()},
    }
    sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    return EXIT_AUDIT_CLEAN if equivalent else EXIT_AUDIT_COMPLETE_WITH_ERRORS


def command_snapshot(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir)
    if args.action == "list":
        entries = list_snapshots(state_dir)
        sys.stdout.write(
            json.dumps(entries, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        )
        return EXIT_AUDIT_CLEAN
    if args.action in {"show", "verify"}:
        payload, path = load_snapshot(state_dir, args.snapshot)
        if args.action == "verify":
            result = {
                "code": "OK",
                "snapshot_id": path.stem,
                "valid": True,
            }
            sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        else:
            sys.stdout.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            )
        return EXIT_AUDIT_CLEAN
    raise WorkbenchError("E_ARGS", "unknown snapshot action", {"line": 1})


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(
        prog="audit_workbench",
        description=(
            "Deterministic offline C/C++ source-admission auditor. "
            "All samples and baselines are local; no network access is used."
        ),
        epilog=EXIT_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser(
        "gen",
        help="generate deterministic source roots, configs, and fixtures",
    )
    gen_parser.add_argument("--seed", type=int, required=True)
    gen_parser.add_argument("--out", required=True)
    gen_parser.add_argument("--force", action="store_true")
    gen_parser.set_defaults(func=command_gen)

    scan_parser = subparsers.add_parser(
        "scan",
        help="audit one or more source roots under one or more configurations",
    )
    scan_parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="source root or generated bundle directory (repeatable/comma-separated)",
    )
    scan_parser.add_argument(
        "--configs",
        default=None,
        help="comma-separated config IDs/files (default: cfgA)",
    )
    scan_parser.add_argument(
        "--config-file",
        action="append",
        default=[],
        help="additional configuration JSON file (repeatable)",
    )
    scan_parser.add_argument("--jobs", type=_positive_int, default=1)
    scan_parser.add_argument("--timeout", type=_positive_float, default=10.0)
    scan_parser.add_argument("--state-dir", default=".audit_state")
    scan_parser.add_argument("--json", help="JSON report output path or '-'")
    scan_parser.add_argument("--text", help="human-readable report output path")
    scan_parser.add_argument("--baseline", help="snapshot ID or committed snapshot path")
    scan_parser.add_argument(
        "--fail-on",
        action="append",
        default=[],
        metavar="SPEC",
        help="new-high>N, new-high>=N, transition=reachable, or transition=field:old->new",
    )
    scan_parser.add_argument(
        "--strict",
        action="store_true",
        help="return the strict_failure exit code when error objects are present",
    )
    scan_parser.add_argument(
        "--strict-content-lock",
        action="store_true",
        help="reject a gate when any source/config content hash differs",
    )
    scan_parser.add_argument(
        "--max-path-hops",
        type=_nonnegative_int,
        default=8,
        help="maximum retained reachability hops (default: 8)",
    )
    scan_parser.set_defaults(func=command_scan)

    metadata_parser = subparsers.add_parser(
        "metadata",
        help="print machine-readable exit/error contract",
    )
    metadata_parser.add_argument("--json", action="store_true")
    metadata_parser.set_defaults(func=command_metadata)

    config_parser = subparsers.add_parser("config", help="configuration utilities")
    config_sub = config_parser.add_subparsers(dest="config_action", required=True)
    normalize_parser = config_sub.add_parser("normalize")
    normalize_parser.add_argument("--input", required=True)
    normalize_parser.add_argument("--output")
    normalize_parser.set_defaults(func=command_config_normalize)
    compare_parser = config_sub.add_parser("compare")
    compare_parser.add_argument("--left", required=True)
    compare_parser.add_argument("--right", required=True)
    compare_parser.set_defaults(func=command_config_compare)

    snapshot_parser = subparsers.add_parser("snapshot", help="snapshot utilities")
    snapshot_sub = snapshot_parser.add_subparsers(dest="action", required=True)
    for action in ("list", "show", "verify"):
        item = snapshot_sub.add_parser(action)
        item.add_argument("--state-dir", default=".audit_state")
        if action in {"show", "verify"}:
            item.add_argument("snapshot")
        item.set_defaults(func=command_snapshot)

    worker_parser = subparsers.add_parser("_worker")
    worker_parser.add_argument("payload")
    worker_parser.add_argument("result")
    worker_parser.set_defaults(func=lambda args: worker_main(Path(args.payload), Path(args.result)))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except WorkbenchError as exc:
        if exc.code in BUSINESS_ERROR_CODES:
            report = minimal_error_report(exc.to_dict())
            sys.stdout.write(report_json(report))
            return EXIT_AUDIT_COMPLETE_WITH_ERRORS
        payload = json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        sys.stdout.write(payload)
        sys.stderr.write(payload)
        return EXIT_TOOL_FAILURE
    except OSError as exc:
        error = AuditError(
            "E_IO",
            f"tool I/O failure: {exc}",
            {"line": 1},
        ).to_dict()
        payload = json.dumps(error, ensure_ascii=False, sort_keys=True) + "\n"
        sys.stdout.write(payload)
        sys.stderr.write(payload)
        return EXIT_TOOL_FAILURE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

