"""Command-line entry point for the patch audit."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable

from .audit import (
    build_audit_report,
    report_json_bytes,
    summary_json_bytes,
    write_audit_report,
)
from .errors import AuditError, EXIT_IO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pe_audit",
        description=(
            "Audit a deterministic PE-style before/after pair using "
            "function-name identity alignment."
        ),
    )
    parser.add_argument(
        "--before",
        "--before-bin",
        dest="before",
        required=True,
        help="patch-before binary",
    )
    parser.add_argument(
        "--after",
        "--after-bin",
        dest="after",
        required=True,
        help="patch-after binary",
    )
    parser.add_argument(
        "--before-manifest",
        "--manifest-before",
        dest="before_manifest",
        required=True,
        help="patch-before function-boundary JSON",
    )
    parser.add_argument(
        "--after-manifest",
        "--manifest-after",
        dest="after_manifest",
        required=True,
        help="patch-after function-boundary JSON",
    )
    parser.add_argument("--defects", required=True, help="defect-rule JSON")
    parser.add_argument(
        "--entries",
        help="optional entry-point JSON; otherwise use defects/manifests",
    )
    parser.add_argument(
        "--output",
        "--report",
        "--out",
        dest="output",
        default="audit_report.json",
        help="report output path (default: audit_report.json)",
    )
    parser.add_argument(
        "--stdout",
        choices=("summary", "report", "none"),
        default="summary",
        help="what to print on success (default: summary)",
    )
    return parser


def _print_error(error: AuditError) -> None:
    text = json.dumps(
        error.to_dict(), ensure_ascii=False, sort_keys=True
    ) + "\n"
    sys.stderr.write(text)


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        report = build_audit_report(
            before_path=args.before,
            after_path=args.after,
            before_manifest_path=args.before_manifest,
            after_manifest_path=args.after_manifest,
            defects_path=args.defects,
            entries_path=args.entries,
        )
        write_audit_report(args.output, report)
    except AuditError as exc:
        _print_error(exc)
        return exc.exit_code
    except OSError as exc:
        error = AuditError(
            "WRITE_ERROR", f"could not write report: {exc}", EXIT_IO
        )
        _print_error(error)
        return error.exit_code
    if args.stdout == "summary":
        sys.stdout.write(summary_json_bytes(report).decode("utf-8"))
    elif args.stdout == "report":
        sys.stdout.write(report_json_bytes(report).decode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
