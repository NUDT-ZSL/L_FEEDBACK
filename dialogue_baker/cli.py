"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .baker import (
    EXIT_FATAL,
    EXIT_OK,
    bake_project,
    write_fatal_report,
)
from .constants import BAKER_VERSION
from .errors import BakeFailure, InputFatal, Issue
from .input import load_input


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dialogue-baker",
        description=(
            "Bake deterministic 48 kHz mono multilingual dialogue PCM, "
            "a chunk index, and a reporting artifact."
        ),
    )
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("full", "incremental"),
        default="incremental",
        help="full re-synthesizes every entry; incremental reuses verified chunks",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not print the one-line JSON summary",
    )
    parser.add_argument("--version", action="version", version=BAKER_VERSION)
    return parser


def _print_summary(result_payload: dict[str, object]) -> None:
    print(json.dumps(result_payload, ensure_ascii=False, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_path: Path = args.project.resolve()
    config_path: Path = args.config.resolve()
    output_dir: Path = args.output_dir.resolve()

    try:
        loaded = load_input(project_path, config_path)
    except InputFatal as exc:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            write_fatal_report(
                output_dir,
                mode=args.mode,
                issue=exc.issue,
                project_path=project_path,
                config_path=config_path,
            )
        except OSError as report_exc:
            print(
                f"fatal input error and report write failed: "
                f"{exc.issue.message}; {report_exc}",
                file=sys.stderr,
            )
        else:
            print(
                f"fatal input error [{exc.issue.code}] {exc.issue.message}",
                file=sys.stderr,
            )
        return EXIT_FATAL

    try:
        result = bake_project(loaded, output_dir, mode=args.mode)
    except BakeFailure as exc:
        issue: Issue = exc.issue
        try:
            write_fatal_report(
                output_dir,
                mode=args.mode,
                issue=issue,
                project_path=project_path,
                config_path=config_path,
            )
        except OSError as report_exc:
            print(
                f"bake failed and report write failed: {issue.message}; {report_exc}",
                file=sys.stderr,
            )
        else:
            print(f"bake failed [{issue.code}] {issue.message}", file=sys.stderr)
        return EXIT_FATAL
    except OSError as exc:
        issue = Issue(
            "OUTPUT_WRITE_FAILED",
            "output directory could not be prepared",
            {"output_dir": str(output_dir), "error": str(exc)},
        )
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            write_fatal_report(
                output_dir,
                mode=args.mode,
                issue=issue,
                project_path=project_path,
                config_path=config_path,
            )
        except OSError as report_exc:
            print(
                f"failure report write also failed: {report_exc}",
                file=sys.stderr,
            )
        print(f"bake failed [{issue.code}] {issue.message}", file=sys.stderr)
        return EXIT_FATAL

    if not args.quiet:
        _print_summary(
            {
                "status": result.report["final_status"],
                "exit_code": result.exit_code,
                "resource_version": result.index["resource_version"],
                "input_version": result.index["input_version"],
                "entries": result.index["summary"]["entry_count"],
                "skipped": result.report["summary"]["skipped_entries"],
                "pack": str(result.pack_path),
                "index": str(output_dir / "index.json"),
                "report": str(output_dir / "bake_report.json"),
            }
        )
    return result.exit_code if result.exit_code in (EXIT_OK, 2) else EXIT_FATAL


if __name__ == "__main__":
    raise SystemExit(main())

