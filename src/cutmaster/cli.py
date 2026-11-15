"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .errors import MasteringError, err
from .manifest import load_manifest
from .pipeline import run_from_paths


class _StructuredArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise err("E_CLI_USAGE", message, exit_code=2)


def _parser() -> argparse.ArgumentParser:
    parser = _StructuredArgumentParser(
        prog="cutmaster",
        description=(
            "Build a deterministic stereo master from a JSON absolute-time "
            "timeline, with dialogue ducking, EBU R128 normalization, "
            "checkpoint recovery and an auditable timing report."
        ),
        epilog=(
            "Exit codes: 0 success; 2 manifest/CLI; 3 input; 4 timeline; "
            "5 ducking; 6 loudness; 7 state/IO; 8 internal."
        ),
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--report", "-r", type=Path)
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="persistent state directory (default: <manifest-dir>/.cutmaster-state)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore a committed checkpoint; valid per-track cache is still reusable",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the success summary on stdout",
    )
    return parser


def _emit_error(error: MasteringError) -> int:
    print(
        json.dumps(error.as_dict(), allow_nan=False),
        file=sys.stderr,
        flush=True,
    )
    return int(error.exit_code if error.exit_code > 0 else 8)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "master":
        arguments = arguments[1:]
    parser = _parser()
    try:
        args = parser.parse_args(arguments)
    except SystemExit as exc:
        return int(exc.code)
    except MasteringError as exc:
        return _emit_error(exc)

    manifest_path = args.manifest.resolve(strict=False)
    state_dir = (
        args.state_dir
        if args.state_dir is not None
        else manifest_path.parent / ".cutmaster-state"
    )
    try:
        manifest = load_manifest(manifest_path)
        master_path = (
            args.output.resolve(strict=False)
            if args.output is not None
            else manifest.output.master
        )
        report_path = (
            args.report.resolve(strict=False)
            if args.report is not None
            else manifest.output.report
        )
        if master_path is None or report_path is None:
            raise err(
                "E_OUTPUT_REQUIRED",
                "master and report output paths are required",
                hint=(
                    "Pass --output/--report or set output.master/output.report "
                    "in the manifest"
                ),
            )
        result = run_from_paths(
            manifest_path=manifest_path,
            master_path=master_path,
            report_path=report_path,
            state_dir=state_dir,
            force=args.force,
        )
        if not args.quiet:
            print(json.dumps(result, allow_nan=False))
        return 0
    except MasteringError as exc:
        return _emit_error(exc)
    except KeyboardInterrupt:
        return _emit_error(
            err(
                "E_INTERRUPTED",
                "mastering was interrupted",
                exit_code=130,
            )
        )
    except OSError as exc:
        return _emit_error(
            err(
                "E_IO",
                "filesystem operation failed",
                exit_code=7,
                reason=str(exc),
            )
        )
    except Exception as exc:  # pragma: no cover - last-resort CLI boundary
        return _emit_error(
            err(
                "E_INTERNAL",
                "unexpected internal error",
                exit_code=8,
                exception=type(exc).__name__,
                reason=str(exc),
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
