from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .errors import BakerError
from .pipeline import run_bake, write_error_report
from .snapshot import STAGE_ORDER


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dialogue_baker",
        description="Deterministic offline dialogue spatialization baker.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    bake = subparsers.add_parser("bake", help="Run the five-stage bake pipeline.")
    bake.add_argument("--project", required=True, help="Project JSON manifest.")
    bake.add_argument("--out", required=True, help="Output directory.")
    bake.add_argument(
        "--stop-after",
        choices=list(STAGE_ORDER),
        default=None,
        help="Stop after a completed stage to test resumable snapshots.",
    )
    bake.add_argument(
        "--force",
        action="store_true",
        help="Discard stage snapshots and rebuild from parse.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] not in {"bake", "-h", "--help"}:
        raw_args.insert(0, "bake")
    args = parser.parse_args(raw_args)
    if args.command != "bake":
        parser.error("unknown command")
    out_dir = Path(args.out)
    try:
        result = run_bake(
            project_path=Path(args.project),
            out_dir=out_dir,
            stop_after=args.stop_after,
            force=args.force,
        )
        summary = {
            "completed_stage": result.completed_stage,
            "report": str(result.report_path) if result.report_path else None,
            "stage_actions": result.stage_actions,
            "status": result.status,
        }
        print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
        return 0
    except BakerError as error:
        error_path = write_error_report(out_dir, error)
        print(
            json.dumps(error.as_dict(), ensure_ascii=True, sort_keys=True),
            file=sys.stderr,
        )
        if error_path is not None:
            print(str(error_path), file=sys.stderr)
        return 2
    except Exception as error:
        wrapped = BakerError(
            "E_INTERNAL",
            "An unexpected internal error occurred.",
            {"type": type(error).__name__, "reason": str(error)},
        )
        write_error_report(out_dir, wrapped)
        print(
            json.dumps(wrapped.as_dict(), ensure_ascii=True, sort_keys=True),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

