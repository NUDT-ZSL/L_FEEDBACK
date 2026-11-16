from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .errors import BusinessError, ProcessingError, emit_error
from .processor import dry_run_manifest, process_manifest


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # pragma: no cover - exercised through CLI
        raise BusinessError(1000, f"invalid command line: {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="broadcast-master",
        description="Offline multilingual broadcast master generator",
    )
    parser.add_argument("--manifest", required=True, help="broadcast manifest JSON")
    parser.add_argument("--output", help="24-bit PCM WAV output path")
    parser.add_argument("--report", help="processing report JSON path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse and simulate the timeline/sync estimate without writing media",
    )
    parser.add_argument(
        "--temp-root",
        default=None,
        help="optional root directory for recovery markers and staging",
    )
    return parser


def _print_json(payload: dict) -> None:
    sys.stdout.write(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    sys.stdout.flush()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if args.dry_run:
            _print_json(dry_run_manifest(args.manifest))
            return 0
        if not args.output:
            raise BusinessError(1000, "--output is required unless --dry-run is used")
        if not args.report:
            raise BusinessError(1000, "--report is required unless --dry-run is used")
        summary, _ = process_manifest(
            args.manifest,
            args.output,
            args.report,
            temp_root=args.temp_root,
        )
        _print_json(summary)
        return 0
    except BusinessError as exc:
        emit_error(exc)
        return 2
    except ProcessingError as exc:
        emit_error(exc)
        return 3
    except KeyboardInterrupt:
        emit_error(ProcessingError(3130, "processing interrupted"))
        return 130
    except Exception as exc:
        emit_error(ProcessingError(3999, f"unexpected failure: {exc}"))
        return 4


if __name__ == "__main__":
    raise SystemExit(main())

