from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .config import load_config
from .errors import ConfigError, ProcessingError, VerificationFailure
from .pipeline import run_pipeline
from .stats import build_report, load_processing_log, write_report
from .verify import verify_run


def _json_print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _error_print(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radio-pipeline",
        description=(
            "Deterministic offline jitter-buffer, PLC, and FFmpeg transcode pipeline"
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="generate/receive packets and transcode")
    run_parser.add_argument("--config", required=True, type=Path, help="JSON configuration")
    run_parser.add_argument(
        "--output-dir", required=True, type=Path, help="artifact directory"
    )
    run_parser.add_argument(
        "--input",
        type=Path,
        help="optional preconstructed packet JSONL; generated deterministically if omitted",
    )

    report_parser = subparsers.add_parser(
        "report", help="rebuild report.json from processing logs"
    )
    report_parser.add_argument(
        "--run-dir", required=True, type=Path, help="run artifact directory"
    )
    report_parser.add_argument(
        "--output",
        type=Path,
        help="report path (default: RUN_DIR/report.json)",
    )

    verify_parser = subparsers.add_parser(
        "verify", help="verify audio timelines, sample counts, metadata, and loudness"
    )
    verify_parser.add_argument(
        "--run-dir", required=True, type=Path, help="run artifact directory"
    )
    verify_parser.add_argument(
        "--report",
        type=Path,
        help="report path (default: RUN_DIR/report.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            config = load_config(args.config)
            summary = run_pipeline(config, args.output_dir, args.input)
            _json_print(summary)
            return 0
        if args.command == "report":
            log_path = args.run_dir / "logs" / "processing_log.json"
            log = load_processing_log(log_path)
            report = build_report(log)
            output_path = args.output or (args.run_dir / "report.json")
            write_report(report, output_path)
            _json_print(
                {
                    "schema": report["schema"],
                    "report": str(output_path.resolve()),
                    "counts": report["counts"],
                    "rates": report["rates"],
                    "buffer": report["buffer"],
                    "output": report["output"],
                    "degradation": report["degradation"],
                    "return_code": 0,
                }
            )
            return 0
        if args.command == "verify":
            result, code = verify_run(args.run_dir, args.report)
            _json_print(result)
            return code
    except ConfigError as exc:
        _error_print(str(exc))
        return 2
    except VerificationFailure as exc:
        _error_print(str(exc))
        return 3
    except ProcessingError as exc:
        _error_print(str(exc))
        return 2
    except OSError as exc:
        _error_print(f"filesystem error: {exc}")
        return 2
    _error_print(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
