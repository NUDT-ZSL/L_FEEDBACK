"""Command line entry point: generate/run/batch/verify."""

from __future__ import annotations

import argparse
import json
import sys

from .errors import ErrorCode, MixerError
from .pipeline import run_batch_file, run_config_file, verify_config_reports
from .report import sanitize_json


EXIT_OK = 0
EXIT_BUSINESS_ERROR = 2


def _print(payload: dict | list) -> None:
    print(json.dumps(sanitize_json(payload), indent=2, sort_keys=True, allow_nan=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiomixer",
        description="Offline deterministic multi-track audio mix processor.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="generate assets and mix one scene")
    run_parser.add_argument("config", help="path to a scene JSON config")
    run_parser.add_argument(
        "--no-generate", action="store_true", help="use existing WAV assets"
    )

    batch_parser = sub.add_parser("batch", help="process multiple scene configs")
    batch_parser.add_argument("config", help="path to a batch JSON config")
    batch_parser.add_argument("--summary", help="summary report output path")

    verify_parser = sub.add_parser("verify", help="re-verify one or more mix reports")
    verify_parser.add_argument("reports", nargs="+", help="mix report JSON paths")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            result = run_config_file(args.config, generate=False if args.no_generate else None)
            _print(result)
            return EXIT_OK if result["status"] == "ok" else EXIT_BUSINESS_ERROR

        if args.command == "batch":
            summary, _ = run_batch_file(args.config, args.summary)
            _print(summary)
            return EXIT_OK if summary["failed_count"] == 0 else EXIT_BUSINESS_ERROR

        if args.command == "verify":
            verifications = verify_config_reports(args.reports)
            _print(verifications)
            return EXIT_OK if all(item["checks"]["all_passed"] for item in verifications) else EXIT_BUSINESS_ERROR
    except MixerError as exc:
        _print(exc.to_payload())
        return EXIT_BUSINESS_ERROR
    except FileNotFoundError as exc:
        _print(
            {
                "status": "error",
                "error_code": ErrorCode.MISSING_ASSET,
                "message": str(exc),
            }
        )
        return EXIT_BUSINESS_ERROR

    parser.error(f"unknown command {args.command!r}")
    return EXIT_BUSINESS_ERROR


if __name__ == "__main__":
    sys.exit(main())
