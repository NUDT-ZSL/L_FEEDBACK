"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .errors import RepairFailure, make_error
from .jsonutil import fixed_json
from .processor import process_files
from .samples import generate_sample_bundle
from .selftest import run_selftest


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # pragma: no cover - argparse path
        payload = {
            "ok": False,
            "exit_code": 2,
            "error_count": 1,
            "errors": [
                make_error(
                    "cli_usage_error",
                    message,
                )
            ],
        }
        print(fixed_json(payload, indent=None), file=sys.stderr)
        raise SystemExit(2)


def _parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(prog="repair.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="apply a repair manifest")
    process.add_argument("--input", "-i", required=True, help="input 48 kHz mono WAV")
    process.add_argument("--manifest", "-m", required=True, help="repair JSON manifest")
    process.add_argument("--output", "-o", required=True, help="output repaired WAV")
    process.add_argument("--report", "-r", required=True, help="output JSON report")

    generate = subparsers.add_parser(
        "generate-sample", help="write the deterministic built-in fixture"
    )
    generate.add_argument("--output-dir", "-d", required=True)

    selftest = subparsers.add_parser(
        "selftest", help="run deterministic end-to-end checks"
    )
    selftest.add_argument(
        "--workdir",
        help="keep generated files in this directory instead of a temporary directory",
    )
    return parser


def _emit(payload: dict[str, Any], *, stream: Any = sys.stdout) -> None:
    print(fixed_json(payload, indent=2), file=stream)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "process":
            result = process_files(
                args.input,
                args.manifest,
                args.output,
                args.report,
            )
            _emit(
                {
                    "ok": True,
                    "sample_count": result.sample_count,
                    "wav_sha256": result.wav_sha256,
                    "report_sha256": result.report_sha256,
                    "output": str(result.output_path),
                    "report": str(result.report_path),
                }
            )
            return 0
        if args.command == "generate-sample":
            bundle = generate_sample_bundle(args.output_dir)
            _emit(
                {
                    "ok": True,
                    "dialogue_wav": str(bundle.dialogue_wav),
                    "reference_wav": str(bundle.reference_wav),
                    "manifest": str(bundle.manifest_json),
                }
            )
            return 0
        if args.command == "selftest":
            result = run_selftest(args.workdir)
            _emit(result)
            return 0 if result["ok"] else 1
        raise AssertionError("unreachable command")
    except RepairFailure as failure:
        print(fixed_json(failure.payload(), indent=None), file=sys.stderr)
        return failure.exit_code
    except Exception as exc:
        payload = {
            "ok": False,
            "exit_code": 3,
            "error_count": 1,
            "errors": [
                make_error(
                    "internal_error",
                    f"unexpected failure: {exc}",
                )
            ],
        }
        print(fixed_json(payload, indent=None), file=sys.stderr)
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

