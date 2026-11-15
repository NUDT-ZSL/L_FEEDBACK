"""Command-line interface for the mastering tool."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from . import __version__
from .errors import MasterError, internal_error
from .pipeline import run_mastering, write_error_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cinema-master",
        description=(
            "Offline dialogue-priority 5.1/stereo mastering, true-peak limiting, "
            "WAV mastering and MP4 packaging."
        ),
    )
    parser.add_argument("input_dir", help="Directory containing WAV tracks and JSON config")
    parser.add_argument("output_dir", help="Directory for WAV, MP4 and report.json")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config path (relative to input_dir); defaults to config.json",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s " + __version__
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_mastering(args.input_dir, args.output_dir, args.config)
        summary = {
            "status": "ok",
            "wav": report["output"]["wav"]["path"],
            "mp4": report["output"]["container"]["path"],
            "report": report["output"]["report_path"],
            "program_lufs": report["loudness"]["program_lufs"],
            "true_peak_dbtp": report["true_peak"]["measured_dbtp"],
            "wav_sha256": report["output"]["wav"]["sha256"],
        }
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    except MasterError as exc:
        report_path = write_error_report(args.output_dir, exc, args.input_dir)
        payload = {
            "status": "error",
            "error": exc.to_dict(),
            "report": report_path,
        }
        sys.stderr.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - defensive process boundary
        error = internal_error(exc)
        report_path = write_error_report(args.output_dir, error, args.input_dir)
        payload = {
            "status": "error",
            "error": error.to_dict(),
            "report": report_path,
        }
        sys.stderr.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        return error.exit_code


if __name__ == "__main__":
    sys.exit(main())

