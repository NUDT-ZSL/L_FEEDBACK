"""Command line entry point for the EBU R128 dialogue normalizer."""
from __future__ import annotations

import argparse
import sys

from .processor import FatalConfigError, run_batch
from .resample import FFmpegError
from .report import build_report, write_report

EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_FATAL = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio-normalizer",
        description=(
            "Offline EBU R128 loudness normalization, true-peak limiting and "
            "SRT timeline correction for dialogue WAV tracks."
        ),
    )
    parser.add_argument("input_dir", help="directory containing WAV/SRT pairs")
    parser.add_argument("output_dir", help="directory for normalized outputs")
    parser.add_argument(
        "-j", "--parallelism", type=int, default=2,
        help="number of tracks processed in parallel (default: 2)",
    )
    parser.add_argument(
        "--report-name", default="processing_report.json",
        help="file name of the JSON report inside the output directory",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        batch_result = run_batch(
            args.input_dir, args.output_dir, args.parallelism
        )
    except (FatalConfigError, FFmpegError) as exc:
        sys.stderr.write(f"fatal configuration error: {exc}\n")
        return EXIT_FATAL

    report = build_report(
        batch_result, args.input_dir, args.output_dir, args.parallelism
    )
    report_path = f"{args.output_dir.rstrip('/')}/{args.report_name}"
    try:
        write_report(report, report_path)
    except OSError as exc:
        sys.stderr.write(f"fatal: cannot write report: {exc}\n")
        return EXIT_FATAL

    print(
        f"processed {report['total_tracks']} track(s): "
        f"{report['successful_tracks']} ok, {report['failed_tracks']} failed"
    )
    print(f"report written to {report_path}")
    if report["overall_status"] == "success":
        return EXIT_OK
    if report["overall_status"] == "partial":
        return EXIT_PARTIAL
    return EXIT_FATAL


if __name__ == "__main__":
    raise SystemExit(main())
