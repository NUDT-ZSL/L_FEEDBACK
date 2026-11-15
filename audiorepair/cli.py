"""Command line interface for single-file and batch repair."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from .batch import load_batch, run_batch
from .errors import EXIT_CODES, PROCESSING_FAILED, RepairError
from .pipeline import run_job


def _error_json(exc: RepairError) -> str:
    return json.dumps(
        exc.as_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiorepair",
        description=(
            "Deterministic EBU R128 segment normalisation, linear crossfade "
            "and anchored WSOLA repair."
        ),
    )
    parser.add_argument(
        "input", nargs="?", help="input 16-bit PCM WAV (44.1/48 kHz, mono/stereo)"
    )
    parser.add_argument("timeline", nargs="?", help="timeline JSON")
    parser.add_argument("-o", "--output", help="repaired WAV path")
    parser.add_argument("-r", "--report", help="mapping report JSON path")
    parser.add_argument("--batch", help="batch manifest JSON")
    parser.add_argument(
        "--output-dir",
        help="batch output directory (overrides per-job output paths)",
    )
    parser.add_argument(
        "--batch-report",
        help="batch summary path (default: alongside the batch manifest)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="worker threads for --batch (default: 1)",
    )
    parser.add_argument(
        "--version", action="version", version="audiorepair 1.0.0"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.batch:
            if args.input or args.timeline:
                raise RepairError(
                    PROCESSING_FAILED,
                    "positional input/timeline cannot be combined with --batch",
                )
            jobs = load_batch(args.batch, args.output_dir)
            summary_path = args.batch_report or os.path.join(
                os.path.dirname(os.path.abspath(args.batch)),
                "batch_report.json",
            )
            summary = run_batch(jobs, args.jobs, summary_path)
            print(
                json.dumps(
                    summary,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                )
            )
            if summary["summary"]["failed"] == 0:
                return 0
            if summary["summary"]["succeeded"] == 0:
                return 2
            return 1

        if not args.input or not args.timeline:
            raise RepairError(
                PROCESSING_FAILED,
                "provide INPUT TIMELINE or --batch",
            )
        output_path = args.output
        if output_path is None:
            stem = os.path.splitext(os.path.basename(args.input))[0]
            output_path = os.path.join("out", stem + ".repaired.wav")
        report_path = args.report
        if report_path is None:
            stem = os.path.splitext(output_path)[0]
            report_path = stem + ".report.json"
        result = run_job(
            args.input, args.timeline, output_path, report_path
        )
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0
    except RepairError as exc:
        print(_error_json(exc), file=sys.stderr)
        return EXIT_CODES.get(exc.code, EXIT_CODES[PROCESSING_FAILED])
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        wrapped = RepairError(
            PROCESSING_FAILED, "{}: {}".format(type(exc).__name__, exc)
        )
        print(_error_json(wrapped), file=sys.stderr)
        return EXIT_CODES[PROCESSING_FAILED]

