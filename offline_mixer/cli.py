"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
import traceback

from .config import load_config
from .engine import run_mix
from .errors import MixerError


def build_parser():
    parser = argparse.ArgumentParser(
        prog="offline-mixer",
        description="Deterministic offline multi-track WAV mixer with BS.1770-4 metering.",
    )
    parser.add_argument("config", help="path to the JSON mix configuration")
    parser.add_argument("--output", help="override output WAV path")
    parser.add_argument("--report", help="override JSON report path")
    parser.add_argument(
        "--workers",
        type=int,
        help="parallel track workers (default: logical CPU count)",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="force one worker; equivalent to --workers 1",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        help="streaming block size in frames",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress success summary")
    parser.add_argument("--debug", action="store_true", help="show tracebacks")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    workers = 1 if args.serial else args.workers
    try:
        config = load_config(
            args.config,
            output_override=args.output,
            report_override=args.report,
            workers_override=workers,
            chunk_frames_override=args.chunk_frames,
        )
        report = run_mix(config)
    except MixerError as exc:
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        sys.stderr.write(json.dumps(exc.as_json(), ensure_ascii=False) + "\n")
        return int(exc.exit_code)
    except Exception as exc:
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        payload = {
            "error": {
                "code": "E_INTERNAL",
                "message": "unexpected internal failure",
                "details": {"reason": str(exc)},
            }
        }
        sys.stderr.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return 1
    if not args.quiet:
        summary = {
            "output": report["output"]["path"],
            "report": str(config.output.report_path),
            "integrated_lufs": report["loudness"]["integrated_lufs_after_mix"],
            "true_peak_dbtp": report["loudness"]["true_peak_dbtp"],
            "workers": config.workers,
        }
        sys.stdout.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return 0
