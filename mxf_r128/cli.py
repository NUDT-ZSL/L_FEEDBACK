"""Windows-friendly command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .constants import (
    E_ARGUMENT,
    E_DIR_NOT_FOUND,
    E_FFMPEG_NOT_FOUND,
    E_REPORT_WRITE,
    E_UNEXPECTED,
)
from .pipeline import PipelineConfig, run_batch
from .probe import resolve_tool
from .report import build_report, write_report_atomic


class UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _Parser(
        prog="python -m mxf_r128",
        description="Batch MXF to H.264 MP4 with EBU R128 normalization and QC.",
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-workers", default=1, type=int)
    return parser.parse_args(argv)


def _stderr(code: str) -> None:
    print(code, file=sys.stderr)


def _prepare_paths(
    input_dir: Path, output_dir: Path, report: Path
) -> int | None:
    if not input_dir.is_dir():
        _stderr(E_DIR_NOT_FOUND)
        return 2
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        _stderr(E_DIR_NOT_FOUND)
        return 2
    if not output_dir.is_dir():
        _stderr(E_DIR_NOT_FOUND)
        return 2
    try:
        report.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        _stderr(E_REPORT_WRITE)
        return 3
    return None


def _run(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
    except UsageError:
        _stderr(E_ARGUMENT)
        return 2

    if args.max_workers < 1:
        _stderr(E_ARGUMENT)
        return 2
    path_error = _prepare_paths(args.input_dir, args.output_dir, args.report)
    if path_error is not None:
        return path_error

    ffmpeg = resolve_tool("FFMPEG_BIN", "ffmpeg")
    ffprobe = resolve_tool("FFPROBE_BIN", "ffprobe")
    if ffmpeg is None or ffprobe is None:
        _stderr(E_FFMPEG_NOT_FOUND)
        return 2

    config = PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        report=args.report,
        max_workers=args.max_workers,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
    )
    records = run_batch(config)
    report = build_report(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        records=records,
    )
    try:
        write_report_atomic(args.report, report)
    except OSError:
        _stderr(E_REPORT_WRITE)
        return 3

    failed_codes = report["summary"]["error_codes"]
    if failed_codes:
        print(" ".join(str(code) for code in failed_codes), file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(argv)
    except Exception:
        _stderr(E_UNEXPECTED)
        return 3
