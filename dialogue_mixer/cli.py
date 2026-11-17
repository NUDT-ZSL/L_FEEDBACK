"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .engine import run_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dialogue-mixer",
        description="Sample-accurate dialogue ducking and time-stretch mixer.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    mix_parser = subparsers.add_parser("mix", help="render a project JSON file")
    mix_parser.add_argument("project", type=Path, help="path to project JSON")
    mix_parser.add_argument("--audio", type=Path, help="override mixed WAV path")
    mix_parser.add_argument("--index", type=Path, help="override track index path")
    mix_parser.add_argument("--log", type=Path, help="override JSON log path")
    return parser


def _summary_from_log(log_path: Path) -> str:
    try:
        payload = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    normalization = payload.get("normalization", {})
    counts = payload.get("counts", {})
    peak = payload.get("outputs", {}).get("output_peak_dbfs")
    factor = normalization.get("normalization_factor")
    alignment = (
        payload.get("outputs", {}).get("index_max_alignment_error_samples")
    )
    return (
        "segments={segments} duck_events={ducks} overlaps={overlaps} "
        "peak_dbfs={peak} norm_factor={factor} alignment_error={alignment}"
    ).format(
        segments=counts.get("rendered_segments"),
        ducks=counts.get("duck_events"),
        overlaps=counts.get("overlap_events"),
        peak=peak,
        factor=factor,
        alignment=alignment,
    )


def main(argv: Optional[List[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] not in ("mix", "-h", "--help"):
        arguments.insert(0, "mix")
    parser = build_parser()
    args = parser.parse_args(arguments)
    result = run_project(
        args.project,
        audio_override=args.audio,
        index_override=args.index,
        log_override=args.log,
    )
    print(
        "status={0} exit_code={1}".format(result.status, result.exit_code)
    )
    if result.audio_path is not None:
        print("audio={0}".format(result.audio_path))
    if result.index_path is not None:
        print("index={0}".format(result.index_path))
    print("log={0}".format(result.log_path))
    summary = _summary_from_log(result.log_path)
    if summary:
        print(summary)
    if result.error_message:
        print(result.error_message, file=sys.stderr)
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())

