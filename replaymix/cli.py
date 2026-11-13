"""Command-line entry points for local sample generation and replay mixing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import EventTableError
from .generator import generate_scenario
from .pipeline import MixConfig, process_session


def _generate(args: argparse.Namespace) -> int:
    result = generate_scenario(
        args.out,
        session_ms=args.session_ms,
        seed=args.seed,
        include_faults=args.faults,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _mix(args: argparse.Namespace) -> int:
    config = MixConfig(
        target_peak_dbfs=args.target_peak_dbfs,
        max_gain_db=args.max_gain_db,
        concurrency=args.concurrency,
        checkpoint_version=args.checkpoint_version,
        checkpoint_enabled=not args.no_checkpoint,
    )
    try:
        result = process_session(
            args.events,
            args.out,
            config=config,
        )
    except EventTableError as exc:
        print(json.dumps({"error": exc.code, "message": str(exc)}), file=sys.stderr)
        return 2
    summary = {
        "report": str(result.report_path),
        "wav": str(result.wav_path),
        "checkpoint": (
            str(result.checkpoint_path) if result.checkpoint_path is not None else None
        ),
        "mix_frames": result.report["mix"]["total_frames"],
        "sample_rate_hz": result.report["output_wav"]["sample_rate_hz"],
        "channels": result.report["output_wav"]["channels"],
        "checkpoint_reused": result.report["checkpoint_reused"],
        "reused_track_ids": result.report["checkpoint"]["reused_track_ids"],
        "error_codes": [item["code"] for item in result.report["errors"]],
        "effective_concurrency": result.effective_concurrency,
        "max_observed_concurrency": result.max_observed_concurrency,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _demo(args: argparse.Namespace) -> int:
    sample_dir = Path(args.out) / "samples"
    generate_scenario(
        sample_dir,
        session_ms=args.session_ms,
        seed=args.seed,
        include_faults=args.faults,
    )
    config = MixConfig(concurrency=args.concurrency)
    result = process_session(
        sample_dir / "event_table.json",
        Path(args.out),
        config=config,
    )
    print(
        json.dumps(
            {
                "event_table": str(sample_dir / "event_table.json"),
                "report": str(result.report_path),
                "wav": str(result.wav_path),
                "mix_frames": result.report["mix"]["total_frames"],
                "error_codes": [item["code"] for item in result.report["errors"]],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="replaymix",
        description="Offline esports voice-comms replay mixer",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic sample WAVs")
    generate.add_argument("--out", required=True, help="sample output directory")
    generate.add_argument("--session-ms", type=int, default=12_000)
    generate.add_argument("--seed", type=int, default=20_261_114)
    generate.add_argument(
        "--faults",
        action="store_true",
        help="add deliberate rate/frame/overrun/missing faults",
    )
    generate.set_defaults(func=_generate)

    mix = subparsers.add_parser("mix", help="mix tracks into a replay WAV and report")
    mix.add_argument("--events", required=True, help="event table JSON")
    mix.add_argument("--out", required=True, help="output directory")
    mix.add_argument("--concurrency", type=int, default=2)
    mix.add_argument("--target-peak-dbfs", type=float, default=-1.0)
    mix.add_argument("--max-gain-db", type=float, default=20.0)
    mix.add_argument("--checkpoint-version", type=int, default=1)
    mix.add_argument("--no-checkpoint", action="store_true")
    mix.set_defaults(func=_mix)

    demo = subparsers.add_parser("demo", help="generate a local sample and mix it")
    demo.add_argument("--out", required=True)
    demo.add_argument("--session-ms", type=int, default=12_000)
    demo.add_argument("--seed", type=int, default=20_261_114)
    demo.add_argument("--concurrency", type=int, default=2)
    demo.add_argument("--faults", action="store_true")
    demo.set_defaults(func=_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, RuntimeError, OSError) as exc:
        print(
            json.dumps(
                {"error": type(exc).__name__, "message": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
