"""Command line interface (Windows / Linux / macOS).

Subcommands:

* ``simulate`` - generate deterministic impaired packets + ground truth,
* ``run``      - parse packets, jitter-buffer / PLC / mix, emit WAV + JSONL,
* ``verify``   - cross-check outputs against the simulator ground truth.

Exit code is 0 on success and non-zero (2 for bad arguments, 3 for IO /
configuration errors) with diagnostics written to stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence

from .constants import MAX_BUFFER_DEPTH, MAX_STREAMS, MIN_BUFFER_DEPTH, MIN_STREAMS
from .engine import AudioPipeline, RunSummary
from .protocol import iter_packet_files
from .simulator import (
    META_FILENAME,
    TRUTH_FILENAME,
    ImpairmentConfig,
    simulate,
)
from .verifier import verify_run

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_IO = 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio_sync",
        description="Deterministic audio frame sync, PLC and mixing service.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="generate impaired packet set")
    sim.add_argument("--output-dir", required=True, help="packet directory")
    sim.add_argument("--streams", type=int, default=3, choices=range(1, 5))
    sim.add_argument("--frames", type=int, default=250, help="frames per stream")
    sim.add_argument("--seed", type=int, default=20261114)
    sim.add_argument("--loss-rate", type=float, default=0.10)
    sim.add_argument("--reorder-rate", type=float, default=0.05)
    sim.add_argument("--jitter-mean-ms", type=float, default=20.0)
    sim.add_argument("--jitter-sigma-ms", type=float, default=15.0)
    sim.add_argument(
        "--jitter-distribution",
        choices=["normal", "uniform"],
        default="normal",
    )
    sim.add_argument("--max-jitter-ms", type=float, default=120.0)
    sim.add_argument(
        "--burst-loss", action="store_true", help="inject 4-frame burst loss"
    )
    sim.add_argument(
        "--inject-faults",
        action="store_true",
        help="append malformed packets for parser testing",
    )

    run = sub.add_parser("run", help="run the frame sync / mixing pipeline")
    run.add_argument("--input-dir", required=True, help="packet directory")
    run.add_argument("--wav", required=True, help="output WAV path")
    run.add_argument("--log", required=True, help="output JSONL log path")
    run.add_argument(
        "--buffer-depth",
        type=int,
        default=4,
        choices=range(MIN_BUFFER_DEPTH, MAX_BUFFER_DEPTH + 1),
        metavar=f"[{MIN_BUFFER_DEPTH}-{MAX_BUFFER_DEPTH}]",
    )
    run.add_argument(
        "--mixer-mode", choices=["equal_power", "linear"], default="equal_power"
    )
    run.add_argument(
        "--frames",
        type=int,
        default=None,
        help="override inferred total frame count",
    )

    verify = sub.add_parser("verify", help="verify outputs against truth")
    verify.add_argument("--input-dir", required=True)
    verify.add_argument("--wav", required=True)
    verify.add_argument("--log", required=True)
    return parser


def _load_meta(input_dir: str) -> Optional[dict]:
    meta_path = os.path.join(input_dir, META_FILENAME)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _print_summary(summary: RunSummary) -> None:
    print("=== audio_sync run summary ===")
    print(f"streams              : {summary.stream_count}")
    print(f"total_frames         : {summary.total_frames}")
    print(f"plc_frames           : {summary.plc_frames}")
    print(f"avg_buffer_water     : {summary.average_buffer_water}")
    print(f"max_delay_ms         : {summary.max_delay_ms}")
    print(f"backpressure_events  : {summary.backpressure_events}")
    print(f"wav                  : {summary.wav_path}")
    print(f"log                  : {summary.log_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "simulate":
            return _run_simulate(args)
        if args.command == "run":
            return _run_pipeline(args)
        return _run_verify(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[E_IO] {exc}", file=sys.stderr)
        return EXIT_IO


def _run_simulate(args: argparse.Namespace) -> int:
    config = ImpairmentConfig(
        seed=args.seed,
        loss_rate=args.loss_rate,
        reorder_rate=args.reorder_rate,
        jitter_mean_ms=args.jitter_mean_ms,
        jitter_sigma_ms=args.jitter_sigma_ms,
        jitter_distribution=args.jitter_distribution,
        max_jitter_ms=args.max_jitter_ms,
        burst_loss=args.burst_loss,
        inject_faults=args.inject_faults,
    )
    metadata = simulate(
        output_dir=args.output_dir,
        stream_count=args.streams,
        frame_count=args.frames,
        config=config,
    )
    total_lost = sum(len(v) for v in metadata["lost_frames"].values())
    print(
        f"simulated {metadata['stream_count']} streams x "
        f"{metadata['frame_count']} frames into {args.output_dir}"
    )
    print(f"lost_frames={total_lost} truth={TRUTH_FILENAME} meta={META_FILENAME}")
    return EXIT_OK


def _run_pipeline(args: argparse.Namespace) -> int:
    if not os.path.isdir(args.input_dir):
        print(f"[E_IO] input directory not found: {args.input_dir}", file=sys.stderr)
        return EXIT_IO
    if not MIN_BUFFER_DEPTH <= args.buffer_depth <= MAX_BUFFER_DEPTH:
        print(
            f"[E_CONFIG] buffer depth must be in "
            f"[{MIN_BUFFER_DEPTH},{MAX_BUFFER_DEPTH}]",
            file=sys.stderr,
        )
        return EXIT_USAGE

    packet_paths = list(iter_packet_files(args.input_dir))
    if not MIN_STREAMS <= len(packet_paths) <= MAX_STREAMS:
        print(
            f"[E_CONFIG] expected {MIN_STREAMS}-{MAX_STREAMS} stream_*.pkt "
            f"files, found {len(packet_paths)}",
            file=sys.stderr,
        )
        return EXIT_USAGE

    meta = _load_meta(args.input_dir)
    pipeline = AudioPipeline(
        packet_paths=packet_paths,
        wav_path=args.wav,
        log_path=args.log,
        buffer_depth=args.buffer_depth,
        mixer_mode=args.mixer_mode,
        frame_count=args.frames,
        meta=meta,
    )
    summary = pipeline.run()
    _print_summary(summary)
    return EXIT_OK


def _run_verify(args: argparse.Namespace) -> int:
    truth_path = os.path.join(args.input_dir, TRUTH_FILENAME)
    if not os.path.exists(truth_path):
        print(f"[E_IO] ground truth missing: {truth_path}", file=sys.stderr)
        return EXIT_IO
    report = verify_run(
        truth_path=truth_path,
        log_path=args.log,
        wav_path=args.wav,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return EXIT_OK if report["ok"] else EXIT_IO
