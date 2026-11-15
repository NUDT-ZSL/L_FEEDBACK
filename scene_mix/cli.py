from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .batch import run_batch
from .pipeline import mix_group
from .synth import generate_group


def _seed_list(value: str) -> List[int]:
    seeds: List[int] = []
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scene-mix",
        description=(
            "Deterministic multi-camera WAV synthesis, integer-sample alignment, "
            "BS.1770 scene gain and stereo delivery mix."
        ),
    )
    parser.add_argument("--version", action="version", version="scene-mix 1.0.0")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="generate deterministic WAV groups")
    generate.add_argument(
        "--seeds",
        "--seed",
        required=True,
        type=_seed_list,
        help="comma-separated integer seeds, e.g. 101,202,303",
    )
    generate.add_argument(
        "--channels",
        "--tracks",
        dest="channels",
        type=int,
        default=4,
        help="number of camera WAVs per seed (default: 4)",
    )
    generate.add_argument(
        "--out", "--output", required=True, help="input root directory"
    )
    generate.add_argument(
        "--silent-track",
        "--silent",
        type=int,
        default=None,
        help="1-based camera number to render as digital silence",
    )
    generate.add_argument(
        "--clip-track",
        "--clip",
        type=int,
        default=None,
        help="1-based camera number to render with hard clipping",
    )
    generate.add_argument(
        "--duration-seconds",
        type=float,
        default=10.0,
        help="WAV duration (default: 10)",
    )

    mix = sub.add_parser("mix", help="process one timeline group")
    mix.add_argument("--timeline", default=None, help="path to timeline.json")
    mix.add_argument(
        "--input-dir",
        default=None,
        help="group directory containing timeline.json",
    )
    mix.add_argument(
        "--output-dir", "--output", required=True, help="output directory"
    )

    batch = sub.add_parser("batch", help="process multiple groups concurrently")
    batch.add_argument(
        "--input-root", "--inputs", required=True, help="generated input root"
    )
    batch.add_argument(
        "--output-root", "--outputs", required=True, help="delivery root"
    )
    batch.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="concurrent groups, default 4, capped at logical CPU count",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            output_root = Path(args.out)
            for seed in args.seeds:
                destination = output_root / f"seed_{seed}"
                timeline = generate_group(
                    seed=seed,
                    output_dir=destination,
                    channels=args.channels,
                    silent_track=args.silent_track,
                    clipped_track=args.clip_track,
                    duration_seconds=args.duration_seconds,
                )
                print(
                    f"generated seed={seed} tracks={len(timeline['tracks'])} "
                    f"dir={destination}"
                )
            return 0

        if args.command == "mix":
            if args.timeline:
                timeline_path = Path(args.timeline)
            elif args.input_dir:
                timeline_path = Path(args.input_dir) / "timeline.json"
            else:
                parser.error("mix requires --timeline or --input-dir")
            report = mix_group(timeline_path, Path(args.output_dir))
            print(
                json.dumps(
                    {
                        "output": str(Path(args.output_dir) / "mix.wav"),
                        "report": str(Path(args.output_dir) / "report.json"),
                        "warnings": report["warnings"],
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        summary = run_batch(
            Path(args.input_root),
            Path(args.output_root),
            jobs_requested=args.jobs,
        )
        for group in summary["groups"]:
            detail = group.get("error", "")
            suffix = f" error={detail}" if detail else ""
            print(
                f"{group['status']:<18} retries={group['retry_count']} "
                f"{group['name']}{suffix}"
            )
        print(
            f"summary success={summary['success']} failed={summary['failed']} "
            f"jobs={summary['jobs_used']}/{summary['cpu_logical_cores']}"
        )
        return 0 if summary["failed"] == 0 else 1
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
