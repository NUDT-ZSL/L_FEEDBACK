"""Command line entry points for sample generation and packaging."""

from __future__ import annotations

import argparse
import json
import sys

from .assets import FailureInjector
from .config import Config
from .pipeline import run_pipeline
from .samples import generate_sample_assets, write_demo_manifest


def _config_from_args(args: argparse.Namespace) -> Config:
    overrides = {
        "max_workers": args.max_workers,
        "target_loudness_db": args.target_loudness_db,
        "max_retries": args.max_retries,
        "crossfade_ms": args.crossfade_ms,
        "gain_min_db": args.gain_min_db,
        "gain_max_db": args.gain_max_db,
    }
    return Config(**{key: value for key, value in overrides.items() if value is not None})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiopack",
        description="Offline esports replay voice-over packaging tool",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate deterministic WAV samples + manifest")
    gen.add_argument("--assets-dir", required=True)
    gen.add_argument("--manifest", required=True)
    gen.add_argument("--seed", type=int, default=20261114)

    pack = sub.add_parser("pack", help="run the packaging pipeline")
    pack.add_argument("--manifest", required=True)
    pack.add_argument("--assets-dir", required=True)
    pack.add_argument("--work-dir", required=True)
    pack.add_argument("--output-dir", required=True)
    pack.add_argument("--max-workers", type=int, default=None)
    pack.add_argument("--target-loudness-db", type=float, default=None)
    pack.add_argument("--max-retries", type=int, default=None)
    pack.add_argument("--crossfade-ms", type=float, default=None)
    pack.add_argument("--gain-min-db", type=float, default=None)
    pack.add_argument("--gain-max-db", type=float, default=None)
    pack.add_argument("--stop-after-lang", default=None,
                      help="simulate interruption after a language block completes")
    pack.add_argument("--inject-missing", action="append", default=[])
    pack.add_argument("--inject-corrupt", action="append", default=[])
    pack.add_argument("--inject-transient", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        events = generate_sample_assets(args.assets_dir, seed=args.seed)
        write_demo_manifest(args.manifest, events)
        print(
            json.dumps(
                {
                    "generated_assets_dir": args.assets_dir,
                    "manifest": args.manifest,
                    "events": len(events),
                },
                indent=2,
            )
        )
        return 0

    config = _config_from_args(args)
    injector = FailureInjector(
        missing=set(args.inject_missing),
        corrupt=set(args.inject_corrupt),
        transient=set(args.inject_transient),
    )
    result = run_pipeline(
        manifest_path=args.manifest,
        assets_dir=args.assets_dir,
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        config=config,
        injector=injector,
        stop_after_lang=args.stop_after_lang,
    )
    print(json.dumps(result.report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result.ok else 2


if __name__ == "__main__":
    sys.exit(main())
