"""Command line interface for the lip-sync alignment tool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .config import load_config
from .pipeline import run_pipeline
from .samples import generate_sample_tree


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lipsync-tool",
        description="Align localized dub audio with cutscene lip events and remux MP4.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser(
        "make-samples", help="deterministically build local sample inputs"
    )
    sample_parser.add_argument("--output-dir", required=True, help="sample tree output directory")

    process_parser = subparsers.add_parser(
        "process", help="process dub languages and emit aligned MP4/JSON outputs"
    )
    process_parser.add_argument("--metadata", required=True, help="animation metadata JSON path")
    process_parser.add_argument("--audio-dir", required=True, help="directory containing dub WAV files")
    process_parser.add_argument("--output-dir", required=True, help="directory for generated outputs")
    process_parser.add_argument(
        "--languages",
        required=True,
        help="comma separated language codes, e.g. en,ja,zh",
    )
    process_parser.add_argument(
        "--video",
        default=None,
        help="silent source MP4 path (defaults to ../video/cutscene_silent.mp4)",
    )
    process_parser.add_argument(
        "--config",
        default=None,
        help="optional JSON config for stretch bounds, tolerance and penalties",
    )
    return parser


def _parse_languages(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "make-samples":
        paths = generate_sample_tree(Path(args.output_dir))
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
        return 0

    languages = _parse_languages(args.languages)
    if not languages:
        parser.error("at least one language code must be supplied via --languages")

    config, config_notes = load_config(args.config)
    report = run_pipeline(
        metadata_path=Path(args.metadata),
        audio_dir=Path(args.audio_dir),
        output_dir=Path(args.output_dir),
        languages=languages,
        config=config,
        config_notes=config_notes,
        video_path=Path(args.video) if args.video else None,
    )

    print(f"processed {len(languages)} language(s); report written to {args.output_dir}/processing_report.json")
    for entry in report["language_reports"]:
        print(
            f"  - {entry['language']}: status={entry['status']}, "
            f"events={entry['event_count']}, stretch={entry['stretch_factor']}, "
            f"mean|offset|={entry['alignment_stats']['mean_offset_ms']:.2f} ms, "
            f"max|offset|={entry['alignment_stats']['max_offset_ms']:.2f} ms"
        )
    if report["failed_segments"]:
        print(f"  failed segments: {', '.join(report['failed_segments'])}", file=sys.stderr)
    if report["mux_fallback_segments"]:
        print(
            f"  mux fallback segments: {', '.join(report['mux_fallback_segments'])}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
