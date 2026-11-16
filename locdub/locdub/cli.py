"""Command line entry point.

Example (Windows cmd)::

    python -m locdub --timeline timeline.json --video source.mp4 ^
        --lang en=dubs\\en --lang ja=dubs\\ja --out output ^
        --workers 2 --gain-db -3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

from . import ffmpeg_utils, processor, report, timeline
from .constants import EXIT_FATAL
from .logging_utils import RunLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="locdub",
        description="Offline game-localisation dub adaptation and muxing tool.",
    )
    parser.add_argument("--timeline", required=True, type=Path,
                        help="Path to the timeline JSON file.")
    parser.add_argument("--video", required=True, type=Path,
                        help="Source video file.")
    parser.add_argument(
        "--lang", action="append", default=[], metavar="NAME=DIR",
        help="Language name and dub directory; repeat for each language.",
    )
    parser.add_argument("--out", required=True, type=Path,
                        help="Output directory for MP4 files and reports.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Maximum number of languages processed in parallel.")
    parser.add_argument("--gain-db", type=float, default=0.0,
                        help="Dub gain in dB applied during mixing.")
    parser.add_argument("--sample-rate", type=int, default=48000,
                        help="Mix-down sample rate (default 48000).")
    parser.add_argument("--channels", type=int, choices=(1, 2), default=2,
                        help="Mix-down channel count (default 2).")
    parser.add_argument("--max-segment-seconds", type=float, default=600.0,
                        help="Per-segment memory guard in seconds.")
    parser.add_argument("--block-seconds", type=float, default=60.0,
                        help="Streaming block size in seconds.")
    parser.add_argument("--report-name", default="report.json",
                        help="Report file name written under --out.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output (errors still logged).")
    return parser


def parse_languages(values: List[str]) -> Dict[str, Path]:
    languages: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--lang expects NAME=DIR, got '{value}'")
        name, directory = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"empty language name in '{value}'")
        if name in languages:
            raise ValueError(f"duplicate language '{name}'")
        languages[name] = Path(directory.strip())
    if not languages:
        raise ValueError("at least one --lang NAME=DIR is required")
    return languages


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.out / "error.log", quiet=args.quiet)

    try:
        languages = parse_languages(args.lang)
    except ValueError as exc:
        logger.error(str(exc))
        return EXIT_FATAL

    # ---- unrecoverable stage: timeline parse + video probe ----------------
    try:
        segments, metadata = timeline.load_timeline(args.timeline)
        probe = ffmpeg_utils.probe_video(args.video)
    except timeline.TimelineFatalError as exc:
        logger.error(f"fatal timeline error: {exc}")
        return EXIT_FATAL
    except ffmpeg_utils.FFmpegError as exc:
        logger.error(f"fatal video error: {exc}")
        return EXIT_FATAL

    video_duration_ms = metadata.get("video_duration_ms") or probe["duration_ms"]
    validation = timeline.validate_segments(segments, video_duration_ms)
    for segment_id, reason in validation.invalid:
        logger.error("invalid timeline segment skipped",
                     segment_id=segment_id, reason=reason)
    valid_segments = validation.valid
    skipped_ids: Dict[str, str] = {
        segment_id: reason for segment_id, reason in validation.invalid
    }

    if not valid_segments:
        logger.error("no valid segments remain after validation")
        return EXIT_FATAL

    config = processor.ProcessConfig(
        video=args.video,
        output_dir=args.out,
        sample_rate=args.sample_rate,
        channels=args.channels,
        gain_db=args.gain_db,
        max_segment_seconds=args.max_segment_seconds,
        block_seconds=args.block_seconds,
    )

    # ---- recoverable stage: per-language processing -----------------------
    results = processor.run_all(
        languages, valid_segments, skipped_ids, config, logger,
        workers=max(1, args.workers),
    )

    document = report.build_report(
        results,
        timeline=args.timeline,
        video=args.video,
        gain_db=args.gain_db,
        workers=max(1, args.workers),
        invalid_segments=[
            {"segment_id": segment_id, "reason": reason}
            for segment_id, reason in validation.invalid
        ],
    )
    report_path = args.out / args.report_name
    report.write_report(document, report_path)
    code = report.exit_code_for(document)
    logger.info(f"report written: {report_path}")
    logger.info(f"exit code: {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
