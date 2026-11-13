"""Command-line interface."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .model import PipelineConfig
from .runner import PipelineRunner


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def _tool_pair(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[Path, Path]:
    ffmpeg = Path(args.ffmpeg) if args.ffmpeg else None
    ffprobe = Path(args.ffprobe) if args.ffprobe else None
    if args.tools_dir:
        tools_dir = Path(args.tools_dir)
        ffmpeg = ffmpeg or tools_dir / "ffmpeg"
        ffprobe = ffprobe or tools_dir / "ffprobe"
    if ffmpeg is None or ffprobe is None:
        parser.error("provide --ffmpeg and --ffprobe (or --tools-dir)")
    return ffmpeg, ffprobe


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio_pipeline.py run",
        description="Mix deterministic show audio jobs into normalized stereo 48 kHz PCM WAV files.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        help="show root directory containing one manifest.json per show subdirectory",
    )
    parser.add_argument(
        "--root",
        dest="root_option",
        help="show root directory (alternative to the positional argument)",
    )
    parser.add_argument("--ffmpeg", help="path to the local ffmpeg executable")
    parser.add_argument("--ffprobe", help="path to the local ffprobe executable")
    parser.add_argument("--tools-dir", help="directory containing ffmpeg and ffprobe")
    parser.add_argument(
        "--jobs",
        "--concurrency",
        dest="jobs",
        type=_positive_int,
        default=2,
        help="maximum concurrent jobs (default: 2)",
    )
    parser.add_argument(
        "--work-dir",
        "--temp-dir",
        dest="work_dir",
        help="temp workspace (default: SHOW_ROOT/.audio-pipeline-tmp)",
    )
    parser.add_argument(
        "--job-temp-limit-mb",
        type=_positive_int,
        default=256,
        help="hard per-job temp workspace limit in MiB (default: 256)",
    )
    parser.add_argument(
        "--min-free-mb",
        type=_nonnegative_int,
        default=32,
        help="free-space reserve in MiB (default: 32)",
    )
    parser.add_argument("--journal", help="journal path (default: SHOW_ROOT/audio_pipeline_jobs.json)")
    parser.add_argument("--result", help="result JSON path (default: SHOW_ROOT/audio_pipeline_result.json)")
    parser.add_argument(
        "--ffmpeg-timeout-sec",
        type=_positive_int,
        default=600,
        help="timeout for each FFmpeg/ffprobe invocation (default: 600)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def run_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _run_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.root and args.root_option and Path(args.root).resolve() != Path(args.root_option).resolve():
        parser.error("positional root and --root point to different directories")
    root_value = args.root_option or args.root
    if not root_value:
        parser.error("provide the show root as a positional argument or with --root")
    ffmpeg, ffprobe = _tool_pair(args, parser)
    config = PipelineConfig(
        root=Path(root_value),
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        jobs=args.jobs,
        work_dir=Path(args.work_dir) if args.work_dir else None,
        job_temp_limit_mb=args.job_temp_limit_mb,
        min_free_mb=args.min_free_mb,
        journal_path=Path(args.journal) if args.journal else None,
        result_path=Path(args.result) if args.result else None,
        ffmpeg_timeout_sec=args.ffmpeg_timeout_sec,
    )
    stop_event = threading.Event()

    def handle_sigint(_signum: int, _frame: object) -> None:
        # Workers and their FFmpeg children are not in the foreground process
        # group, so the first Ctrl+C lets already-running jobs reach their
        # atomic commit points.
        stop_event.set()

    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handle_sigint)
    try:
        result = PipelineRunner(config, stop_event=stop_event).run()
    finally:
        signal.signal(signal.SIGINT, previous_handler)

    counts = result["counts"]
    summary = (
        f"audio-pipeline exit={result['exit_code']} total={counts['total']} "
        f"completed={counts['completed']} skipped={counts['skipped']} "
        f"failed={counts['failed']} interrupted={counts['interrupted']} "
        f"result={result['result_file']}"
    )
    print(summary, file=sys.stderr)
    return int(result["exit_code"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if arguments and arguments[0] == "self-test":
        from .selftest import self_test_main

        return self_test_main(arguments[1:])
    if arguments and arguments[0] == "run":
        arguments = arguments[1:]
    return run_main(arguments)
