"""Command-line interface for dialogue conform delivery."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from . import __version__
from .errors import DialogueSyncError
from .fftools import FFmpegTools
from .model import load_project
from .pipeline import run_build
from .state import inspect_state


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dialogue-sync",
        description=(
            "Offline broadcast dialogue conform: resample, time-fit, mix, "
            "EBU R128 normalize, synchronize SRT, and package MKV."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _build_parser() -> argparse.ArgumentParser:
    parser = _common_parser()
    parser.add_argument("--timeline", required=True, type=Path, help="timeline JSON")
    parser.add_argument("--srt", type=Path, help="source SRT (overrides JSON)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="checkpoint directory (default: <output-dir>/.dialogue_sync_work)",
    )
    parser.add_argument(
        "--channels",
        choices=("mono", "stereo", "1", "2"),
        help="output channel layout (overrides JSON)",
    )
    parser.add_argument("--language", help="audio language tag, for example eng")
    parser.add_argument(
        "--subtitle-language", help="subtitle language tag, for example eng"
    )
    parser.add_argument(
        "--total-duration",
        help="target timeline duration HH:MM:SS.mmm (overrides JSON)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="discard an existing checkpoint and rebuild from the first entry",
    )
    parser.add_argument(
        "--simulate-abort-after",
        type=int,
        metavar="N",
        help=(
            "test hook: call os._exit(75) after N newly checkpointed entries, "
            "simulating forced termination"
        ),
    )
    parser.add_argument("--ffmpeg", help="ffmpeg executable (default: PATH)")
    parser.add_argument("--ffprobe", help="ffprobe executable (default: PATH)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def _state_parser() -> argparse.ArgumentParser:
    parser = _common_parser()
    parser.add_argument("--work-dir", required=True, type=Path)
    return parser


def _run_state(args: argparse.Namespace) -> int:
    try:
        result = inspect_state(args.work_dir.expanduser().resolve())
    except DialogueSyncError as exc:
        print(json.dumps({"valid": False, "error_code": exc.code, "message": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _run_build(args: argparse.Namespace) -> int:
    if args.simulate_abort_after is not None and args.simulate_abort_after < 1:
        print("--simulate-abort-after must be at least 1", file=sys.stderr)
        return 2
    timeline = args.timeline.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir
        else output_dir / ".dialogue_sync_work"
    )
    try:
        tools = FFmpegTools(args.ffmpeg, args.ffprobe)
        project = load_project(
            timeline,
            srt_path=args.srt,
            channels=args.channels,
            language=args.language,
            subtitle_language=args.subtitle_language,
            total_duration=args.total_duration,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        report = run_build(
            project,
            output_dir=output_dir,
            work_dir=work_dir,
            tools=tools,
            force=args.force,
            simulate_abort_after=args.simulate_abort_after,
            progress=not args.quiet,
        )
    except DialogueSyncError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 2
    except (OSError, ValueError) as exc:
        print(f"E_IO: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 2
    except Exception as exc:
        print(f"E_INTERNAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 3

    if not args.quiet:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "entries": report["summary"],
                    "loudness": {
                        "integrated_lufs": report["loudness"]["after"][
                            "integrated_lufs"
                        ],
                        "true_peak_dbtp": report["loudness"]["after"][
                            "true_peak_dbtp"
                        ],
                    },
                    "outputs": report["outputs"],
                    "work_dir": str(work_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] == "state":
        parser = _state_parser()
        return _run_state(parser.parse_args(raw_args[1:]))
    if raw_args and raw_args[0] == "build":
        raw_args = raw_args[1:]
    parser = _build_parser()
    return _run_build(parser.parse_args(raw_args))


if __name__ == "__main__":
    raise SystemExit(main())

