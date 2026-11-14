from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .core import (
    AudioMixerError,
    EncodedOutputError,
    ProjectError,
    load_project,
    render_project,
)


EXIT_PROJECT_NOT_FOUND = 2
EXIT_INVALID_JSON = 3
EXIT_INVALID_PROJECT = 4
EXIT_FFMPEG_NOT_FOUND = 5
EXIT_ENCODE_FAILED = 6
EXIT_IO_ERROR = 7


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mix a deterministic audio timeline and encode it with FFmpeg."
    )
    parser.add_argument("project", type=Path, help="path to the project JSON file")
    parser.add_argument("output_dir", type=Path, help="directory for audio and report")
    parser.add_argument(
        "--ffmpeg",
        default=None,
        help="FFmpeg executable path (defaults to FFMPEG_BINARY or PATH)",
    )
    return parser


def _exit_code_for(exc: AudioMixerError) -> int:
    if exc.code == "PROJECT_NOT_FOUND":
        return EXIT_PROJECT_NOT_FOUND
    if exc.code == "INVALID_JSON":
        return EXIT_INVALID_JSON
    if exc.code == "FFMPEG_NOT_FOUND":
        return EXIT_FFMPEG_NOT_FOUND
    if exc.code == "FFMPEG_ENCODE_FAILED":
        return EXIT_ENCODE_FAILED
    if exc.code in {"OUTPUT_IO_ERROR", "OUTPUT_DECODE_ERROR"}:
        return EXIT_IO_ERROR
    return EXIT_INVALID_PROJECT


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        project = load_project(args.project)
        output_path, report_path, report = render_project(
            project, args.output_dir, ffmpeg_path=args.ffmpeg
        )
    except EncodedOutputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if exc.stderr.strip():
            print(exc.stderr.strip(), file=sys.stderr)
        return EXIT_ENCODE_FAILED
    except ProjectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _exit_code_for(exc)
    except AudioMixerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _exit_code_for(exc)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_IO_ERROR

    skipped = sum(1 for clip in report["clips"] if clip["status"] == "skipped")
    print(f"output: {output_path}")
    print(f"report: {report_path}")
    print(
        f"duration: {report['duration_sec']:.3f}s, "
        f"peak: {report['peak_dbfs']:.3f} dBFS, "
        f"skipped clips: {skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

