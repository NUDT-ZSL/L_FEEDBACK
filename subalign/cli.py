"""Command line interface for the subtitle alignment tool."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .pipeline import VIDEO_EXTENSIONS, run
from .subtitles import SubtitleError, format_for_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subalign",
        description="Detect and correct subtitle timeline drift against a reference.",
    )
    parser.add_argument(
        "-r",
        "--reference",
        required=True,
        help="reference subtitle (.srt/.ass) or video file for packet timestamps",
    )
    parser.add_argument(
        "-t",
        "--targets",
        nargs="+",
        required=True,
        help="one or more target subtitle files to align",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="directory for aligned files; defaults to each input file's directory",
    )
    parser.add_argument(
        "--report",
        default="subalign_report.json",
        help="path of the JSON correction report (default: subalign_report.json)",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> list[dict]:
    errors: list[dict] = []
    if not os.path.exists(args.reference):
        errors.append(
            {
                "path": args.reference,
                "error_type": "FileNotFound",
                "message": "reference file does not exist",
            }
        )
    else:
        lower = args.reference.lower()
        is_video = any(lower.endswith(ext) for ext in VIDEO_EXTENSIONS)
        if not is_video:
            try:
                format_for_path(args.reference)
            except SubtitleError as error:
                errors.append(error.to_dict())

    for target in args.targets:
        if not os.path.exists(target):
            errors.append(
                {
                    "path": target,
                    "error_type": "FileNotFound",
                    "message": "target file does not exist",
                }
            )
            continue
        try:
            format_for_path(target)
        except SubtitleError as error:
            errors.append(error.to_dict())

    if args.output_dir and os.path.splitext(args.output_dir)[1]:
        if os.path.exists(args.output_dir) and not os.path.isdir(args.output_dir):
            errors.append(
                {
                    "path": args.output_dir,
                    "error_type": "InvalidOutputDir",
                    "message": "output directory path exists and is a file",
                }
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    validation_errors = validate_arguments(args)
    if validation_errors:
        payload = {"status": "error", "errors": validation_errors}
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    try:
        result = run(
            reference_path=args.reference,
            target_paths=args.targets,
            output_dir=args.output_dir,
            report_path=args.report,
        )
    except SubtitleError as error:
        payload = {"status": "error", "errors": [error.to_dict()]}
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "status": "ok",
                "aligned_files": result.output_paths,
                "report": os.path.abspath(args.report),
                "error_count": len(result.errors),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
