"""Command-line interface with machine-readable errors."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Sequence

from . import __version__
from .errors import EXIT_INTERNAL, BusinessError
from .pipeline import run_localization


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise BusinessError(
            code="INVALID_ARGUMENTS",
            message=message,
            exit_code=2,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="dialogue-localizer",
        description=(
            "Replace a cut-scene dialogue timeline with localized, loudness-"
            "normalized WAV audio."
        ),
    )
    parser.add_argument("--manifest", required=True, help="JSON timeline manifest")
    parser.add_argument(
        "--target-language", required=True, help="language key under targets"
    )
    parser.add_argument("--output", required=True, help="output WAV path")
    parser.add_argument("--report", required=True, help="output JSON report path")
    parser.add_argument(
        "--offset-ms",
        type=int,
        default=0,
        help="global target timeline offset in milliseconds",
    )
    parser.add_argument(
        "--crossfade-ms",
        type=float,
        default=10.0,
        help="maximum equal-power crossfade length (0..10 ms)",
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=-23.0,
        help="per-segment integrated loudness target (default: -23)",
    )
    parser.add_argument(
        "--peak-dbfs",
        type=float,
        default=-1.0,
        help="per-segment peak ceiling (default: -1)",
    )
    parser.add_argument("--ffmpeg", help="explicit ffmpeg executable")
    parser.add_argument("--ffprobe", help="explicit ffprobe executable")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def _write_json(stream: object, payload: dict[str, object]) -> None:
    # ASCII escaping avoids Windows console code-page failures on localized
    # paths and messages while remaining valid UTF-8/JSON at the byte level.
    text = json.dumps(payload, ensure_ascii=True, allow_nan=False)
    print(text, file=stream)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if not -60.0 <= args.target_lufs <= 0.0:
            raise BusinessError(
                code="INVALID_ARGUMENTS",
                message="--target-lufs must be between -60 and 0",
                exit_code=2,
            )
        if not -20.0 <= args.peak_dbfs < 0.0:
            raise BusinessError(
                code="INVALID_ARGUMENTS",
                message="--peak-dbfs must be between -20 (inclusive) and 0",
                exit_code=2,
            )
        report = run_localization(
            manifest_path=args.manifest,
            target_language=args.target_language,
            output_path=args.output,
            report_path=args.report,
            offset_ms=args.offset_ms,
            target_lufs=args.target_lufs,
            peak_dbfs=args.peak_dbfs,
            crossfade_ms=args.crossfade_ms,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
        )
        _write_json(
            sys.stdout,
            {
                "status": "ok",
                "code": "OK",
                "exit_code": 0,
                "output": report["output"]["path"],
                "report": report["report_path"],
                "target_language": report["target_language"],
                "warnings": len(report["warnings"]),
            },
        )
        return 0
    except BusinessError as exc:
        _write_json(sys.stderr, exc.as_dict())
        return exc.exit_code
    except KeyboardInterrupt:
        error = BusinessError(
            code="INTERRUPTED",
            message="processing was interrupted",
            exit_code=130,
        )
        _write_json(sys.stderr, error.as_dict())
        return error.exit_code
    except Exception as exc:  # pragma: no cover - defensive boundary
        error = BusinessError(
            code="INTERNAL_ERROR",
            message=f"{type(exc).__name__}: {exc}",
            exit_code=EXIT_INTERNAL,
            details=(
                {"traceback": traceback.format_exc()[-8000:]}
                if os.environ.get("DIALOGUE_LOCALIZER_DEBUG")
                else {}
            ),
        )
        _write_json(sys.stderr, error.as_dict())
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

