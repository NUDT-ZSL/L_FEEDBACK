"""Command-line interface for generation, source synthesis, and replay checks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any, Sequence

from . import __version__
from .engine import generate_batch, verify_manifest
from .errors import SkillSFXError
from .synth import synthesize_sources


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _print_json(payload: dict[str, Any], *, stream) -> None:
    stream.write(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=True,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skillsfx",
        description=(
            "Deterministic batch generator for layered game skill sound effects."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show a Python traceback for unexpected internal failures",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="render all recipe variants for one skill level",
    )
    generate.add_argument("--recipe", required=True, help="recipe JSON path")
    generate.add_argument(
        "--level",
        required=True,
        type=int,
        help="skill level key from bus_by_level",
    )
    generate.add_argument(
        "--output",
        required=True,
        help="new or empty output directory for the atomic batch",
    )
    generate.add_argument(
        "--workers",
        type=int,
        default=1,
        help="worker thread count, 1..64 (default: 1)",
    )

    verify = subparsers.add_parser(
        "verify",
        help="reconstruct and verify an entire batch using only its manifest",
    )
    verify.add_argument("--manifest", required=True, help="manifest.json path")

    synth = subparsers.add_parser(
        "synth-sources",
        help="write the built-in deterministic demo source WAV library",
    )
    synth.add_argument(
        "--out",
        required=True,
        help="directory that will receive the synthesized source WAV files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            result = generate_batch(
                args.recipe,
                level=args.level,
                output_directory=args.output,
                workers=args.workers,
            )
        elif args.command == "verify":
            result = verify_manifest(args.manifest)
        elif args.command == "synth-sources":
            sources = synthesize_sources(args.out)
            result = {
                "status": "ok",
                "output_directory": str(Path(args.out).resolve()),
                "source_count": len(sources),
                "sources": sources,
            }
        else:
            raise SkillSFXError(
                "SFX-E9001",
                "unknown command",
                {"command": args.command},
            )
        _print_json(result, stream=sys.stdout)
        return 0
    except SkillSFXError as exc:
        _print_json(exc.to_dict(), stream=sys.stderr)
        return 2
    except KeyboardInterrupt:
        error = SkillSFXError(
            "SFX-E9001",
            "interrupted by user",
            {"interrupted": True},
        )
        _print_json(error.to_dict(), stream=sys.stderr)
        return 130
    except BaseException as exc:
        if args.debug:
            traceback.print_exc()
        error = SkillSFXError(
            "SFX-E9001",
            "unexpected internal failure",
            {
                "exception_type": type(exc).__name__,
                "reason": str(exc),
            },
        )
        _print_json(error.to_dict(), stream=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
