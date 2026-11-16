"""Command-line interface with stable 0/1/2 exit semantics."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional

from . import __version__
from .compression import CompressorConfig
from .errors import ParameterError, RenderRuntimeError
from .render import render_scene
from .scene import SceneSpec, load_scene


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ParameterError(message)


def _cli_path(raw: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(raw))
    if os.name != "nt":
        expanded = expanded.replace("\\", "/")
    return Path(expanded)


def _build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="spatial-render",
        description=(
            "Render a deterministic first-person spatial scene to a "
            "16-bit stereo WAV."
        ),
    )
    parser.add_argument("scene", help="scene JSON path")
    parser.add_argument("output", help="output .wav path")
    parser.add_argument(
        "--version", action="version", version="%(prog)s " + __version__
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="enable loudness normalization (default target: -16 LUFS)",
    )
    parser.add_argument(
        "--normalize-lufs",
        "--target-lufs",
        dest="normalize_lufs",
        type=float,
        default=None,
        metavar="LUFS",
        help="enable normalization and use this integrated loudness target",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="enable master dynamics compression with default parameters",
    )
    parser.add_argument("--threshold-db", type=float, default=None)
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--attack-ms", type=float, default=None)
    parser.add_argument("--release-ms", type=float, default=None)
    parser.add_argument("--knee-db", type=float, default=None)
    parser.add_argument("--makeup-db", type=float, default=None)
    return parser


def _compressor_from_args(
    args: argparse.Namespace,
) -> Optional[CompressorConfig]:
    tuning = (
        args.threshold_db,
        args.ratio,
        args.attack_ms,
        args.release_ms,
        args.knee_db,
        args.makeup_db,
    )
    if not args.compress and all(value is None for value in tuning):
        return None
    defaults = CompressorConfig()
    config = CompressorConfig(
        threshold_db=(
            defaults.threshold_db
            if args.threshold_db is None
            else args.threshold_db
        ),
        ratio=defaults.ratio if args.ratio is None else args.ratio,
        attack_ms=(
            defaults.attack_ms if args.attack_ms is None else args.attack_ms
        ),
        release_ms=(
            defaults.release_ms
            if args.release_ms is None
            else args.release_ms
        ),
        knee_db=defaults.knee_db if args.knee_db is None else args.knee_db,
        makeup_db=(
            defaults.makeup_db
            if args.makeup_db is None
            else args.makeup_db
        ),
    )
    config.validate()
    return config


def _validate_output_path(
    scene: SceneSpec, scene_argument: Path, output: Path
) -> None:
    if not str(output):
        raise ParameterError("output path must not be empty")
    if output.suffix.lower() != ".wav":
        raise ParameterError("output path must use the .wav extension")
    output_resolved = output.resolve(strict=False)
    if output_resolved == scene_argument.resolve(strict=False):
        raise ParameterError("output path must not overwrite the scene JSON")
    for source in scene.sources:
        if output_resolved == source.file_path.resolve(strict=False):
            raise ParameterError(
                "output path must not overwrite source track '%s'"
                % source.name
            )


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        scene_argument = _cli_path(args.scene)
        output_path = _cli_path(args.output)
        compressor = _compressor_from_args(args)
        normalize_lufs = args.normalize_lufs
        if args.normalize and normalize_lufs is None:
            normalize_lufs = -16.0
        scene = load_scene(scene_argument)
        _validate_output_path(scene, scene_argument, output_path)
        render_scene(
            scene,
            output_path,
            normalize_lufs=normalize_lufs,
            compressor_config=compressor,
            stderr=sys.stderr,
        )
        return 0
    except ParameterError as exc:
        sys.stderr.write("error: %s\n" % exc)
        sys.stderr.flush()
        return 1
    except RenderRuntimeError as exc:
        sys.stderr.write("runtime error: %s\n" % exc)
        sys.stderr.flush()
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("runtime error: interrupted\n")
        sys.stderr.flush()
        return 2
    except Exception as exc:
        if os.environ.get("SPATIAL_RENDER_DEBUG") == "1":
            traceback.print_exc()
        else:
            sys.stderr.write("runtime error: %s\n" % exc)
            sys.stderr.flush()
        return 2


if __name__ == "__main__":
    sys.exit(main())
