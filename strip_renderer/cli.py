from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Sequence

from .errors import EXIT_ASSET_OR_ENVIRONMENT, RenderError
from .output import prepare_output_directory, write_error_report, write_render_outputs
from .render import render_scene
from .scene import Scene, load_scene
from .texture import load_textures


def _error_candidates(
    error: RenderError,
    scene: Scene | None,
    scene_argument: Path,
) -> list[Path]:
    candidates: list[Path] = []
    if error.report_dir is not None:
        candidates.append(error.report_dir)
    if scene is not None:
        candidates.append(scene.output.directory)
        candidates.append(scene.scene_path.parent / "render_error_output")
    else:
        candidates.append(scene_argument.parent / "render_error_output")
    candidates.append(Path.cwd())
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve())
        except (OSError, ValueError):
            key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically render a textured sphere into four layered PNGs."
        )
    )
    parser.add_argument("--scene", required=True, help="path to the scene JSON")
    parser.add_argument(
        "--output-dir",
        help="override output.directory from the scene",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    scene_argument = Path(args.scene)
    output_override = (
        Path(args.output_dir).resolve() if args.output_dir is not None else None
    )
    scene: Scene | None = None
    try:
        scene = load_scene(scene_argument, output_override=output_override)
        prepare_output_directory(scene)
        textures = load_textures(scene.material)
        result = render_scene(scene, textures)
        report_path = write_render_outputs(scene, result)
        print(
            f"rendered {scene.output.width}x{scene.output.height} "
            f"filter={scene.output.texture_filter} "
            f"hits={result.hit_pixels} report={report_path}"
        )
        return 0
    except RenderError as error:
        print(f"{error.code}: {error.message}", file=sys.stderr)
        report_path = write_error_report(
            error,
            _error_candidates(error, scene, scene_argument),
        )
        if report_path is not None:
            print(f"error report: {report_path}", file=sys.stderr)
        return error.exit_code
    except Exception as exc:  # pragma: no cover - last-resort CLI guard
        error = RenderError(
            code="E_INTERNAL",
            message=f"unexpected renderer failure: {type(exc).__name__}: {exc}",
            exit_code=EXIT_ASSET_OR_ENVIRONMENT,
        )
        print(f"{error.code}: {error.message}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        write_error_report(
            error,
            _error_candidates(error, scene, scene_argument),
        )
        return error.exit_code
