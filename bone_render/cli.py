from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from . import __version__
from .errors import SceneError, emit_error
from .export import export_scene
from .scene import load_scene


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise SceneError("E_USAGE", "arguments", message)


def build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(
        prog="python renderer.py",
        description="Deterministic offline skinned-mesh renderer with cascaded shadows.",
    )
    parser.add_argument(
        "--scene", "--input", dest="scene", required=True, help="input scene JSON path"
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        required=True,
        help="dedicated output directory",
    )
    parser.add_argument(
        "--incremental",
        "--append",
        action="store_true",
        dest="incremental",
        help="append frames newer than the verified exported prefix",
    )
    parser.add_argument(
        "--threads",
        "--jobs",
        dest="threads",
        type=int,
        default=1,
        help="number of frame worker threads (1..CPU count, default: 1)",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def _validate_threads(threads: int) -> int:
    cpu_count = os.cpu_count() or 1
    if threads < 1 or threads > cpu_count:
        raise SceneError(
            "E_THREADS",
            "threads",
            f"thread count must be in [1, {cpu_count}]",
        )
    return threads


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        threads = _validate_threads(args.threads)
        scene = load_scene(Path(args.scene))
        result = export_scene(
            scene,
            Path(args.output),
            incremental=bool(args.incremental),
            threads=threads,
        )
        summary = {
            "rendered_frames": list(result.rendered_frames),
            "skipped_frames": list(result.skipped_frames),
            "output_dir": str(result.output_dir),
            "scene_hash": result.scene_hash,
            "threads": threads,
        }
        sys.stdout.write(
            json.dumps(
                summary,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        sys.stdout.flush()
        return 0
    except SceneError as exc:
        emit_error(exc)
        return exc.exit_code
    except KeyboardInterrupt:
        emit_error(SceneError("E_RENDER", "$", "interrupted"))
        return 130
    except SystemExit as exc:
        return int(exc.code or 0)
    except BaseException as exc:
        if os.environ.get("BONE_RENDER_DEBUG"):
            traceback.print_exc(file=sys.stderr)
        emit_error(
            SceneError(
                "E_RENDER",
                "$",
                f"{type(exc).__name__}: {exc}",
            )
        )
        return 40
