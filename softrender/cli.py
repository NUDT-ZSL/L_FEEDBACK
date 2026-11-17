"""Command-line entry point."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from PIL import Image

from .errors import BusinessError
from .renderer import render_scene
from .scene import load_scene


def _json_line(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise BusinessError("E_OUTPUT", f"cannot write output file: {exc}", str(path)) from exc


def _prepare_output_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise BusinessError("E_OUTPUT", "output path exists and is not a directory", str(path))
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BusinessError("E_OUTPUT", f"cannot create output directory: {exc}", str(path)) from exc
    if not os.access(path, os.W_OK):
        raise BusinessError("E_OUTPUT", "output directory is not writable", str(path))


def _prepare_stats_path(path: Path) -> None:
    if path.exists() and path.is_dir():
        raise BusinessError("E_OUTPUT", "stats path is a directory", str(path))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BusinessError("E_OUTPUT", f"cannot create stats directory: {exc}", str(path)) from exc
    if not os.access(path.parent, os.W_OK):
        raise BusinessError("E_OUTPUT", "stats directory is not writable", str(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="softrender",
        description="Deterministic CPU character preview renderer.",
    )
    parser.add_argument("--scene", required=True, help="Path to the scene JSON file.")
    parser.add_argument("--output", default="out", help="Output directory (default: out).")
    parser.add_argument(
        "--stats",
        default=None,
        help="Statistics JSON path (default: <output>/render_stats.json).",
    )
    parser.add_argument("--version", action="version", version="softrender 1.0.0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        scene = load_scene(args.scene)
        result = render_scene(scene)

        output_directory = Path(args.output)
        _prepare_output_directory(output_directory)
        png_path = output_directory / "render.png"
        stats_path = (
            Path(args.stats)
            if args.stats is not None
            else output_directory / "render_stats.json"
        )
        if png_path.resolve() == stats_path.resolve():
            raise BusinessError(
                "E_OUTPUT", "stats path must differ from render.png", str(stats_path)
            )
        _prepare_stats_path(stats_path)

        image = Image.fromarray(result.image, mode="RGB")
        buffer = io.BytesIO()
        try:
            image.save(buffer, format="PNG", optimize=False)
        except (OSError, ValueError) as exc:
            raise BusinessError("E_OUTPUT", f"cannot encode PNG output: {exc}") from exc
        png_bytes = buffer.getvalue()
        _atomic_write(png_path, png_bytes)
        digest = hashlib.sha256(png_bytes).hexdigest()

        stats = result.stats
        stats["output"] = {
            "format": "PNG",
            "file": png_path.name,
            "sha256": digest,
            "bytes": len(png_bytes),
            "colorSpace": "sRGB",
            "gamma": "IEC 61966-2-1",
        }
        stats["determinism"] = {
            "pixelQuantization": "round-half-up to 8-bit sRGB",
            "randomness": "none",
            "sha256": "SHA-256 over the complete PNG byte stream",
        }
        stats_payload = (
            json.dumps(stats, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        _atomic_write(stats_path, stats_payload)
        summary = {
            "ok": True,
            "png": str(png_path),
            "stats": str(stats_path),
            "sha256": digest,
            "resolution": stats["scene"]["resolution"],
            "triangles": stats["counts"]["total_triangles"],
            "culledTriangles": stats["counts"]["culled_triangles"],
            "shadedPixels": stats["counts"]["shaded_pixels"],
        }
        print(_json_line(summary))
        return 0
    except BusinessError as error:
        print(_json_line(error.as_dict()), file=sys.stderr)
        return error.exit_code
    except OSError as exc:
        error = BusinessError("E_OUTPUT", f"filesystem error: {exc}")
        print(_json_line(error.as_dict()), file=sys.stderr)
        return error.exit_code
    except Exception as exc:  # Keep the CLI contract even for unexpected renderer failures.
        error = BusinessError("E_RENDER", f"render failed: {type(exc).__name__}: {exc}")
        print(_json_line(error.as_dict()), file=sys.stderr)
        return error.exit_code
