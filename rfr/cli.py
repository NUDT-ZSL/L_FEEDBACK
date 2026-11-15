"""Command line entry point for the offline renderer."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

from . import __version__
from .compare import compare_framebuffers
from .errors import RenderError
from .factory import create_backend
from .renderer import Renderer, build_draw_list
from .scene import Scene

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rfr", description="Deterministic offline renderer (OpenGL / NumPy software).")
    parser.add_argument("--scene", required=True, help="Path to the JSON scene file.")
    parser.add_argument("--output", required=True, help="Output PNG image path.")
    parser.add_argument("--stats", help="Output JSON statistics path (defaults to <output>.json).")
    parser.add_argument("--frames", type=int, default=1,
                        help="Number of deterministic frames to render.")
    parser.add_argument("--backend", choices=["opengl", "software", "auto"],
                        default="auto", help="Render backend selection.")
    parser.add_argument("--check-consistency", action="store_true",
                        help="Also render with the other backend and compare pixels.")
    parser.add_argument("--tolerance", type=int, default=2,
                        help="Max per-channel absolute difference allowed.")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Frame rate used by deterministic animation.")
    parser.add_argument("--version", action="version", version=f"rfr {__version__}")
    return parser


def _save_png(path: str, framebuffer: np.ndarray) -> None:
    if Image is None:
        raise RenderError("E_IO", "Pillow is not available for PNG output.")
    try:
        Image.fromarray(framebuffer, mode="RGBA").save(path, format="PNG")
    except OSError as exc:
        raise RenderError("E_IO", "Failed to write PNG image.",
                          {"path": path}) from exc


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise RenderError("E_IO", "Failed to write statistics JSON.",
                          {"path": path}) from exc


def run(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    partial: Dict[str, Any] = {
        "scene": args.scene, "output": args.output, "frames_requested": args.frames,
        "backend_requested": args.backend, "frames_completed": 0,
    }
    primary_backend = None
    consistency_backend = None
    try:
        if args.frames <= 0:
            raise RenderError("E_SCENE_INVALID", "--frames must be a positive integer.",
                              {"frames": args.frames})
        scene = Scene.load(args.scene)
        partial["width"] = scene.width
        partial["height"] = scene.height

        requested = "opengl" if args.backend == "auto" else args.backend
        warnings: List[str] = []
        primary_backend, effective = create_backend(requested, scene.width,
                                                    scene.height, warnings)
        partial["backend_effective"] = effective
        renderer = Renderer(primary_backend, scene)
        renderer.warnings.extend(warnings)

        started = time.perf_counter()
        last_frame = None
        for frame_index in range(args.frames):
            scene.update(frame_index, fps=args.fps)
            draws = build_draw_list(scene)
            last_frame = renderer.render_frame(draws)
            partial["frames_completed"] = frame_index + 1
        elapsed = time.perf_counter() - started
        framebuffer = primary_backend.read_pixels()
        stats = renderer.statistics(elapsed)
        stats["frames_requested"] = args.frames
        stats["frames_completed"] = partial["frames_completed"]
        stats["backend_requested"] = args.backend
        stats["backend_effective"] = effective

        consistency = None
        if args.check_consistency:
            other_name = "software" if effective == "opengl" else "opengl"
            consistency_warnings: List[str] = []
            consistency_backend, other_effective = create_backend(
                other_name, scene.width, scene.height, consistency_warnings)
            # Re-evaluate the final frame deterministically for a fair compare.
            scene.update(args.frames - 1, fps=args.fps)
            other_renderer = Renderer(consistency_backend, scene)
            other_renderer.render_frame(build_draw_list(scene))
            other_pixels = consistency_backend.read_pixels()
            reference = framebuffer if effective == "opengl" else other_pixels
            candidate = other_pixels if effective == "opengl" else framebuffer
            consistency = compare_framebuffers(reference, candidate, args.tolerance)
            consistency["compared_backends"] = [effective, other_effective]
            stats["consistency"] = consistency
            stats["warnings"].extend(consistency_warnings)

        _save_png(args.output, framebuffer)
        stats_path = args.stats or f"{args.output}.json"
        stats["stats_path"] = stats_path
        _save_json(stats_path, stats)
        _print_summary(stats, args.output, stats_path, consistency)
        return 0
    except RenderError as exc:
        payload = dict(exc.to_dict())
        payload["partial_statistics"] = _safe_partial(
            partial, primary_backend, consistency_backend)
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
        return 1
    except Exception as exc:  # defensive: never fail silently
        payload = {
            "error": True, "code": "E_RENDER", "message": str(exc),
            "context": {"type": type(exc).__name__},
            "partial_statistics": _safe_partial(
                partial, primary_backend, consistency_backend),
        }
        print(json.dumps(payload, ensure_ascii=False, default=str), file=sys.stderr)
        return 1
    finally:
        for backend in (primary_backend, consistency_backend):
            if backend is not None:
                try:
                    backend.release()
                except Exception:
                    pass


def _safe_partial(partial: Dict[str, Any], primary, secondary) -> Dict[str, Any]:
    data = dict(partial)
    for label, backend in (("primary", primary), ("consistency", secondary)):
        if backend is not None:
            try:
                data[f"{label}_backend_stats"] = backend.backend_statistics()
            except Exception:
                continue
    return data


def _print_summary(stats: Dict[str, Any], output: str, stats_path: str,
                   consistency: Optional[Dict[str, Any]]) -> None:
    print(f"[rfr] backend={stats['backend_effective']} "
          f"draw_calls={stats['draw_calls']} triangles={stats['triangles']} "
          f"state_switches={stats['state_switches']} "
          f"elapsed={stats['elapsed_seconds']:.4f}s")
    print(f"[rfr] image={output}")
    print(f"[rfr] stats={stats_path}")
    if consistency is not None:
        status = "PASS" if consistency["within_tolerance"] else "FAIL"
        print(f"[rfr] consistency={status} "
              f"mismatched={consistency['mismatched_pixels']} "
              f"max_diff={consistency['max_channel_diff']}")


def main() -> None:  # console script entry point
    raise SystemExit(run())


if __name__ == "__main__":
    main()
