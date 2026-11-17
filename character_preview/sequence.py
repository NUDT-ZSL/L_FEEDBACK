from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ._gl import GL
from .config import RenderConfig
from .errors import AppError, RenderError
from .gl_context import OffscreenGLContext
from .model import SceneModel
from .renderer import CharacterRenderer


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_png(path: Path, image: np.ndarray) -> str:
    temporary = path.with_name(path.name + ".tmp")
    try:
        Image.fromarray(image, mode="RGBA").save(
            temporary, format="PNG", compress_level=6
        )
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, path)
        return digest
    finally:
        if temporary.exists():
            temporary.unlink()


def _normalise_exception(exc: BaseException) -> AppError:
    if isinstance(exc, AppError):
        return exc
    gl_error = getattr(exc, "err", None)
    if isinstance(gl_error, int):
        return RenderError(
            "E_GL",
            f"OpenGL error 0x{gl_error:04X}",
            str(exc),
        )
    if isinstance(exc, KeyboardInterrupt):
        return RenderError("E_INTERRUPTED", "render interrupted")
    return RenderError("E_RENDER_UNEXPECTED", str(exc), type(exc).__name__)


def _error_payload(exc: BaseException, completed_frames: list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(exc, AppError):
        code, message, detail = exc.code, exc.message, exc.detail
    else:
        code, message, detail = "E_RENDER_UNEXPECTED", str(exc), type(exc).__name__
    return {
        "schema_version": 1,
        "status": "failed",
        "error": {"code": code, "message": message, "detail": detail},
        "completed_frame_count": len(completed_frames),
        "frames": completed_frames,
    }


def render_sequence(
    model: SceneModel,
    config: RenderConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Render all configured frames, preserving any already written output on failure."""
    output = Path(output_dir)
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RenderError("E_OUTPUT", f"cannot create output directory {output}", str(exc)) from exc

    completed: list[dict[str, Any]] = []
    report_path = output / "render_report.json"
    gl_info: dict[str, Any] = {}
    try:
        with OffscreenGLContext():
            major = int(GL.glGetIntegerv(GL.GL_MAJOR_VERSION))
            minor = int(GL.glGetIntegerv(GL.GL_MINOR_VERSION))
            if (major, minor) < (3, 3):
                raise RenderError(
                    "E_GL_VERSION",
                    "OpenGL 3.3 or newer is required",
                    f"context reports {major}.{minor}",
                )
            gl_info = {
                "version": GL.glGetString(GL.GL_VERSION).decode("utf-8", errors="replace"),
                "renderer": GL.glGetString(GL.GL_RENDERER).decode("utf-8", errors="replace"),
                "vendor": GL.glGetString(GL.GL_VENDOR).decode("utf-8", errors="replace"),
                "glsl_version": GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode(
                    "utf-8", errors="replace"
                ),
                "context_major": major,
                "context_minor": minor,
            }
            renderer = CharacterRenderer(model, config)
            try:
                renderer.initialize()
                for frame_index in range(config.frame_count):
                    time_seconds = frame_index * config.time_step
                    image_name = f"frame_{frame_index:04d}.png"
                    metadata_name = f"frame_{frame_index:04d}.json"
                    image, metadata = renderer.render_frame(
                        frame_index, time_seconds, image_name
                    )
                    digest = _atomic_png(output / image_name, image)
                    metadata["image_sha256"] = digest
                    metadata["model"] = {
                        "units": model.units,
                        "color_space": model.color_space,
                        "mesh_count": len(model.nodes),
                        "material_count": len(model.materials),
                    }
                    _atomic_json(output / metadata_name, metadata)
                    completed.append(
                        {
                            "frame_index": frame_index,
                            "time_seconds": float(time_seconds),
                            "image": image_name,
                            "metadata": metadata_name,
                            "image_sha256": digest,
                        }
                    )
                manifest = {
                    "schema_version": 1,
                    "status": "complete",
                    "config": {
                        "width": config.width,
                        "height": config.height,
                        "shadow_map_size": config.shadow_map_size,
                        "point_shadow_size": config.point_shadow_size,
                        "max_point_lights": config.max_point_lights,
                        "frame_count": config.frame_count,
                        "time_step": config.time_step,
                    },
                    "opengl": gl_info,
                    "resource_counts_during_render": renderer.resource_counts,
                    "frames": completed,
                }
                _atomic_json(output / "manifest.json", manifest)
                if report_path.exists():
                    report_path.unlink()
                return manifest
            finally:
                renderer.close()
    except BaseException as exc:
        normalised = _normalise_exception(exc)
        try:
            _atomic_json(report_path, _error_payload(normalised, completed))
        except OSError:
            pass
        raise normalised from exc

