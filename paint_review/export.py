from __future__ import annotations

import json
import os
import platform
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal

from .params import OrbitCamera, PaintParams, render_hash
from .renderer import PaintRenderer, RenderCancelled


LOG_FILENAME = "render_log.jsonl"


def _timestamped_path(output_dir: Path, params_hash: str) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return output_dir / "paint_review_{}_{}.png".format(stamp, params_hash[:12])


def save_png_deterministic(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        Image.fromarray(image).save(
            str(temporary), format="PNG", compress_level=6
        )
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def append_export_log(output_dir: Path, record: Dict[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / LOG_FILENAME
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return log_path


class ExportJob(QThread):
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(dict)
    cancelled = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        renderer: PaintRenderer,
        params: PaintParams,
        camera: OrbitCamera,
        output_dir: Path,
        preset_name: str,
        width: int = 1920,
        height: int = 1080,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.renderer = renderer
        self.params = params.normalized()
        self.camera = camera.normalized()
        self.output_dir = Path(output_dir)
        self.preset_name = preset_name
        self.width = int(width)
        self.height = int(height)
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        started = time.perf_counter()
        try:
            image, stats = self.renderer.render(
                self.params,
                self.camera,
                self.width,
                self.height,
                self._cancel_event,
                self._emit_progress,
            )
            if self._cancel_event.is_set():
                raise RenderCancelled()
            params_hash = render_hash(self.params, self.camera, self.width, self.height)
            path = _timestamped_path(self.output_dir, params_hash)
            self.progress.emit(99, "写出确定性 PNG")
            save_png_deterministic(image, path)
            if self._cancel_event.is_set():
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                raise RenderCancelled()

            completed_at = datetime.now().astimezone()
            total_ms = (time.perf_counter() - started) * 1000.0
            stats = dict(stats)
            stats["output_path"] = str(path.resolve())
            stats["output_ms"] = round(total_ms - float(stats.get("render_ms", 0.0)), 3)
            stats["total_ms"] = round(total_ms, 3)
            record: Dict[str, object] = {
                "log_schema": "layered-paint-export-v1",
                "timestamp": completed_at.isoformat(timespec="milliseconds"),
                "filename": path.name,
                "output_path": str(path.resolve()),
                "preset": self.preset_name,
                "params_hash": params_hash,
                "material_hash": self.params.material_hash(),
                "camera_hash": self.camera.camera_hash(),
                "params": self.params.canonical_dict(),
                "camera": {
                    "yaw_deg": self.camera.yaw_deg,
                    "elevation_deg": self.camera.elevation_deg,
                    "distance": self.camera.distance,
                    "fov_deg": self.camera.fov_deg,
                    "target": list(self.camera.target),
                },
                "stats": stats,
                "environment": {
                    "source_size": [
                        self.renderer.environment.source_width,
                        self.renderer.environment.source_height,
                    ],
                    "level_dimensions": [
                        list(x) for x in self.renderer.environment.level_dimensions
                    ],
                    "prefilter_ms": round(self.renderer.environment.prefilter_ms, 3),
                    "digest": self.renderer.environment.digest,
                },
                "software": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "pillow": getattr(Image, "__version__", "unknown"),
                    "platform": platform.platform(),
                },
            }
            append_export_log(self.output_dir, record)
            self.progress.emit(100, "出图完成")
            self.completed.emit(record)
        except RenderCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit("{}: {}".format(type(exc).__name__, exc))

    def _emit_progress(self, value: int, message: str) -> None:
        self.progress.emit(int(value), str(message))

