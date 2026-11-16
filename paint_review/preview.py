from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from .params import OrbitCamera, PaintParams
from .renderer import PaintRenderer, RenderCancelled


PreviewJob = Tuple[int, PaintParams, OrbitCamera, int, int]


class PreviewWorker(QThread):
    preview_ready = pyqtSignal(int, object, dict)
    failed = pyqtSignal(str)

    def __init__(self, renderer: PaintRenderer, parent=None) -> None:
        super().__init__(parent)
        self.renderer = renderer
        self._condition = threading.Condition()
        self._pending: Optional[PreviewJob] = None
        self._current_cancel: Optional[threading.Event] = None
        self._stopping = False

    def submit(
        self,
        token: int,
        params: PaintParams,
        camera: OrbitCamera,
        width: int,
        height: int,
    ) -> None:
        with self._condition:
            self._pending = (int(token), params.normalized(), camera.normalized(), int(width), int(height))
            if self._current_cancel is not None:
                self._current_cancel.set()
            self._condition.notify()

    def stop_and_wait(self) -> None:
        with self._condition:
            self._stopping = True
            self._pending = None
            if self._current_cancel is not None:
                self._current_cancel.set()
            self._condition.notify_all()
        if self.isRunning():
            self.wait()

    def run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                job = self._pending
                self._pending = None
                cancel = threading.Event()
                self._current_cancel = cancel
            assert job is not None
            token, params, camera, width, height = job
            try:
                if cancel.is_set():
                    continue
                image, stats = self.renderer.render(
                    params, camera, width, height, cancel
                )
                with self._condition:
                    superseded = self._pending is not None or self._stopping
                if not superseded and not cancel.is_set():
                    self.preview_ready.emit(token, image, stats)
            except RenderCancelled:
                pass
            except Exception as exc:
                if not cancel.is_set():
                    self.failed.emit("{}: {}".format(type(exc).__name__, exc))
            finally:
                with self._condition:
                    if self._current_cancel is cancel:
                        self._current_cancel = None

