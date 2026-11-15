"""Abstract render backend interface shared by the GPU and software paths."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from ..pipeline import DrawCall
from ..scene import Scene


class RenderBackend(ABC):
    name = "abstract"

    @abstractmethod
    def begin_frame(self, scene: Scene) -> np.ndarray:
        """Clear color/depth and return the RGBA8 framebuffer reference."""

    @abstractmethod
    def apply_state(self, draw: DrawCall, changes: Dict[str, bool]) -> None:
        """Forward only the changed pipeline-state fields."""

    @abstractmethod
    def draw(self, draw: DrawCall) -> None:
        """Rasterize a single draw call into the current framebuffer."""

    @abstractmethod
    def end_frame(self) -> None:
        """Finalize the frame (flush / readback)."""

    @abstractmethod
    def read_pixels(self) -> np.ndarray:
        """Return an (H, W, 4) uint8 RGBA copy of the framebuffer."""

    def backend_statistics(self) -> Dict[str, int]:
        return {}

    def release(self) -> None:  # pragma: no cover - optional resource cleanup
        pass
