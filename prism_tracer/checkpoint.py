"""Atomic tile checkpointing for exact resumability."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from .errors import BusinessError, business_error

_ARRAY_NAMES = (
    "sum_rgb",
    "sum_sq_rgb",
    "sum_direct_rgb",
    "sum_indirect_rgb",
    "aux_contribution",
    "aux_hits",
)


@dataclass
class RenderState:
    width: int
    height: int
    tile: int
    tiles_x: int
    tiles_y: int
    completed: np.ndarray
    sum_rgb: np.ndarray
    sum_sq_rgb: np.ndarray
    sum_direct_rgb: np.ndarray
    sum_indirect_rgb: np.ndarray
    aux_contribution: np.ndarray
    aux_hits: np.ndarray

    @classmethod
    def empty(cls, width: int, height: int, tile: int) -> "RenderState":
        tiles_x = (width + tile - 1) // tile
        tiles_y = (height + tile - 1) // tile
        shape = (height, width, 3)
        return cls(
            width,
            height,
            tile,
            tiles_x,
            tiles_y,
            np.zeros((tiles_y, tiles_x), dtype=bool),
            np.zeros(shape, dtype=np.float64),
            np.zeros(shape, dtype=np.float64),
            np.zeros(shape, dtype=np.float64),
            np.zeros(shape, dtype=np.float64),
            np.zeros(shape, dtype=np.float64),
            np.zeros(shape, dtype=np.float64),
        )

    def add_tile(self, tile_x: int, tile_y: int, tile_data: dict[str, np.ndarray]) -> None:
        x0 = tile_x * self.tile
        y0 = tile_y * self.tile
        x1 = min(x0 + self.tile, self.width)
        y1 = min(y0 + self.tile, self.height)
        shaped = {
            name: tile_data[name].reshape(y1 - y0, x1 - x0, 3)
            for name in _ARRAY_NAMES
        }
        self.sum_rgb[y0:y1, x0:x1] += shaped["sum_rgb"]
        self.sum_sq_rgb[y0:y1, x0:x1] += shaped["sum_sq_rgb"]
        self.sum_direct_rgb[y0:y1, x0:x1] += shaped["sum_direct_rgb"]
        self.sum_indirect_rgb[y0:y1, x0:x1] += shaped["sum_indirect_rgb"]
        self.aux_contribution[y0:y1, x0:x1] += shaped["aux_contribution"]
        self.aux_hits[y0:y1, x0:x1] += shaped["aux_hits"]
        self.completed[tile_y, tile_x] = True


class Checkpoint:
    def __init__(self, path: Path, fingerprint: str) -> None:
        self.path = path
        self.fingerprint = fingerprint

    def load_or_create(self, width: int, height: int, tile: int) -> RenderState:
        if not self.path.exists():
            return RenderState.empty(width, height, tile)
        try:
            with np.load(self.path, allow_pickle=False) as archive:
                fingerprint = str(archive["fingerprint"].item())
                if fingerprint != self.fingerprint:
                    raise business_error(
                        "E_CHECKPOINT",
                        f"checkpoint parameters differ from this run: {self.path}",
                    )
                expected_shape = (height, width, 3)
                completed = np.array(archive["completed"], dtype=bool, copy=True)
                arrays = {
                    name: np.array(archive[name], dtype=np.float64, copy=True)
                    for name in _ARRAY_NAMES
                }
                for name, array in arrays.items():
                    if array.shape != expected_shape:
                        raise business_error(
                            "E_CHECKPOINT",
                            f"checkpoint array {name} has shape {array.shape}, expected {expected_shape}",
                        )
                tiles_x = (width + tile - 1) // tile
                tiles_y = (height + tile - 1) // tile
                if completed.shape != (tiles_y, tiles_x):
                    raise business_error(
                        "E_CHECKPOINT",
                        "checkpoint tile mask shape does not match this run",
                    )
                return RenderState(
                    width,
                    height,
                    tile,
                    tiles_x,
                    tiles_y,
                    completed,
                    arrays["sum_rgb"],
                    arrays["sum_sq_rgb"],
                    arrays["sum_direct_rgb"],
                    arrays["sum_indirect_rgb"],
                    arrays["aux_contribution"],
                    arrays["aux_hits"],
                )
        except BusinessError:
            raise
        except Exception as exc:
            raise business_error(
                "E_CHECKPOINT", f"cannot read checkpoint {self.path}: {exc}"
            ) from exc

    def save(self, state: RenderState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez(
                    handle,
                    fingerprint=np.asarray([self.fingerprint]),
                    completed=state.completed,
                    sum_rgb=state.sum_rgb,
                    sum_sq_rgb=state.sum_sq_rgb,
                    sum_direct_rgb=state.sum_direct_rgb,
                    sum_indirect_rgb=state.sum_indirect_rgb,
                    aux_contribution=state.aux_contribution,
                    aux_hits=state.aux_hits,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass

    def remove(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

