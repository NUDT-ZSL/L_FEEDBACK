from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DilationResult:
    maps: np.ndarray
    coverage_before: np.ndarray
    coverage_after: np.ndarray
    added_mask: np.ndarray
    seam_mask: np.ndarray
    seam_pixels: int
    dilated_pixels: int


def _erode4(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    return (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )


def dilate_maps(
    maps: np.ndarray,
    coverage: np.ndarray,
    radius: int,
) -> DilationResult:
    maps = np.asarray(maps)
    coverage = np.asarray(coverage, dtype=bool)
    if maps.ndim != 3 or maps.shape[:2] != coverage.shape:
        raise ValueError("Dilation maps must have shape (H, W, C) matching coverage.")
    radius = int(radius)
    if radius < 0:
        raise ValueError("Dilation radius must be non-negative.")

    before = coverage.copy()
    after = coverage.copy()
    working_maps = maps.copy()
    height, width = coverage.shape
    directions = ((-1, 0), (0, -1), (0, 1), (1, 0))

    for _ in range(radius):
        candidate = np.zeros_like(after)
        for dy, dx in directions:
            source = np.zeros_like(after)
            if dy == -1:
                source[1:, :] = after[:-1, :]
            elif dy == 1:
                source[:-1, :] = after[1:, :]
            elif dx == -1:
                source[:, 1:] = after[:, :-1]
            elif dx == 1:
                source[:, :-1] = after[:, 1:]
            candidate |= source
        fill = candidate & ~after
        if not np.any(fill):
            break

        ys, xs = np.nonzero(fill)
        source_y = np.full(ys.shape, -1, dtype=np.int64)
        source_x = np.full(xs.shape, -1, dtype=np.int64)
        for dy, dx in directions:
            ny = ys + dy
            nx = xs + dx
            inside = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
            unset = source_y < 0
            valid_source = np.zeros_like(inside)
            valid_source[inside] = after[ny[inside], nx[inside]]
            choose = inside & unset & valid_source
            source_y[choose] = ny[choose]
            source_x[choose] = nx[choose]

        has_source = source_y >= 0
        working_maps[ys[has_source], xs[has_source]] = working_maps[
            source_y[has_source], source_x[has_source]
        ]
        after[ys[has_source], xs[has_source]] = True

    added = after & ~before
    seam = before & ~_erode4(before)
    return DilationResult(
        maps=working_maps,
        coverage_before=before,
        coverage_after=after,
        added_mask=added,
        seam_mask=seam,
        seam_pixels=int(np.count_nonzero(seam)),
        dilated_pixels=int(np.count_nonzero(added)),
    )

