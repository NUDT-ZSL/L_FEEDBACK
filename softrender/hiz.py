"""Hierarchical Z conservative occlusion tests."""

from __future__ import annotations

import math

import numpy as np


class HierarchicalZ:
    """A max-depth pyramid with conservative screen-space rectangle queries."""

    def __init__(self, depth: np.ndarray) -> None:
        if depth.ndim != 2 or depth.size == 0:
            raise ValueError("depth must be a non-empty 2D array")
        height, width = depth.shape
        power_width = 1 << max(0, math.ceil(math.log2(width)))
        power_height = 1 << max(0, math.ceil(math.log2(height)))
        base = np.full((power_height, power_width), -np.inf, dtype=np.float64)
        base[:height, :width] = depth
        levels = [base]
        current = base
        while current.shape[0] > 1 or current.shape[1] > 1:
            next_height = max(1, (current.shape[0] + 1) // 2)
            next_width = max(1, (current.shape[1] + 1) // 2)
            reduced = np.full((next_height, next_width), -np.inf, dtype=np.float64)
            source_h = next_height * 2
            source_w = next_width * 2
            padded = np.full((source_h, source_w), -np.inf, dtype=np.float64)
            padded[: current.shape[0], : current.shape[1]] = current
            reduced = np.maximum(
                np.maximum(padded[0::2, 0::2], padded[0::2, 1::2]),
                np.maximum(padded[1::2, 0::2], padded[1::2, 1::2]),
            )
            levels.append(reduced)
            current = reduced
        self.levels = tuple(levels)

    @property
    def level_count(self) -> int:
        return len(self.levels)

    def fully_occluded(
        self,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        nearest_depth: float,
        epsilon: float = 1e-7,
    ) -> bool:
        """Return True only when every pixel in the rectangle is closer than the triangle.

        The query proves occlusion by descending a power-of-two max-depth pyramid.
        A rectangle is occluded when each intersecting pyramid node fully inside the
        rectangle stores a depth smaller than ``nearest_depth``.
        """
        if x_max < x_min or y_max < y_min:
            return False
        limit = nearest_depth - epsilon
        stack: list[tuple[int, int, int]] = [(len(self.levels) - 1, 0, 0)]
        while stack:
            level, cell_x, cell_y = stack.pop()
            extent = 1 << level
            node_x0 = cell_x * extent
            node_y0 = cell_y * extent
            node_x1 = node_x0 + extent - 1
            node_y1 = node_y0 + extent - 1
            if node_x1 < x_min or node_x0 > x_max or node_y1 < y_min or node_y0 > y_max:
                continue
            fully_inside = (
                node_x0 >= x_min
                and node_y0 >= y_min
                and node_x1 <= x_max
                and node_y1 <= y_max
            )
            if fully_inside:
                if float(self.levels[level][cell_y, cell_x]) >= limit:
                    return False
                continue
            if level == 0:
                return False
            child_base = level - 1
            stack.append((child_base, cell_x * 2, cell_y * 2))
            stack.append((child_base, cell_x * 2 + 1, cell_y * 2))
            stack.append((child_base, cell_x * 2, cell_y * 2 + 1))
            stack.append((child_base, cell_x * 2 + 1, cell_y * 2 + 1))
        return True
