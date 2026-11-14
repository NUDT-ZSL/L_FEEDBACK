"""Deterministic MaxRects atlas packing and uniform temporal downsampling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import atlas_capacity


@dataclass(frozen=True)
class FreeRect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Placement:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class BlockPlan:
    vertex_index: int
    padded_width: int
    padded_height: int
    payload_width: int
    payload_height: int
    curve_min: tuple[float, float]
    curve_max: tuple[float, float]


@dataclass(frozen=True)
class PackedBlock:
    vertex_index: int
    x: int
    y: int
    padded_width: int
    padded_height: int
    payload_width: int
    payload_height: int
    curve_min: tuple[float, float]
    curve_max: tuple[float, float]

    @property
    def payload_offset(self) -> tuple[int, int]:
        return (self.x + self.padding, self.y + self.padding)

    padding: int = 1


@dataclass(frozen=True)
class PackResult:
    atlas_width: int
    atlas_height: int
    padding: int
    sample_count: int
    base_sample_count: int
    actual_sample_rate: float
    base_sample_rate: float
    downsample_factor: float
    attempts: int
    blocks: tuple[PackedBlock, ...]


def _contains(outer: FreeRect, inner: FreeRect) -> bool:
    return (
        inner.x >= outer.x
        and inner.y >= outer.y
        and inner.x + inner.width <= outer.x + outer.width
        and inner.y + inner.height <= outer.y + outer.height
    )


def _overlaps(a: FreeRect, b: FreeRect) -> bool:
    return not (
        a.x >= b.x + b.width
        or b.x >= a.x + a.width
        or a.y >= b.y + b.height
        or b.y >= a.y + a.height
    )


def _split_free_rect(free: FreeRect, used: FreeRect) -> list[FreeRect]:
    if not _overlaps(free, used):
        return [free]
    pieces: list[FreeRect] = []
    if used.x > free.x:
        pieces.append(FreeRect(free.x, free.y, used.x - free.x, free.height))
    if used.x + used.width < free.x + free.width:
        pieces.append(
            FreeRect(
                used.x + used.width,
                free.y,
                free.x + free.width - (used.x + used.width),
                free.height,
            )
        )
    if used.y > free.y:
        pieces.append(FreeRect(free.x, free.y, free.width, used.y - free.y))
    if used.y + used.height < free.y + free.height:
        pieces.append(
            FreeRect(
                free.x,
                used.y + used.height,
                free.width,
                free.y + free.height - (used.y + used.height),
            )
        )
    return [piece for piece in pieces if piece.width > 0 and piece.height > 0]


def _prune_free(free_rects: list[FreeRect]) -> list[FreeRect]:
    unique: list[FreeRect] = []
    for index, rect in enumerate(free_rects):
        if any(
            _contains(other, rect)
            and (other != rect or other_index < index)
            for other_index, other in enumerate(free_rects)
            if other_index != index
        ):
            continue
        if rect not in unique:
            unique.append(rect)
    return unique


def maxrects_pack(
    width: int,
    height: int,
    sizes: list[tuple[int, int, int]],
) -> list[tuple[int, Placement]] | None:
    """Pack (vertex_index, padded_width, padded_height) with deterministic BSSF."""
    order = sorted(sizes, key=lambda item: (-(item[1] * item[2]), -max(item[1], item[2]), item[0]))
    free_rects = [FreeRect(0, 0, width, height)]
    placements: list[tuple[int, Placement]] = []
    for vertex, item_w, item_h in order:
        if item_w > width or item_h > height:
            return None
        best: tuple[tuple[float, float, int, int, int], FreeRect, FreeRect] | None = None
        for free in free_rects:
            if item_w > free.width or item_h > free.height:
                continue
            leftover_w = free.width - item_w
            leftover_h = free.height - item_h
            score = (
                float(min(leftover_w, leftover_h)),
                float(max(leftover_w, leftover_h)),
                free.y,
                free.x,
                vertex,
            )
            candidate = FreeRect(free.x, free.y, item_w, item_h)
            if best is None or score < best[0]:
                best = (score, free, candidate)
        if best is None:
            return None
        _, _, used = best
        placements.append(
            (
                vertex,
                Placement(
                    x=used.x, y=used.y, width=used.width, height=used.height
                ),
            )
        )
        next_free: list[FreeRect] = []
        for free in free_rects:
            next_free.extend(_split_free_rect(free, used))
        free_rects = _prune_free(next_free)
    return placements


def block_payload_dimensions(
    sample_count: int,
    extent_u: float,
    extent_v: float,
    min_block_size: int,
) -> tuple[int, int]:
    """Choose a rectangle whose aspect follows the curve bounding box."""
    sample_count = max(1, int(sample_count))
    aspect = (max(extent_u, 0.0) + 1e-9) / (max(extent_v, 0.0) + 1e-9)
    limit = max(min_block_size, sample_count)
    height = max(
        min_block_size,
        min(limit, int(round(math.sqrt(sample_count / max(aspect, 1e-12))))),
    )
    width = max(
        min_block_size,
        min(limit, int(math.ceil(sample_count / max(1, height)))),
    )
    height = max(height, int(math.ceil(sample_count / max(1, width))))
    while width * height < sample_count:
        if width < limit:
            width += 1
        elif height < limit:
            height += 1
        else:
            break
    return width, height


def sample_count_for_rate(
    texture_sample_rate: float,
    duration: float,
    loop: bool,
    scale: float = 1.0,
) -> int:
    if loop:
        return max(1, int(round(texture_sample_rate * duration * scale)))
    return max(2, int(round(texture_sample_rate * duration * scale)) + 1)


def actual_rate_for_count(sample_count: int, duration: float, loop: bool) -> float:
    return (
        sample_count / duration
        if loop
        else (sample_count - 1) / duration
    )


def plan_and_pack(
    curves: np.ndarray,
    duration: float,
    loop: bool,
    texture_sample_rate: float,
    atlas_width: int,
    atlas_height: int,
    min_block_size: int,
    padding: int,
) -> PackResult:
    """Use one global temporal scale for every block and repack until it fits."""
    vertex_count = curves.shape[1]
    extents: list[tuple[float, float, tuple[float, float], tuple[float, float]]] = []
    for vertex in range(vertex_count):
        series = curves[:, vertex, :]
        if len(series) == 0:
            minimum = np.zeros(2)
            maximum = np.zeros(2)
        else:
            minimum = np.min(series, axis=0)
            maximum = np.max(series, axis=0)
        extents.append(
            (
                float(maximum[0] - minimum[0]),
                float(maximum[1] - minimum[1]),
                (float(minimum[0]), float(minimum[1])),
                (float(maximum[0]), float(maximum[1])),
            )
        )

    base_count = sample_count_for_rate(texture_sample_rate, duration, loop)
    factor = 1.0
    last_count = -1
    for attempt in range(1, 129):
        sample_count = sample_count_for_rate(texture_sample_rate, duration, loop, factor)
        minimum_count = 1 if loop else 2
        if sample_count < minimum_count:
            sample_count = minimum_count
        actual_rate = actual_rate_for_count(sample_count, duration, loop)
        effective_factor = actual_rate / texture_sample_rate
        plans: list[BlockPlan] = []
        for vertex, (extent_u, extent_v, curve_min, curve_max) in enumerate(extents):
            payload_w, payload_h = block_payload_dimensions(
                sample_count, extent_u, extent_v, min_block_size
            )
            plans.append(
                BlockPlan(
                    vertex_index=vertex,
                    padded_width=payload_w + 2 * padding,
                    padded_height=payload_h + 2 * padding,
                    payload_width=payload_w,
                    payload_height=payload_h,
                    curve_min=curve_min,
                    curve_max=curve_max,
                )
            )
        placements = maxrects_pack(
            atlas_width,
            atlas_height,
            [(plan.vertex_index, plan.padded_width, plan.padded_height) for plan in plans],
        )
        if placements is not None:
            plan_by_vertex = {plan.vertex_index: plan for plan in plans}
            blocks: list[PackedBlock] = []
            for vertex_index, placement in placements:
                plan = plan_by_vertex[vertex_index]
                blocks.append(
                    PackedBlock(
                        vertex_index=plan.vertex_index,
                        x=placement.x,
                        y=placement.y,
                        padded_width=placement.width,
                        padded_height=placement.height,
                        payload_width=plan.payload_width,
                        payload_height=plan.payload_height,
                        curve_min=plan.curve_min,
                        curve_max=plan.curve_max,
                        padding=padding,
                    )
                )
            blocks.sort(key=lambda block: block.vertex_index)
            return PackResult(
                atlas_width=atlas_width,
                atlas_height=atlas_height,
                padding=padding,
                sample_count=sample_count,
                base_sample_count=base_count,
                actual_sample_rate=actual_rate,
                base_sample_rate=texture_sample_rate,
                downsample_factor=effective_factor,
                attempts=attempt,
                blocks=tuple(blocks),
            )

        if sample_count == minimum_count and sample_count == last_count:
            break
        last_count = sample_count
        factor *= 0.9

    raise atlas_capacity(
        f"atlas {atlas_width}x{atlas_height} cannot hold {vertex_count} padded vertex curves "
        f"at the minimum block resolution"
    )

