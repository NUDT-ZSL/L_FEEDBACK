"""Deterministic coverage rasterization and layer compositing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .geometry import (
    as_array,
    build_segments,
    clip_polygon_to_rect,
    clip_segments_to_rect,
    distances_to_segment,
    has_self_intersection,
    points_inside_even_odd,
    simplify_for_low,
)
from .model import LayerSpec, SceneSpec, ShapeSpec


SUPERSAMPLE_COUNT = 4
SUPERSAMPLE_OFFSETS = (np.arange(SUPERSAMPLE_COUNT, dtype=np.float64) + 0.5) / (
    SUPERSAMPLE_COUNT
)


@dataclass(frozen=True)
class PreparedShape:
    spec: ShapeSpec
    lod: str
    points: np.ndarray
    segments: np.ndarray
    closed: bool
    width: float
    bbox: tuple[int, int, int, int] | None
    source_bbox: tuple[float, float, float, float] | None
    vertex_count: int
    self_intersecting: bool
    self_intersecting_by_lod: dict[str, bool]
    source_vertex_counts: dict[str, int]


@dataclass(frozen=True)
class PreparedLayer:
    spec: LayerSpec
    shapes: tuple[PreparedShape, ...]


@dataclass
class RasterStats:
    peak_rasterized_pixels: int = 0
    total_rasterized_pixels: int = 0
    peak_candidate_pixels: int = 0
    tiles_processed: int = 0

    def observe_tile(self, pixels: int, candidate_pixels: int) -> None:
        self.tiles_processed += 1
        self.total_rasterized_pixels += pixels
        self.peak_rasterized_pixels = max(self.peak_rasterized_pixels, pixels)
        self.peak_candidate_pixels = max(
            self.peak_candidate_pixels, candidate_pixels
        )


@dataclass(frozen=True)
class LayerRasterResult:
    id: str
    order: int
    bbox: tuple[int, int, int, int] | None
    nontransparent_pixels: int
    self_intersecting: bool
    lod: str
    vertex_counts: dict[str, int]
    shapes: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class RasterResult:
    albedo_rgba: np.ndarray
    height_coverage: np.ndarray
    layers: tuple[LayerRasterResult, ...]
    stats: RasterStats
    tiles_x: int
    tiles_y: int


def _bbox_of_points(points: np.ndarray) -> tuple[float, float, float, float] | None:
    if len(points) == 0:
        return None
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return (
        float(minimum[0]),
        float(minimum[1]),
        float(maximum[0]),
        float(maximum[1]),
    )


def _raster_bbox(
    source_bbox: tuple[float, float, float, float] | None,
    width: int,
    height: int,
    padding: float = 0.0,
) -> tuple[int, int, int, int] | None:
    if source_bbox is None:
        return None
    xmin = max(0.0, source_bbox[0] - padding)
    ymin = max(0.0, source_bbox[1] - padding)
    xmax = min(float(width), source_bbox[2] + padding)
    ymax = min(float(height), source_bbox[3] + padding)
    if xmax <= xmin or ymax <= ymin:
        return None
    x0 = max(0, int(math.floor(xmin)))
    y0 = max(0, int(math.floor(ymin)))
    x1 = min(width, int(math.ceil(xmax)))
    y1 = min(height, int(math.ceil(ymax)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def prepare_layers(
    scene: SceneSpec,
    *,
    lod: str,
    width: int,
    height: int,
) -> tuple[PreparedLayer, ...]:
    prepared_layers: list[PreparedLayer] = []
    clip_rect = (-2.0, -2.0, float(width) + 2.0, float(height) + 2.0)

    for layer in scene.layers:
        prepared_shapes: list[PreparedShape] = []
        for shape in layer.shapes:
            raw_points = as_array(shape.levels[lod])
            points = raw_points
            if lod == "low":
                points = simplify_for_low(
                    raw_points,
                    kind=shape.kind,
                    closed=shape.closed,
                    merge_distance=shape.merge_distance,
                    tolerance=shape.simplify_tolerance,
                )

            closed = shape.kind == "polygon" or shape.closed
            self_by_lod = {
                level: has_self_intersection(
                    as_array(shape.levels[level]), closed=closed
                )
                for level in ("high", "medium", "low")
            }
            self_intersecting = self_by_lod[lod]
            source_vertex_counts = {
                level: len(shape.levels[level])
                for level in ("high", "medium", "low")
            }

            source_bbox = _bbox_of_points(raw_points)
            if shape.kind == "polygon":
                if self_intersecting:
                    render_points = points
                else:
                    render_points = clip_polygon_to_rect(points, clip_rect)
                if len(render_points) < 3:
                    render_points = np.empty((0, 2), dtype=np.float64)
                segments = build_segments(render_points, closed=True)
                bbox = _raster_bbox(source_bbox, width, height)
                padding = 0.0
            else:
                radius = shape.width * 0.5
                raw_segments = build_segments(points, closed=closed)
                expanded_clip = (
                    -radius - 2.0,
                    -radius - 2.0,
                    float(width) + radius + 2.0,
                    float(height) + radius + 2.0,
                )
                segments = clip_segments_to_rect(raw_segments, expanded_clip)
                bbox = _raster_bbox(
                    source_bbox, width, height, padding=radius + 1.0
                )
                padding = radius + 1.0

            prepared_shapes.append(
                PreparedShape(
                    spec=shape,
                    lod=lod,
                    points=render_points,
                    segments=segments,
                    closed=closed,
                    width=shape.width,
                    bbox=bbox,
                    source_bbox=source_bbox,
                    vertex_count=len(points),
                    self_intersecting=self_intersecting,
                    self_intersecting_by_lod=self_by_lod,
                    source_vertex_counts=source_vertex_counts,
                )
            )
        prepared_layers.append(PreparedLayer(spec=layer, shapes=tuple(prepared_shapes)))

    return tuple(prepared_layers)


def _tile_span(
    bbox: tuple[int, int, int, int], grid_size: int
) -> tuple[range, range]:
    x0, y0, x1, y1 = bbox
    return (
        range(x0 // grid_size, ((x1 - 1) // grid_size) + 1),
        range(y0 // grid_size, ((y1 - 1) // grid_size) + 1),
    )


def _boundary_candidates(
    segments: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    expansion: float,
) -> np.ndarray:
    candidate = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    if len(segments) == 0:
        return candidate
    for segment in segments:
        sx0, sy0, sx1, sy1 = (float(value) for value in segment)
        min_x = min(sx0, sx1) - expansion
        max_x = max(sx0, sx1) + expansion
        min_y = min(sy0, sy1) - expansion
        max_y = max(sy0, sy1) + expansion
        gx0 = max(x0, int(math.floor(min_x)) - 1)
        gx1 = min(x1, int(math.floor(max_x)) + 2)
        gy0 = max(y0, int(math.floor(min_y)) - 1)
        gy1 = min(y1, int(math.floor(max_y)) + 2)
        if gx1 > gx0 and gy1 > gy0:
            candidate[gy0 - y0 : gy1 - y0, gx0 - x0 : gx1 - x0] = True
    return candidate


def _sample_points(
    pixel_coords: np.ndarray, x0: int, y0: int
) -> np.ndarray:
    """Build a 4x4 sample grid for each (row, column) tile coordinate."""

    pixel_x = pixel_coords[:, 1].astype(np.float64) + x0
    pixel_y = pixel_coords[:, 0].astype(np.float64) + y0
    sample_x_4 = pixel_x[:, None] + SUPERSAMPLE_OFFSETS[None, :]
    sample_y_4 = pixel_y[:, None] + SUPERSAMPLE_OFFSETS[None, :]
    sample_x = np.repeat(sample_x_4, SUPERSAMPLE_COUNT, axis=1)
    sample_y = np.tile(sample_y_4, (1, SUPERSAMPLE_COUNT))
    return np.stack((sample_x, sample_y), axis=2).reshape(-1, 2)


def _scanline_center_mask(
    points: np.ndarray,
    segments: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> np.ndarray:
    width = x1 - x0
    height = y1 - y0
    mask = np.zeros((height, width), dtype=np.float64)
    x_centers = np.arange(x0, x1, dtype=np.float64) + 0.5
    if len(points) < 3 or len(segments) == 0:
        return mask

    for row in range(height):
        scan_y = y0 + row + 0.5
        sy0 = segments[:, 1]
        sy1 = segments[:, 3]
        crossing = ((sy0 <= scan_y) & (scan_y < sy1)) | (
            (sy1 <= scan_y) & (scan_y < sy0)
        )
        active = segments[crossing]
        if len(active) == 0:
            continue
        dy = active[:, 3] - active[:, 1]
        sx = active[:, 0]
        ex = active[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            intersections = sx + (scan_y - active[:, 1]) * (ex - sx) / dy
        intersections = np.sort(intersections)
        pair_count = len(intersections) // 2
        for pair in range(pair_count):
            left = float(intersections[2 * pair])
            right = float(intersections[2 * pair + 1])
            if right <= left:
                continue
            start = int(np.searchsorted(x_centers, left, side="left"))
            end = int(np.searchsorted(x_centers, right, side="left"))
            if end > start:
                mask[row, start:end] = 1.0
    return mask


def rasterize_polygon_tile(
    shape: PreparedShape,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[np.ndarray, int]:
    coverage = _scanline_center_mask(
        shape.points, shape.segments, x0, y0, x1, y1
    )
    candidates = _boundary_candidates(shape.segments, x0, y0, x1, y1, expansion=1.0)
    candidate_count = int(np.count_nonzero(candidates))
    if candidate_count:
        coords = np.argwhere(candidates)
        samples = _sample_points(coords, x0, y0)
        inside = points_inside_even_odd(samples, shape.points)
        counts = inside.reshape(candidate_count, SUPERSAMPLE_COUNT ** 2).sum(axis=1)
        coverage[coords[:, 0], coords[:, 1]] = counts.astype(np.float64) / 16.0
    return coverage, candidate_count


def rasterize_polyline_tile(
    shape: PreparedShape,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[np.ndarray, int]:
    coverage = np.zeros((y1 - y0, x1 - x0), dtype=np.float64)
    if len(shape.segments) == 0:
        return coverage, 0
    radius = shape.width * 0.5
    relevant = np.zeros(len(shape.segments), dtype=bool)
    for index, segment in enumerate(shape.segments):
        min_x = min(float(segment[0]), float(segment[2])) - radius
        max_x = max(float(segment[0]), float(segment[2])) + radius
        min_y = min(float(segment[1]), float(segment[3])) - radius
        max_y = max(float(segment[1]), float(segment[3])) + radius
        if max_x >= x0 and min_x < x1 and max_y >= y0 and min_y < y1:
            relevant[index] = True
    if not np.any(relevant):
        return coverage, 0

    candidates = _boundary_candidates(
        shape.segments[relevant], x0, y0, x1, y1, expansion=radius + 0.5
    )
    candidate_count = int(np.count_nonzero(candidates))
    if candidate_count == 0:
        return coverage, 0

    coords = np.argwhere(candidates)
    samples = _sample_points(coords, x0, y0)
    minimum_distance = np.full(len(samples), np.inf, dtype=np.float64)
    for segment in shape.segments[relevant]:
        distance = distances_to_segment(samples, segment)
        minimum_distance = np.minimum(minimum_distance, distance)
    inside = minimum_distance <= radius
    counts = inside.reshape(candidate_count, SUPERSAMPLE_COUNT ** 2).sum(axis=1)
    coverage[coords[:, 0], coords[:, 1]] = (
        counts.astype(np.float64) / float(SUPERSAMPLE_COUNT ** 2)
    )
    return coverage, candidate_count


class CoverageRasterizer:
    """Rasterize prepared layers tile-by-tile into a straight-alpha RGBA canvas."""

    def __init__(
        self,
        scene: SceneSpec,
        layered_shapes: Iterable[PreparedLayer],
        *,
        width: int,
        height: int,
        grid_size: int,
    ) -> None:
        self.scene = scene
        self.layers = tuple(layered_shapes)
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.tiles_x = (width + grid_size - 1) // grid_size
        self.tiles_y = (height + grid_size - 1) // grid_size
        self.stats = RasterStats()
        self.destination = np.zeros((height, width, 4), dtype=np.float64)
        self.height_coverage = np.zeros((height, width), dtype=np.float64)

        background = scene.background
        self.destination[:, :, 0] = background[0] / 255.0
        self.destination[:, :, 1] = background[1] / 255.0
        self.destination[:, :, 2] = background[2] / 255.0
        self.destination[:, :, 3] = background[3] / 255.0

    def _composite_layer_tile(
        self,
        color: np.ndarray,
        alpha: float,
        layer_coverage: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> None:
        source_alpha = alpha * layer_coverage
        active = source_alpha > 0.0
        if not np.any(active):
            return
        destination = self.destination[y0:y1, x0:x1]
        old_alpha = destination[:, :, 3]
        new_alpha = source_alpha + old_alpha * (1.0 - source_alpha)
        weighted_rgb = (
            color[None, None, :] * source_alpha[:, :, None]
            + destination[:, :, :3]
            * old_alpha[:, :, None]
            * (1.0 - source_alpha[:, :, None])
        )
        destination[active, 3] = new_alpha[active]
        destination[active, :3] = np.divide(
            weighted_rgb[active],
            new_alpha[active, None],
            out=np.zeros((int(np.count_nonzero(active)), 3), dtype=np.float64),
            where=new_alpha[active, None] > 0.0,
        )

    def _rasterize_shape_tile(
        self,
        shape: PreparedShape,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> np.ndarray:
        if shape.spec.kind == "polygon":
            coverage, candidates = rasterize_polygon_tile(
                shape, x0=x0, y0=y0, x1=x1, y1=y1
            )
        else:
            coverage, candidates = rasterize_polyline_tile(
                shape, x0=x0, y0=y0, x1=x1, y1=y1
            )
        self.stats.observe_tile((x1 - x0) * (y1 - y0), candidates)
        return coverage

    def run(self) -> RasterResult:
        layer_results: list[LayerRasterResult] = []

        for layer in self.layers:
            layer_tiles: set[tuple[int, int]] = set()
            for shape in layer.shapes:
                if shape.bbox is None:
                    continue
                x_tiles, y_tiles = _tile_span(shape.bbox, self.grid_size)
                for ty in y_tiles:
                    for tx in x_tiles:
                        layer_tiles.add((ty, tx))

            bbox_min_x: int | None = None
            bbox_min_y: int | None = None
            bbox_max_x: int | None = None
            bbox_max_y: int | None = None
            nontransparent_pixels = 0
            color = np.asarray(layer.spec.color, dtype=np.float64) / 255.0

            for tile_y, tile_x in sorted(layer_tiles):
                x0 = tile_x * self.grid_size
                y0 = tile_y * self.grid_size
                x1 = min(self.width, x0 + self.grid_size)
                y1 = min(self.height, y0 + self.grid_size)
                layer_coverage = np.zeros((y1 - y0, x1 - x0), dtype=np.float64)

                for shape in layer.shapes:
                    if shape.bbox is None:
                        continue
                    sx0, sy0, sx1, sy1 = shape.bbox
                    if sx1 <= x0 or sx0 >= x1 or sy1 <= y0 or sy0 >= y1:
                        continue
                    shape_coverage = self._rasterize_shape_tile(
                        shape, x0, y0, x1, y1
                    )
                    np.maximum(layer_coverage, shape_coverage, out=layer_coverage)

                hit_rows, hit_columns = np.nonzero(layer_coverage > 0.0)
                if len(hit_rows) == 0:
                    continue

                global_rows = hit_rows + y0
                global_columns = hit_columns + x0
                nontransparent_pixels += len(hit_rows)
                tile_min_x = int(global_columns.min())
                tile_max_x = int(global_columns.max()) + 1
                tile_min_y = int(global_rows.min())
                tile_max_y = int(global_rows.max()) + 1
                bbox_min_x = (
                    tile_min_x if bbox_min_x is None else min(bbox_min_x, tile_min_x)
                )
                bbox_min_y = (
                    tile_min_y if bbox_min_y is None else min(bbox_min_y, tile_min_y)
                )
                bbox_max_x = (
                    tile_max_x if bbox_max_x is None else max(bbox_max_x, tile_max_x)
                )
                bbox_max_y = (
                    tile_max_y if bbox_max_y is None else max(bbox_max_y, tile_max_y)
                )

                np.maximum(
                    self.height_coverage[y0:y1, x0:x1],
                    layer_coverage,
                    out=self.height_coverage[y0:y1, x0:x1],
                )
                self._composite_layer_tile(
                    color,
                    layer.spec.alpha,
                    layer_coverage,
                    x0,
                    y0,
                    x1,
                    y1,
                )

            layer_bbox = None
            if bbox_min_x is not None:
                assert bbox_min_y is not None
                assert bbox_max_x is not None
                assert bbox_max_y is not None
                layer_bbox = (bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y)

            selected_vertex_count = sum(
                shape.vertex_count for shape in layer.shapes
            )
            all_lod_vertex_counts = {
                lod: sum(
                    shape.source_vertex_counts[lod] for shape in layer.shapes
                )
                for lod in ("high", "medium", "low")
            }
            shape_metadata: list[dict[str, object]] = []
            for shape in layer.shapes:
                shape_metadata.append(
                    {
                        "id": shape.spec.id,
                        "kind": shape.spec.kind,
                        "vertex_count": shape.vertex_count,
                        "source_vertex_counts": dict(shape.source_vertex_counts),
                        "source_bbox": (
                            list(shape.source_bbox)
                            if shape.source_bbox is not None
                            else None
                        ),
                        "canvas_bbox": list(shape.bbox) if shape.bbox else None,
                        "self_intersecting": shape.self_intersecting,
                        "self_intersecting_by_lod": dict(
                            shape.self_intersecting_by_lod
                        ),
                    }
                )

            layer_results.append(
                LayerRasterResult(
                    id=layer.spec.id,
                    order=layer.spec.order,
                    bbox=layer_bbox,
                    nontransparent_pixels=nontransparent_pixels,
                    self_intersecting=any(
                        shape.self_intersecting for shape in layer.shapes
                    ),
                    lod=layer.shapes[0].lod if layer.shapes else "",
                    vertex_counts=all_lod_vertex_counts,
                    shapes=tuple(shape_metadata),
                )
            )

        albedo = _to_uint8_rgba(self.destination)
        return RasterResult(
            albedo_rgba=albedo,
            height_coverage=self.height_coverage,
            layers=tuple(layer_results),
            stats=self.stats,
            tiles_x=self.tiles_x,
            tiles_y=self.tiles_y,
        )


def _to_uint8_rgba(values: np.ndarray) -> np.ndarray:
    encoded = np.floor(np.clip(values, 0.0, 1.0) * 255.0 + 0.5)
    return encoded.astype(np.uint8)


def rasterize_scene(
    scene: SceneSpec,
    *,
    width: int,
    height: int,
    lod: str,
    grid_size: int,
) -> RasterResult:
    prepared = prepare_layers(scene, lod=lod, width=width, height=height)
    rasterizer = CoverageRasterizer(
        scene,
        prepared,
        width=width,
        height=height,
        grid_size=grid_size,
    )
    return rasterizer.run()
