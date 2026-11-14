from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .aperture import (
    ApertureKernel,
    build_aperture_kernel,
    quantized_kernel_specs,
)
from .config import ApertureConfig, LensConfig, RenderConfig
from .depth import DepthFrame
from .lens import ThinLensModel


@dataclass(frozen=True)
class RenderResult:
    image_bgr: np.ndarray
    opacity: np.ndarray
    min_depth_m: np.ndarray
    mean_depth_m: np.ndarray


@dataclass
class _Accumulator:
    rgb_sum: np.ndarray
    weight_sum: np.ndarray
    depth_sum: np.ndarray
    min_depth: np.ndarray

    @classmethod
    def zeros(cls, height: int, width: int) -> _Accumulator:
        return cls(
            rgb_sum=np.zeros((height, width, 3), dtype=np.float64),
            weight_sum=np.zeros((height, width), dtype=np.float64),
            depth_sum=np.zeros((height, width), dtype=np.float64),
            min_depth=np.full((height, width), np.inf, dtype=np.float64),
        )

    def merge(self, other: _Accumulator) -> None:
        self.rgb_sum += other.rgb_sum
        self.weight_sum += other.weight_sum
        self.depth_sum += other.depth_sum
        np.minimum(self.min_depth, other.min_depth, out=self.min_depth)


class _KernelBank:
    def __init__(
        self,
        aperture: ApertureConfig,
        max_radius_px: float,
        gather_rounds: int,
    ):
        self.quantization = 0.5
        self.max_radius_px = max_radius_px
        specs = quantized_kernel_specs(
            blades=aperture.blades,
            max_radius_px=self.max_radius_px,
            rotation_deg=aperture.rotation_deg,
            supersample=aperture.supersample,
            quantization_px=self.quantization,
        )
        self.radii = sorted(specs)
        self.kernels: list[ApertureKernel] = [
            build_aperture_kernel(specs[radius]) for radius in self.radii
        ]
        self.offset_groups: list[list[list[tuple[int, int, float]]]] = []
        for kernel in self.kernels:
            offsets: list[tuple[int, int, float]] = []
            half = kernel.half_size
            for dy in range(-half, half + 1):
                for dx in range(-half, half + 1):
                    weight = float(kernel.weights[dy + half, dx + half])
                    if weight > 0.0:
                        offsets.append((dy, dx, weight))
            groups = [[] for _ in range(gather_rounds)]
            if offsets:
                denominator = max(float(half), 1e-12)
                for dy, dx, weight in offsets:
                    radial = min(1.0, math.hypot(dy, dx) / denominator)
                    round_index = min(
                        gather_rounds - 1, math.floor(radial * gather_rounds)
                    )
                    groups[round_index].append((dy, dx, weight))
            self.offset_groups.append(groups)

    def radius_bin_index(self, radius_px: np.ndarray) -> np.ndarray:
        if len(self.radii) == 1:
            return np.zeros(radius_px.shape, dtype=np.int16)
        midpoints = (
            np.asarray(self.radii[:-1], dtype=np.float64)
            + np.asarray(self.radii[1:], dtype=np.float64)
        ) * 0.5
        return np.searchsorted(
            midpoints, np.asarray(radius_px, dtype=np.float64), side="left"
        ).astype(np.int16)


class DOFRenderer:
    """Depth-binned, visibility-aware aperture scattering renderer."""

    def __init__(
        self,
        lens_config: LensConfig,
        aperture_config: ApertureConfig,
        render_config: RenderConfig,
        width: int,
        height: int,
    ):
        self.lens_config = lens_config
        self.aperture_config = aperture_config
        self.render_config = render_config
        self.width = int(width)
        self.height = int(height)
        self.lens = ThinLensModel(lens_config, width)
        self.bank = _KernelBank(
            aperture_config,
            lens_config.max_blur_radius_px,
            render_config.gather_rounds,
        )

    def _slice_index(self, depth_m: np.ndarray) -> np.ndarray:
        span = self.lens_config.far_plane_m - self.lens_config.near_plane_m
        normalized = np.clip(
            (depth_m - self.lens_config.near_plane_m) / span, 0.0, 1.0
        )
        index = np.floor(normalized * self.render_config.depth_slices).astype(np.int16)
        return np.clip(index, 0, self.render_config.depth_slices - 1)

    def _scatter_offset(
        self,
        accumulator: _Accumulator,
        color: np.ndarray,
        depth_m: np.ndarray,
        source_mask: np.ndarray,
        dy: int,
        dx: int,
        weight: float,
    ) -> None:
        height, width = depth_m.shape
        source_y0 = max(0, -dy)
        source_y1 = min(height, height - dy)
        source_x0 = max(0, -dx)
        source_x1 = min(width, width - dx)
        if source_y0 >= source_y1 or source_x0 >= source_x1:
            return
        dest_y0 = source_y0 + dy
        dest_y1 = source_y1 + dy
        dest_x0 = source_x0 + dx
        dest_x1 = source_x1 + dx

        src_depth = depth_m[source_y0:source_y1, source_x0:source_x1]
        dst_depth = depth_m[dest_y0:dest_y1, dest_x0:dest_x1]
        active = source_mask[source_y0:source_y1, source_x0:source_x1] & (
            src_depth <= dst_depth + self.lens_config.depth_tolerance_m
        )
        if not np.any(active):
            return

        src_rgb = color[source_y0:source_y1, source_x0:source_x1]
        active_f = active.astype(np.float64)
        contribution = weight * active_f
        accumulator.rgb_sum[dest_y0:dest_y1, dest_x0:dest_x1] += (
            contribution[:, :, None] * src_rgb
        )
        accumulator.weight_sum[dest_y0:dest_y1, dest_x0:dest_x1] += contribution
        accumulator.depth_sum[dest_y0:dest_y1, dest_x0:dest_x1] += (
            contribution * src_depth
        )
        candidate_depth = np.where(active, src_depth, np.inf)
        np.minimum(
            accumulator.min_depth[dest_y0:dest_y1, dest_x0:dest_x1],
            candidate_depth,
            out=accumulator.min_depth[dest_y0:dest_y1, dest_x0:dest_x1],
        )

    def _render_layer(
        self,
        color: np.ndarray,
        depth_m: np.ndarray,
        radius_bins: np.ndarray,
        layer_mask: np.ndarray,
    ) -> _Accumulator:
        height, width = depth_m.shape
        layer = _Accumulator.zeros(height, width)
        for radius_index, groups in enumerate(self.bank.offset_groups):
            group_mask = layer_mask & (radius_bins == radius_index)
            if not np.any(group_mask):
                continue
            for gather_round, offsets in enumerate(groups):
                if not offsets:
                    continue
                round_accumulator = _Accumulator.zeros(height, width)
                for dy, dx, weight in offsets:
                    self._scatter_offset(
                        round_accumulator,
                        color,
                        depth_m,
                        group_mask,
                        dy,
                        dx,
                        weight,
                    )
                # Multi-round gather: each round gathers a radial aperture band
                # into the layer. Depth is monotonically reduced, never increased.
                layer.merge(round_accumulator)
        return layer

    def render(self, color_bgr: np.ndarray, depth_frame: DepthFrame, focus_m: float) -> RenderResult:
        color = color_bgr.astype(np.float64) / 255.0
        depth_m = depth_frame.meters
        radius_px = self.lens.coc_radius_px(depth_m, focus_m)
        radius_bins = self.bank.radius_bin_index(radius_px)
        slice_index = self._slice_index(depth_m)

        global_accumulator = _Accumulator.zeros(self.height, self.width)
        # Composite from the farthest slice to the nearest slice. Every source
        # sample was visibility-tested at its destination, so adding the far
        # contribution first cannot place a far pixel in front of a near one.
        for slice_number in range(self.render_config.depth_slices - 1, -1, -1):
            layer_mask = slice_index == slice_number
            if not np.any(layer_mask):
                continue
            layer = self._render_layer(color, depth_m, radius_bins, layer_mask)
            global_accumulator.merge(layer)

        weight = global_accumulator.weight_sum
        valid = weight > 1e-15
        normalized = np.empty_like(global_accumulator.rgb_sum)
        normalized[valid] = global_accumulator.rgb_sum[valid] / weight[valid, None]
        normalized[~valid] = color[~valid]
        image = np.clip(np.rint(normalized * 255.0), 0.0, 255.0).astype(np.uint8)

        mean_depth = np.full(weight.shape, np.inf, dtype=np.float64)
        mean_depth[valid] = global_accumulator.depth_sum[valid] / weight[valid]
        result = RenderResult(
            image_bgr=np.ascontiguousarray(image),
            opacity=weight,
            min_depth_m=global_accumulator.min_depth,
            mean_depth_m=mean_depth,
        )
        return result


def image_difference_8bit(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    if first.shape != second.shape:
        raise ValueError(f"image shape mismatch: {first.shape} vs {second.shape}")
    difference = np.abs(first.astype(np.float64) - second.astype(np.float64))
    return {
        "max_abs_error_8bit": float(np.max(difference)) if difference.size else 0.0,
        "mean_abs_error_8bit": float(np.mean(difference)) if difference.size else 0.0,
    }
