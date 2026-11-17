"""Deterministic CPU path tracer for layered surfaces and area lights."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .geometry import (
    Hit,
    RectLight,
    SphereLight,
    Vec3,
    add,
    any_occlusion,
    cross,
    dot,
    hadamard,
    length,
    length_squared,
    mul,
    nearest_geometry,
    normalize,
    offset_point,
    reflect,
    sub,
)
from .materials import Material, coating_fresnel_and_transmission, energy_bounds, refract
from .sampling import (
    cosine_hemisphere,
    hash01,
    low_discrepancy_2d,
    perturb_around_axis,
    radical_inverse,
)
from .scene import Scene

_LIGHT_TYPES = (RectLight, SphereLight)


@dataclass
class _Counters:
    primary_rays: int = 0
    indirect_rays: int = 0
    reflection_rays: int = 0
    refraction_rays: int = 0
    shadow_rays: int = 0
    depth_clip_count: int = 0
    nan_samples: int = 0

    @property
    def total_rays(self) -> int:
        return (
            self.primary_rays
            + self.indirect_rays
            + self.reflection_rays
            + self.refraction_rays
            + self.shadow_rays
        )


@dataclass
class RenderResult:
    rgb: np.ndarray
    linear_rgb: np.ndarray
    stats: Dict[str, Any]

    def save_png(self, path: str) -> None:
        Image.fromarray(self.rgb, mode="RGB").save(path, format="PNG", compress_level=9)

    def save_stats(self, path: str) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.stats, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")


def _component_multiply(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])


def _finite_color(color: Sequence[float]) -> bool:
    return all(math.isfinite(float(channel)) for channel in color)


class Renderer:
    """A serial deterministic renderer.

    Every stochastic decision is addressed by integer counter coordinates
    (seed, pixel, subpixel, path, bounce, stream). No global RNG, threading, or
    GPU state is involved.
    """

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self._counters = _Counters()
        self._cos_half_fov = math.tan(0.5 * math.radians(scene.camera.fov_degrees))
        self._aspect = float(scene.width) / float(scene.height)
        self._energy_min, self._energy_max, self._energy_warnings = energy_bounds(
            scene.materials.values()
        )

    def _intersect_path(
        self, origin: Vec3, direction: Vec3
    ) -> Optional[Hit]:
        geometry_hit = nearest_geometry(
            self.scene.geometry, origin, direction, 1.0e-9, float("inf")
        )
        light_hit = nearest_geometry(
            self.scene.lights, origin, direction, 1.0e-9, float("inf")
        )
        if geometry_hit is None:
            return light_hit
        if light_hit is None:
            return geometry_hit
        return geometry_hit if geometry_hit.t <= light_hit.t else light_hit

    def _shadow_occluded(self, origin: Vec3, direction: Vec3, max_distance: float) -> bool:
        return any_occlusion(self.scene.geometry, origin, direction, max_distance)

    def _sample_rect_light(
        self, light: RectLight, point: Vec3, surface_normal: Vec3, u: float, v: float
    ) -> Optional[Tuple[Vec3, float, Vec3]]:
        sample = light.sample_point(u, v)
        to_light = sub(sample, point)
        distance_squared = length_squared(to_light)
        if distance_squared <= 1.0e-16:
            return None
        distance = math.sqrt(distance_squared)
        wi = mul(to_light, 1.0 / distance)
        cos_surface = dot(surface_normal, wi)
        if cos_surface <= 0.0:
            return None
        cos_light = abs(dot(light.normal, wi))
        if cos_light <= 0.0:
            return None
        geometry_factor = cos_surface * cos_light * light.area / distance_squared
        return sample, distance, mul(light.intensity, geometry_factor * (1.0 / math.pi))

    def _sample_sphere_light(
        self, light: SphereLight, point: Vec3, surface_normal: Vec3, u: float, v: float
    ) -> Optional[Tuple[Vec3, float, Vec3]]:
        sample, visible_area = light.sample_visible_point(point, u, v)
        to_light = sub(sample, point)
        distance_squared = length_squared(to_light)
        if distance_squared <= 1.0e-16:
            return None
        distance = math.sqrt(distance_squared)
        wi = mul(to_light, 1.0 / distance)
        cos_surface = dot(surface_normal, wi)
        if cos_surface <= 0.0:
            return None
        light_normal = normalize(sub(sample, light.center))
        cos_light = dot(light_normal, mul(wi, -1.0))
        if cos_light <= 0.0:
            return None
        geometry_factor = cos_surface * cos_light * visible_area / distance_squared
        return sample, distance, mul(light.intensity, geometry_factor * (1.0 / math.pi))

    def _direct_lighting(
        self,
        point: Vec3,
        normal: Vec3,
        hit_distance: float,
        pixel_id: int,
        sub_id: int,
        path_id: int,
        depth: int,
    ) -> Tuple[Tuple[float, float, float], float]:
        result = [0.0, 0.0, 0.0]
        visible_samples = 0
        valid_samples = 0
        shadow_samples = self.scene.render.shadow_samples
        epsilon = self.scene.render.self_intersection_epsilon
        shifted_origin = offset_point(point, normal, hit_distance, epsilon)
        for light_index, light in enumerate(self.scene.lights):
            shift_u = hash01(
                self.scene.seed, pixel_id, sub_id, path_id, depth, light_index, 4101
            )
            shift_v = hash01(
                self.scene.seed, pixel_id, sub_id, path_id, depth, light_index, 4102
            )
            light_sum = [0.0, 0.0, 0.0]
            for sample_index in range(shadow_samples):
                u, v = low_discrepancy_2d(sample_index + 1, shift_u, shift_v)
                if isinstance(light, RectLight):
                    sampled = self._sample_rect_light(light, point, normal, u, v)
                else:
                    sampled = self._sample_sphere_light(light, point, normal, u, v)
                if sampled is None:
                    continue
                sample, distance, contribution = sampled
                valid_samples += 1
                direction = mul(sub(sample, point), 1.0 / distance)
                shadow_distance = length(sub(sample, shifted_origin))
                self._counters.shadow_rays += 1
                if self._shadow_occluded(shifted_origin, direction, shadow_distance):
                    continue
                visible_samples += 1
                light_sum[0] += contribution[0]
                light_sum[1] += contribution[1]
                light_sum[2] += contribution[2]
            result[0] += light_sum[0] / float(shadow_samples)
            result[1] += light_sum[1] / float(shadow_samples)
            result[2] += light_sum[2] / float(shadow_samples)
        visibility = (
            float(visible_samples) / float(valid_samples) if valid_samples > 0 else 0.0
        )
        return (result[0], result[1], result[2]), visibility

    def _count_depth_clip(self) -> None:
        self._counters.depth_clip_count += 1

    def _trace_sample(
        self,
        origin: Vec3,
        direction: Vec3,
        pixel_id: int,
        sub_id: int,
        path_id: int,
    ) -> Tuple[Vec3, float, float]:
        self._counters.primary_rays += 1
        throughput: Vec3 = (1.0, 1.0, 1.0)
        radiance: Vec3 = (0.0, 0.0, 0.0)
        specular_bounce = True
        depth = 0
        max_depth = self.scene.render.max_depth
        epsilon = self.scene.render.self_intersection_epsilon
        background = self.scene.background
        shadow_luminance = 0.0
        shadow_visibility = 0.0
        shadow_captured = False

        while depth < max_depth:
            hit = self._intersect_path(origin, direction)
            if hit is None:
                radiance = (
                    radiance[0] + throughput[0] * background[0],
                    radiance[1] + throughput[1] * background[1],
                    radiance[2] + throughput[2] * background[2],
                )
                break

            if isinstance(hit.obj, _LIGHT_TYPES):
                if specular_bounce:
                    emission = hit.obj.intensity
                    radiance = (
                        radiance[0] + throughput[0] * emission[0],
                        radiance[1] + throughput[1] * emission[1],
                        radiance[2] + throughput[2] * emission[2],
                    )
                break

            material: Material = hit.obj.material
            normal = hit.normal
            if dot(direction, normal) > 0.0:
                normal = mul(normal, -1.0)
            cos_incident = max(1.0e-6, -dot(direction, normal))
            fresnel, transmission = coating_fresnel_and_transmission(material, cos_incident)

            branch_shift = hash01(self.scene.seed, pixel_id, sub_id, depth, 5001)
            branch_roll = (radical_inverse(path_id + 1, 5) + branch_shift) % 1.0
            # Increase the coating-reflection event frequency for small F, then
            # compensate its weight by F/p. This preserves the estimator while
            # avoiding rare full-strength light hits ("fireflies") on diffuse
            # surfaces at low path counts.
            reflection_probability = (
                max(fresnel, 0.12) if fresnel > 1.0e-12 else 0.0
            )
            if branch_roll < reflection_probability:
                self._counters.reflection_rays += 1
                reflection_weight = fresnel / reflection_probability
                throughput = mul(throughput, reflection_weight)
                reflected = reflect(direction, normal)
                direction = normalize(reflected)
                origin = offset_point(hit.point, normal, hit.t, epsilon)
                specular_bounce = True
                if depth + 1 >= max_depth:
                    self._count_depth_clip()
                    break
                depth += 1
                continue

            if material.kind == "layered":
                self._counters.refraction_rays += 1
                entered, inside_direction = refract(direction, normal, 1.0, material.coating.ior)
                if entered:
                    # The exit interface's outward normal is opposite the entry
                    # one, but it must be flipped to face the incident ray.
                    exited, outside_direction = refract(
                        inside_direction, normal, material.coating.ior, 1.0
                    )
                    if exited:
                        direction = normalize(outside_direction)

            # The base branch event has probability 1-p, so its conditional
            # throughput is T_eff * Rb / (1-p). T_eff already contains
            # (1-F)^2, hence the (1-F)/(1-p) factor below.
            base_branch_scale = (1.0 - fresnel) / (1.0 - reflection_probability)
            base_gain = mul(_component_multiply(transmission, material.base.color), base_branch_scale)
            scattered_throughput = _component_multiply(throughput, base_gain)

            if material.base.kind == "diffuse":
                direct, visibility = self._direct_lighting(
                    hit.point, normal, hit.t, pixel_id, sub_id, path_id, depth
                )
                if not shadow_captured:
                    shadow_luminance = (
                        0.2126 * scattered_throughput[0] * direct[0]
                        + 0.7152 * scattered_throughput[1] * direct[1]
                        + 0.0722 * scattered_throughput[2] * direct[2]
                    )
                    shadow_visibility = visibility
                    shadow_captured = True
                radiance = (
                    radiance[0] + scattered_throughput[0] * direct[0],
                    radiance[1] + scattered_throughput[1] * direct[1],
                    radiance[2] + scattered_throughput[2] * direct[2],
                )
                if depth + 1 >= max_depth:
                    self._count_depth_clip()
                    break
                shift_u = hash01(self.scene.seed, pixel_id, sub_id, depth, 5101)
                shift_v = hash01(self.scene.seed, pixel_id, sub_id, depth, 5102)
                u1, u2 = low_discrepancy_2d(path_id + 1, shift_u, shift_v)
                direction = normalize(cosine_hemisphere(normal, u1, u2))
                origin = offset_point(hit.point, normal, hit.t, epsilon)
                throughput = scattered_throughput
                specular_bounce = False
                self._counters.indirect_rays += 1
            else:
                if depth + 1 >= max_depth:
                    self._count_depth_clip()
                    break
                mirror = reflect(direction, normal)
                shift_u = hash01(self.scene.seed, pixel_id, sub_id, depth, 5201)
                shift_v = hash01(self.scene.seed, pixel_id, sub_id, depth, 5202)
                u1, u2 = low_discrepancy_2d(path_id + 1, shift_u, shift_v)
                direction = normalize(
                    perturb_around_axis(mirror, normal, material.base.roughness, u1, u2)
                )
                origin = offset_point(hit.point, normal, hit.t, epsilon)
                throughput = scattered_throughput
                specular_bounce = True
                self._counters.reflection_rays += 1
            depth += 1

        return radiance, shadow_luminance, shadow_visibility

    def _primary_direction(self, x: int, y: int, sub_x: int, sub_y: int, n: int) -> Vec3:
        if n == 1:
            offset_x = 0.5
            offset_y = 0.5
        else:
            jitter_x = hash01(self.scene.seed, x, y, sub_x, sub_y, 6001)
            jitter_y = hash01(self.scene.seed, x, y, sub_x, sub_y, 6002)
            offset_x = (float(sub_x) + jitter_x) / float(n)
            offset_y = (float(sub_y) + jitter_y) / float(n)
        pixel_x = (float(x) + offset_x) / float(self.scene.width)
        pixel_y = (float(y) + offset_y) / float(self.scene.height)
        camera_x = (2.0 * pixel_x - 1.0) * self._aspect * self._cos_half_fov
        camera_y = (1.0 - 2.0 * pixel_y) * self._cos_half_fov
        camera = self.scene.camera
        direction = (
            camera.forward[0] + camera.right[0] * camera_x + camera.true_up[0] * camera_y,
            camera.forward[1] + camera.right[1] * camera_x + camera.true_up[1] * camera_y,
            camera.forward[2] + camera.right[2] * camera_x + camera.true_up[2] * camera_y,
        )
        return normalize(direction)

    def _display_linear(self, linear_rgb: np.ndarray) -> np.ndarray:
        exposed = np.maximum(linear_rgb, 0.0) * self.scene.render.exposure
        return 1.0 - np.exp(-exposed)

    @staticmethod
    def _to_srgb(display_linear: np.ndarray) -> np.ndarray:
        srgb = np.where(
            display_linear <= 0.0031308,
            12.92 * display_linear,
            1.055 * np.power(display_linear, 1.0 / 2.4) - 0.055,
        )
        return np.clip(srgb, 0.0, 1.0)

    def _metrics(
        self,
        linear_rgb: np.ndarray,
        nan_mask: np.ndarray,
        shadow_luminance: np.ndarray,
        shadow_visibility: np.ndarray,
    ) -> Dict[str, float]:
        finite_linear = np.where(nan_mask[..., None], 0.0, linear_rgb)
        display = self._display_linear(finite_linear)
        luminance = (
            0.2126 * display[..., 0]
            + 0.7152 * display[..., 1]
            + 0.0722 * display[..., 2]
        )
        shadow_display = 1.0 - np.exp(
            -np.maximum(shadow_luminance, 0.0) * self.scene.render.exposure
        )
        if self.scene.width >= 3 and self.scene.height >= 3:
            image_windows = np.lib.stride_tricks.sliding_window_view(luminance, (3, 3))
            image_local_variance = float(np.mean(np.var(image_windows, axis=(2, 3))))
            windows = np.lib.stride_tricks.sliding_window_view(shadow_display, (3, 3))
            local_variance = np.var(windows, axis=(2, 3))
            interior_visibility = shadow_visibility[1:-1, 1:-1]
            penumbra = (interior_visibility > 0.001) & (interior_visibility < 0.999)
            penumbra_count = int(np.count_nonzero(penumbra))
            shadow_noise = float(np.mean(local_variance))
        else:
            shadow_noise = 0.0
            penumbra_count = 0
            image_local_variance = 0.0
        if self.scene.width >= 2:
            gradient_x = np.abs(np.diff(luminance, axis=1))
        else:
            gradient_x = np.zeros((self.scene.height, 1), dtype=np.float64)
        if self.scene.height >= 2:
            gradient_y = np.abs(np.diff(luminance, axis=0))
        else:
            gradient_y = np.zeros((1, self.scene.width), dtype=np.float64)
        edge_gradient = float(0.5 * (np.mean(gradient_x) + np.mean(gradient_y)))
        return {
            "shadow_noise": shadow_noise,
            "shadow_noise_method": "mean_3x3_variance_of_primary_direct_tonemapped_luminance",
            "shadow_penumbra_pixels": penumbra_count,
            "image_neighborhood_luminance_variance": image_local_variance,
            "edge_gradient": edge_gradient,
            "edge_gradient_method": "mean_adjacent_tonemapped_luminance_difference",
            "mean_tonemapped_luminance": float(np.mean(luminance)),
        }

    def render(self) -> RenderResult:
        self._counters = _Counters()
        width = self.scene.width
        height = self.scene.height
        linear = np.zeros((height, width, 3), dtype=np.float64)
        nan_mask = np.zeros((height, width), dtype=bool)
        shadow_luminance = np.zeros((height, width), dtype=np.float64)
        shadow_visibility = np.zeros((height, width), dtype=np.float64)
        aa = self.scene.render.aa_samples
        spp = self.scene.render.samples_per_pixel
        camera_origin = self.scene.camera.position
        subpixel_count = aa * aa * spp

        for y in range(height):
            for x in range(width):
                pixel_id = y * width + x
                accumulator = [0.0, 0.0, 0.0]
                shadow_accumulator = 0.0
                visibility_accumulator = 0.0
                pixel_has_nan = False
                sub_id = 0
                for sub_y in range(aa):
                    for sub_x in range(aa):
                        direction = self._primary_direction(x, y, sub_x, sub_y, aa)
                        for path_id in range(spp):
                            color, direct_luminance, direct_visibility = self._trace_sample(
                                camera_origin, direction, pixel_id, sub_id, path_id
                            )
                            if not _finite_color(color):
                                pixel_has_nan = True
                                self._counters.nan_samples += 1
                            else:
                                accumulator[0] += color[0]
                                accumulator[1] += color[1]
                                accumulator[2] += color[2]
                                shadow_accumulator += direct_luminance
                                visibility_accumulator += direct_visibility
                        sub_id += 1
                if pixel_has_nan:
                    nan_mask[y, x] = True
                    linear[y, x, :] = 0.0
                else:
                    inverse = 1.0 / float(subpixel_count)
                    linear[y, x, 0] = accumulator[0] * inverse
                    linear[y, x, 1] = accumulator[1] * inverse
                    linear[y, x, 2] = accumulator[2] * inverse
                    shadow_luminance[y, x] = shadow_accumulator * inverse
                    shadow_visibility[y, x] = visibility_accumulator * inverse

        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            display = self._display_linear(linear)
            srgb = self._to_srgb(display)
        rgb = np.floor(srgb * 255.0 + 0.5).astype(np.uint8)
        rgb[nan_mask] = np.array([255, 0, 255], dtype=np.uint8)

        metrics = self._metrics(linear, nan_mask, shadow_luminance, shadow_visibility)
        nan_pixels = int(np.count_nonzero(nan_mask))
        pixel_count = float(width * height)
        counters = self._counters
        stats: Dict[str, Any] = {
            "schema_version": 1,
            "scene_sha256": self.scene.description_hash,
            "width": width,
            "height": height,
            "seed": self.scene.seed,
            "samples_per_pixel": spp,
            "aa_samples_per_axis": aa,
            "aa_subsamples_per_pixel": aa * aa,
            "total_primary_rays": counters.primary_rays,
            "total_rays": counters.total_rays,
            "average_rays_per_pixel": counters.total_rays / pixel_count,
            "indirect_rays": counters.indirect_rays,
            "reflection_rays": counters.reflection_rays,
            "refraction_rays": counters.refraction_rays,
            "shadow_samples_per_light": self.scene.render.shadow_samples,
            "shadow_rays": counters.shadow_rays,
            "average_shadow_rays_per_pixel": counters.shadow_rays / pixel_count,
            "max_depth": self.scene.render.max_depth,
            "depth_clip_count": counters.depth_clip_count,
            "nan_pixels": nan_pixels,
            "nan_samples": counters.nan_samples,
            "max_layer_reflectance": self._energy_max,
            "min_layer_reflectance": self._energy_min,
            "energy_warnings": list(self._energy_warnings),
            "sampling_sequences": {
                "pixel_jitter": "counter-hash-per-pixel-and-stratum",
                "area_lights": "pixel-rotated-2d-halton",
                "random_events": "counter-hash-separate-stream-ids",
            },
        }
        stats.update(metrics)
        return RenderResult(rgb=rgb, linear_rgb=linear, stats=stats)

