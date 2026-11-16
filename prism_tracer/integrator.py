"""Monte Carlo path tracing, spectral transport, and caustic photon mapping."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .geometry import (
    KIND_GROUND,
    KIND_LIGHT,
    KIND_NONE,
    KIND_TRIANGLE,
    KIND_WATER,
    intersect_scene,
    occluded_by_prism,
)
from .rng import cosine_hemisphere, r2_2d, uniform
from .scene import Scene

MEDIUM_AIR = 0
MEDIUM_GLASS = 1
MEDIUM_WATER = 2

STREAM_CAMERA = 11
STREAM_MATERIAL = 12
STREAM_LIGHT = 13
STREAM_SPECTRAL = 14
STREAM_PHOTON_INIT = 15
STREAM_PHOTON_PATH = 16
STREAM_CAMERA_STRATA = 21
STREAM_MATERIAL_STRATA = 22
STREAM_LIGHT_STRATA = 23
STREAM_PHOTON_AREA_STRATA = 24
STREAM_PHOTON_DIR_STRATA = 25
STREAM_PHOTON_WAVE_STRATA = 26

CLASS_DIRECT = 0
CLASS_REFRACT = 1
CLASS_TIR = 2

_EPS = 1.0e-5
_INV_PI = 1.0 / math.pi
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)


@dataclass
class CausticMap:
    """Bilinear splat/query map of glass-path photon power on the floor."""

    x0: float
    x1: float
    z0: float
    z1: float
    resolution: int
    refract_energy: np.ndarray
    tir_energy: np.ndarray
    photons_emitted: int
    photons_deposited_refract: int
    photons_deposited_tir: int
    photons_with_glass_transmission: int
    smoothing_passes: int

    @property
    def cell_area(self) -> float:
        dx = (self.x1 - self.x0) / self.resolution
        dz = (self.z1 - self.z0) / self.resolution
        return dx * dz

    def query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        count = points.shape[0]
        refract = np.zeros((count, 3), dtype=np.float64)
        tir = np.zeros((count, 3), dtype=np.float64)
        if self.photons_emitted == 0 or count == 0:
            return refract, tir

        x = points[:, 0]
        z = points[:, 2]
        inside = (
            (x >= self.x0)
            & (x <= self.x1)
            & (z >= self.z0)
            & (z <= self.z1)
        )
        if not np.any(inside):
            return refract, tir

        scale = self.resolution - 1
        fx = (x - self.x0) / (self.x1 - self.x0) * scale
        fz = (z - self.z0) / (self.z1 - self.z0) * scale
        ix0 = np.floor(fx).astype(np.int64)
        iz0 = np.floor(fz).astype(np.int64)
        ix0 = np.clip(ix0, 0, self.resolution - 1)
        iz0 = np.clip(iz0, 0, self.resolution - 1)
        ix1 = np.minimum(ix0 + 1, self.resolution - 1)
        iz1 = np.minimum(iz0 + 1, self.resolution - 1)
        tx = np.clip(fx - ix0, 0.0, 1.0)
        tz = np.clip(fz - iz0, 0.0, 1.0)
        w00 = (1.0 - tx) * (1.0 - tz)
        w10 = tx * (1.0 - tz)
        w01 = (1.0 - tx) * tz
        w11 = tx * tz

        refr_sub = self.refract_energy / self.cell_area
        tir_sub = self.tir_energy / self.cell_area
        for row, column, weight in (
            (iz0, ix0, w00),
            (iz0, ix1, w10),
            (iz1, ix0, w01),
            (iz1, ix1, w11),
        ):
            refract[inside] += refr_sub[row[inside], column[inside]] * weight[inside, None]
            tir[inside] += tir_sub[row[inside], column[inside]] * weight[inside, None]
        return refract, tir


def _normalize(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(lengths, 1.0e-30)


def _medium_ior(scene: Scene, medium: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    assert scene.spectral is not None
    ior = np.ones_like(wavelength, dtype=np.float64)
    glass = medium == MEDIUM_GLASS
    water = medium == MEDIUM_WATER
    if np.any(glass):
        ior[glass] = scene.spectral.ior("glass", wavelength[glass])
    if np.any(water):
        ior[water] = scene.spectral.ior("water", wavelength[water])
    return ior


def _segment_absorption(
    scene: Scene,
    throughput: np.ndarray,
    medium: np.ndarray,
    wavelength: np.ndarray,
    distance: np.ndarray,
) -> np.ndarray:
    assert scene.spectral is not None
    sigma = np.zeros_like(wavelength, dtype=np.float64)
    glass = medium == MEDIUM_GLASS
    water = medium == MEDIUM_WATER
    if np.any(glass):
        sigma[glass] = scene.spectral.sigma("glass", wavelength[glass])
    if np.any(water):
        sigma[water] = scene.spectral.sigma("water", wavelength[water])
    safe_distance = np.where(np.isfinite(distance), distance, 0.0)
    attenuation = np.exp(-sigma * safe_distance)
    return throughput * attenuation[:, None]


def _fresnel_unpolarized(cos_i: np.ndarray, cos_t: np.ndarray, n1: np.ndarray, n2: np.ndarray) -> np.ndarray:
    rs = (n1 * cos_i - n2 * cos_t) / np.maximum(n1 * cos_i + n2 * cos_t, 1.0e-30)
    rp = (n1 * cos_t - n2 * cos_i) / np.maximum(n1 * cos_t + n2 * cos_i, 1.0e-30)
    return 0.5 * (rs * rs + rp * rp)


def _dielectric_response(
    scene: Scene,
    directions: np.ndarray,
    outward_normals: np.ndarray,
    medium: np.ndarray,
    target_medium: np.ndarray,
    wavelength: np.ndarray,
    branch_random: np.ndarray,
    *,
    radiance_mode: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return new directions, new media, transmission factors, and TIR flags."""

    assert scene.spectral is not None
    ndotd = np.einsum("ij,ij->i", directions, outward_normals)
    normals = np.where(ndotd[:, None] > 0.0, -outward_normals, outward_normals)
    cos_i = np.clip(-np.einsum("ij,ij->i", directions, normals), 0.0, 1.0)
    n1 = _medium_ior(scene, medium, wavelength)
    n2 = _medium_ior(scene, target_medium, wavelength)
    eta = n1 / n2
    sin2_t = eta * eta * np.maximum(0.0, 1.0 - cos_i * cos_i)
    total_internal_reflection = sin2_t >= 1.0
    cos_t = np.sqrt(np.maximum(0.0, 1.0 - sin2_t))
    fresnel = _fresnel_unpolarized(cos_i, cos_t, n1, n2)
    fresnel = np.where(total_internal_reflection, 1.0, np.clip(fresnel, 0.0, 1.0))

    reflected = total_internal_reflection | (branch_random < fresnel)
    reflected_direction = _normalize(directions + 2.0 * cos_i[:, None] * normals)
    transmitted_direction = _normalize(
        eta[:, None] * directions
        + (eta * cos_i - cos_t)[:, None] * normals
    )
    new_direction = np.where(reflected[:, None], reflected_direction, transmitted_direction)
    new_medium = np.where(reflected, medium, target_medium).astype(np.int8)
    transmission_factor = np.ones_like(medium, dtype=np.float64)
    if radiance_mode:
        transmission_factor = np.where(reflected, 1.0, eta * eta)
    return new_direction, new_medium, transmission_factor, total_internal_reflection


def _add_bilinear(
    energy: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    power: np.ndarray,
    x0: float,
    x1: float,
    z0: float,
    z1: float,
) -> None:
    resolution = energy.shape[0]
    inside = (x >= x0) & (x <= x1) & (z >= z0) & (z <= z1)
    if not np.any(inside):
        return
    x = x[inside]
    z = z[inside]
    power = power[inside]
    scale = resolution - 1
    fx = (x - x0) / (x1 - x0) * scale
    fz = (z - z0) / (z1 - z0) * scale
    ix0 = np.clip(np.floor(fx).astype(np.int64), 0, resolution - 1)
    iz0 = np.clip(np.floor(fz).astype(np.int64), 0, resolution - 1)
    ix1 = np.minimum(ix0 + 1, resolution - 1)
    iz1 = np.minimum(iz0 + 1, resolution - 1)
    tx = np.clip(fx - ix0, 0.0, 1.0)
    tz = np.clip(fz - iz0, 0.0, 1.0)
    for iy, ix, weight in (
        (iz0, ix0, (1.0 - tx) * (1.0 - tz)),
        (iz0, ix1, tx * (1.0 - tz)),
        (iz1, ix0, (1.0 - tx) * tz),
        (iz1, ix1, tx * tz),
    ):
        np.add.at(energy, (iy, ix), power * weight[:, None])


def _smooth_map(energy: np.ndarray, passes: int) -> np.ndarray:
    """Separable 5-tap Gaussian; zero padding preserves total map energy."""

    result = energy
    kernel = (1.0, 4.0, 6.0, 4.0, 1.0)
    for _ in range(passes):
        padded = np.pad(result, ((2, 2), (0, 0), (0, 0)), mode="constant")
        result = sum(
            weight * padded[offset : offset + result.shape[0], :, :]
            for offset, weight in enumerate(kernel)
        ) / 16.0
        padded = np.pad(result, ((0, 0), (2, 2), (0, 0)), mode="constant")
        result = sum(
            weight * padded[:, offset : offset + result.shape[1], :]
            for offset, weight in enumerate(kernel)
        ) / 16.0
    return result


def _pixel_sample_permutation(
    seed: int,
    pixel_ids: np.ndarray,
    sample_ids: np.ndarray,
    pixel_count: int,
    spp: int,
    stream: int,
) -> np.ndarray:
    """Independent per-pixel permutation used to decorrelate strata sets."""

    keys = uniform(seed, pixel_ids, sample_ids, stream, 0, 0).reshape(pixel_count, spp)
    order = np.argsort(keys, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks,
        order,
        np.broadcast_to(np.arange(spp, dtype=np.int64), (pixel_count, spp)),
        axis=1,
    )
    return ranks.reshape(-1)


def _photon_permutation(seed: int, photon_ids: np.ndarray, stream: int) -> np.ndarray:
    keys = uniform(seed, photon_ids, 0, stream, 0, 0)
    order = np.argsort(keys, kind="stable")
    ranks = np.empty(photon_ids.size, dtype=np.int64)
    ranks[order] = np.arange(photon_ids.size, dtype=np.int64)
    return ranks


class Renderer:
    def __init__(
        self,
        scene: Scene,
        width: int,
        height: int,
        spp: int,
        seed: int,
        tile: int,
        max_depth: int,
        use_indirect: bool,
        caustic_photons: int,
        caustic_map_resolution: int,
    ) -> None:
        self.scene = scene
        self.width = width
        self.height = height
        self.spp = spp
        self.seed = seed
        self.tile = tile
        self.max_depth = max_depth
        self.use_indirect = use_indirect
        self.caustic_photons = caustic_photons
        self.caustic_map_resolution = caustic_map_resolution
        self.caustic_map: CausticMap | None = None

    # ------------------------------------------------------------------
    # Light sampling / direct visibility
    # ------------------------------------------------------------------
    def _sample_light_points(
        self,
        pixel_ids: np.ndarray,
        strata_ids: np.ndarray,
        bounce: int,
    ) -> np.ndarray:
        assert (
            self.scene.light_center is not None
            and self.scene.light_u_hat is not None
            and self.scene.light_v_hat is not None
        )
        rotation_u = uniform(self.seed, pixel_ids, 0, STREAM_LIGHT, 1000 + bounce, 0)
        rotation_v = uniform(self.seed, pixel_ids, 0, STREAM_LIGHT, 1000 + bounce, 1)
        u, v = r2_2d(strata_ids, rotation_u, rotation_v)
        return (
            self.scene.light_center
            + ((2.0 * u - 1.0) * self.scene.light_u_half)[:, None] * self.scene.light_u_hat
            + ((2.0 * v - 1.0) * self.scene.light_v_half)[:, None] * self.scene.light_v_hat
        )

    def _shadow_visibility(
        self,
        points: np.ndarray,
        directions: np.ndarray,
        distances: np.ndarray,
        medium: np.ndarray,
        wavelength: np.ndarray,
    ) -> np.ndarray:
        """Visibility through the planar water interface; glass is an occluder.

        The caustic photon map supplies light that actually bends through the
        prism, so NEE deliberately treats the glass mesh as opaque.  This is
        the non-overlapping decomposition used by the report's direct term.
        """

        assert self.scene.spectral is not None
        visibility = np.ones(points.shape[0], dtype=np.float64)
        underwater = medium == MEDIUM_WATER
        if np.any(underwater):
            indices = np.nonzero(underwater)[0]
            p = points[indices]
            d = directions[indices]
            dist = distances[indices]
            dy = d[:, 1]
            valid = dy > 1.0e-8
            t_cross = np.full(indices.shape[0], np.inf, dtype=np.float64)
            t_cross[valid] = (self.scene.water_y - p[valid, 1]) / dy[valid]
            crossing = p + t_cross[:, None] * d
            valid &= (
                np.isfinite(t_cross)
                & (t_cross > 0.0)
                & (t_cross <= dist)
                & (crossing[:, 0] >= self.scene.water_x0)
                & (crossing[:, 0] <= self.scene.water_x1)
                & (crossing[:, 2] >= self.scene.water_z0)
                & (crossing[:, 2] <= self.scene.water_z1)
            )

            n_water = self.scene.spectral.ior("water", wavelength[indices])
            cos_i = np.clip(dy, 0.0, 1.0)
            sin2_t = (n_water / 1.0) ** 2 * np.maximum(0.0, 1.0 - cos_i * cos_i)
            tir = sin2_t >= 1.0
            cos_t = np.sqrt(np.maximum(0.0, 1.0 - sin2_t))
            fresnel = _fresnel_unpolarized(cos_i, cos_t, n_water, np.ones_like(n_water))
            transmittance = np.where(tir, 0.0, 1.0 - fresnel)
            sigma = self.scene.spectral.sigma("water", wavelength[indices])
            water_distance = np.where(valid, np.minimum(t_cross, dist), 0.0)
            water_visibility = np.where(
                valid,
                transmittance * np.exp(-sigma * water_distance),
                0.0,
            )
            visibility[indices] *= water_visibility
            medium[indices[valid]] = MEDIUM_AIR

        if np.any(visibility > 0.0):
            blocked = occluded_by_prism(self.scene, points, directions, distances)
            visibility[blocked] = 0.0
        return np.clip(visibility, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Camera path integration
    # ------------------------------------------------------------------
    def _trace_camera_rays(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        wavelength: np.ndarray,
        rgb_weights: np.ndarray,
        pixel_ids: np.ndarray,
        sample_ids: np.ndarray,
        light_strata_ids: np.ndarray,
        material_strata_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Trace a batch of camera rays.

        Returns total RGB, direct RGB, indirect RGB, aux-class contribution
        (N x 3), and aux-class hit counts (N x 3).
        """

        count = origins.shape[0]
        total = np.zeros((count, 3), dtype=np.float64)
        direct = np.zeros((count, 3), dtype=np.float64)
        indirect = np.zeros((count, 3), dtype=np.float64)
        aux_contribution = np.zeros((count, 3), dtype=np.float64)
        aux_counts = np.zeros((count, 3), dtype=np.float64)
        aux_seen = np.zeros((count, 3), dtype=bool)

        throughput = np.ones((count, 3), dtype=np.float64)
        medium = np.zeros(count, dtype=np.int8)
        diffuse_bounces = np.zeros(count, dtype=np.int16)
        had_glass = np.zeros(count, dtype=bool)
        had_tir = np.zeros(count, dtype=bool)
        glass_since_diffuse = np.zeros(count, dtype=bool)
        reflection_since_diffuse = np.zeros(count, dtype=bool)
        alive = np.ones(count, dtype=bool)

        def record_aux(indices: np.ndarray, class_id: int, contribution: np.ndarray) -> None:
            if indices.size == 0:
                return
            luminance = np.maximum(0.0, contribution @ _LUMA)
            positive = luminance > 1.0e-15
            if not np.any(positive):
                return
            aux_contribution[indices, class_id] += luminance
            fresh = positive & ~aux_seen[indices, class_id]
            aux_seen[indices[positive], class_id] = True
            aux_counts[indices[fresh], class_id] += 1.0

        for bounce in range(self.max_depth + 1):
            if not np.any(alive):
                break
            distance, kind, primitive = intersect_scene(
                self.scene, origins, directions, tmin=1.0e-5
            )
            hit = alive & (kind != KIND_NONE)

            if np.any(hit):
                hit_indices = np.nonzero(hit)[0]
                throughput[hit_indices] = _segment_absorption(
                    self.scene,
                    throughput[hit_indices],
                    medium[hit_indices],
                    wavelength[hit_indices],
                    distance[hit_indices],
                )

            specular = hit & ((kind == KIND_TRIANGLE) | (kind == KIND_WATER))
            if np.any(specular):
                idx = np.nonzero(specular)[0]
                p = origins[idx] + distance[idx, None] * directions[idx]
                triangle_subset = idx[kind[idx] == KIND_TRIANGLE]
                water_subset = idx[kind[idx] == KIND_WATER]
                if triangle_subset.size:
                    tri_pos = triangle_subset
                    tri_p = origins[tri_pos] + distance[tri_pos, None] * directions[tri_pos]
                    tri_branch = uniform(
                        self.seed,
                        pixel_ids[tri_pos],
                        sample_ids[tri_pos],
                        STREAM_MATERIAL,
                        bounce,
                        0,
                    )
                    assert self.scene.triangle_normals is not None
                    outward = self.scene.triangle_normals[primitive[tri_pos]]
                    target = np.where(
                        medium[tri_pos] == MEDIUM_GLASS, MEDIUM_AIR, MEDIUM_GLASS
                    )
                    new_d, new_m, factor, tir = _dielectric_response(
                        self.scene,
                        directions[tri_pos],
                        outward,
                        medium[tri_pos],
                        target,
                        wavelength[tri_pos],
                        tri_branch,
                        radiance_mode=True,
                    )
                    reflected = new_m == medium[tri_pos]
                    transmitted = ~reflected
                    had_tir[tri_pos[tir]] = True
                    had_glass[tri_pos[transmitted]] = True
                    glass_since_diffuse[tri_pos] = True
                    reflection_since_diffuse[tri_pos[reflected]] = True
                    throughput[tri_pos[transmitted]] *= factor[transmitted, None]
                    medium[tri_pos] = new_m
                    directions[tri_pos] = new_d
                    origins[tri_pos] = tri_p + new_d * _EPS
                if water_subset.size:
                    wat_pos = water_subset
                    wat_p = origins[wat_pos] + distance[wat_pos, None] * directions[wat_pos]
                    wat_branch = uniform(
                        self.seed,
                        pixel_ids[wat_pos],
                        sample_ids[wat_pos],
                        STREAM_MATERIAL,
                        bounce,
                        0,
                    )
                    outward = np.broadcast_to(_UP, (wat_pos.size, 3)).copy()
                    target = np.where(
                        medium[wat_pos] == MEDIUM_WATER, MEDIUM_AIR, MEDIUM_WATER
                    )
                    new_d, new_m, factor, tir = _dielectric_response(
                        self.scene,
                        directions[wat_pos],
                        outward,
                        medium[wat_pos],
                        target,
                        wavelength[wat_pos],
                        wat_branch,
                        radiance_mode=True,
                    )
                    reflected = new_m == medium[wat_pos]
                    transmitted = ~reflected
                    had_tir[wat_pos[tir]] = True
                    reflection_since_diffuse[wat_pos[reflected]] = True
                    throughput[wat_pos[transmitted]] *= factor[transmitted, None]
                    medium[wat_pos] = new_m
                    directions[wat_pos] = new_d
                    origins[wat_pos] = wat_p + new_d * _EPS

            light_hit = hit & (kind == KIND_LIGHT)
            if np.any(light_hit):
                idx = np.nonzero(light_hit)[0]
                covered_by_nee = (
                    (diffuse_bounces[idx] > 0)
                    & (
                        glass_since_diffuse[idx]
                        | ~reflection_since_diffuse[idx]
                    )
                )
                contribution = (
                    throughput[idx]
                    * rgb_weights[idx]
                    * self.scene.light_emission
                )
                contribution[covered_by_nee] = 0.0
                is_direct = diffuse_bounces[idx] == 0
                direct[idx[is_direct]] += contribution[is_direct]
                indirect[idx[~is_direct]] += contribution[~is_direct]
                total[idx] += contribution
                path_class = np.where(
                    had_tir[idx],
                    CLASS_TIR,
                    np.where(had_glass[idx], CLASS_REFRACT, CLASS_DIRECT),
                )
                for class_id in (CLASS_DIRECT, CLASS_REFRACT, CLASS_TIR):
                    selected = idx[path_class == class_id]
                    if selected.size:
                        record_aux(selected, class_id, contribution[path_class == class_id])
                alive[idx] = False

            ground_hit = hit & (kind == KIND_GROUND)
            if np.any(ground_hit):
                idx = np.nonzero(ground_hit)[0]
                p = origins[idx] + distance[idx, None] * directions[idx]
                assert self.scene.ground_albedo is not None
                albedo = self.scene.ground_albedo

                light_points = self._sample_light_points(
                    pixel_ids[idx], light_strata_ids[idx], bounce
                )
                to_light = light_points - p
                light_distance = np.linalg.norm(to_light, axis=1)
                wi = to_light / np.maximum(light_distance[:, None], 1.0e-30)
                assert self.scene.light_normal is not None
                cos_surface = np.clip(wi[:, 1], 0.0, 1.0)
                cos_light = np.clip(-(wi @ self.scene.light_normal), 0.0, 1.0)
                geometry = (
                    self.scene.light_area
                    * cos_surface
                    * cos_light
                    / np.maximum(light_distance * light_distance, 1.0e-12)
                )
                visibility = self._shadow_visibility(
                    p,
                    wi,
                    light_distance,
                    medium[idx].copy(),
                    wavelength[idx],
                )
                direct_contribution = (
                    throughput[idx]
                    * (albedo * _INV_PI)[None, :]
                    * self.scene.light_emission
                    * (geometry * visibility)[:, None]
                    * rgb_weights[idx]
                )
                is_direct = diffuse_bounces[idx] == 0
                direct[idx[is_direct]] += direct_contribution[is_direct]
                indirect[idx[~is_direct]] += direct_contribution[~is_direct]
                total[idx] += direct_contribution
                path_class = np.where(
                    had_tir[idx],
                    CLASS_TIR,
                    np.where(had_glass[idx], CLASS_REFRACT, CLASS_DIRECT),
                )
                for class_id in (CLASS_DIRECT, CLASS_REFRACT, CLASS_TIR):
                    selected = idx[path_class == class_id]
                    if selected.size:
                        record_aux(
                            selected,
                            class_id,
                            direct_contribution[path_class == class_id],
                        )

                if self.caustic_map is not None:
                    caustic_refract, caustic_tir = self.caustic_map.query(p)
                    for class_id, irradiance in (
                        (CLASS_REFRACT, caustic_refract),
                        (CLASS_TIR, caustic_tir),
                    ):
                        caustic_contribution = (
                            throughput[idx]
                            * (albedo * _INV_PI)[None, :]
                            * irradiance
                        )
                        direct[idx[is_direct]] += caustic_contribution[is_direct]
                        indirect[idx[~is_direct]] += caustic_contribution[~is_direct]
                        total[idx] += caustic_contribution
                        record_aux(idx, class_id, caustic_contribution)

                if (not self.use_indirect) or bounce >= self.max_depth:
                    alive[idx] = False
                else:
                    rotation_u = uniform(
                        self.seed, pixel_ids[idx], 0, STREAM_MATERIAL, 1000 + bounce, 1
                    )
                    rotation_v = uniform(
                        self.seed, pixel_ids[idx], 0, STREAM_MATERIAL, 1000 + bounce, 2
                    )
                    u1, u2 = r2_2d(material_strata_ids[idx], rotation_u, rotation_v)
                    normals = np.broadcast_to(_UP, (idx.size, 3)).copy()
                    directions[idx] = cosine_hemisphere(normals, u1, u2)
                    origins[idx] = p + normals * 2.0e-5
                    through_water = (
                        (p[:, 0] >= self.scene.water_x0)
                        & (p[:, 0] <= self.scene.water_x1)
                        & (p[:, 2] >= self.scene.water_z0)
                        & (p[:, 2] <= self.scene.water_z1)
                        & (self.scene.water_y > self.scene.ground_y)
                    )
                    medium[idx] = np.where(
                        through_water, MEDIUM_WATER, MEDIUM_AIR
                    ).astype(np.int8)
                    throughput[idx] *= albedo[None, :]
                    diffuse_bounces[idx] += 1
                    glass_since_diffuse[idx] = False
                    reflection_since_diffuse[idx] = False

            missed = alive & (kind == KIND_NONE)
            alive[missed] = False

        return total, direct, indirect, aux_contribution, aux_counts

    # ------------------------------------------------------------------
    # Deterministic forward photon pass for crack-free caustics
    # ------------------------------------------------------------------
    def build_caustic_map(self) -> CausticMap:
        scene = self.scene
        assert (
            scene.spectral is not None
            and scene.light_center is not None
            and scene.light_normal is not None
        )
        resolution = self.caustic_map_resolution
        refract = np.zeros((resolution, resolution, 3), dtype=np.float64)
        tir = np.zeros((resolution, resolution, 3), dtype=np.float64)
        photons = self.caustic_photons
        if photons == 0:
            self.caustic_map = CausticMap(
                scene.ground_x0,
                scene.ground_x1,
                scene.ground_z0,
                scene.ground_z1,
                resolution,
                refract,
                tir,
                0,
                0,
                0,
                0,
                3,
            )
            return self.caustic_map

        photon_ids = np.arange(photons, dtype=np.int64)
        photon_area_ids = _photon_permutation(
            self.seed, photon_ids, STREAM_PHOTON_AREA_STRATA
        )
        photon_dir_ids = _photon_permutation(
            self.seed, photon_ids, STREAM_PHOTON_DIR_STRATA
        )
        photon_wave_ids = _photon_permutation(
            self.seed, photon_ids, STREAM_PHOTON_WAVE_STRATA
        )
        rotation_u = uniform(self.seed, 0, 0, STREAM_PHOTON_INIT, 0, 0)
        rotation_v = uniform(self.seed, 0, 0, STREAM_PHOTON_INIT, 0, 1)
        u, v = r2_2d(photon_area_ids, rotation_u, rotation_v)
        assert scene.light_u_hat is not None and scene.light_v_hat is not None
        positions = (
            scene.light_center
            + ((2.0 * u - 1.0) * scene.light_u_half)[:, None] * scene.light_u_hat
            + ((2.0 * v - 1.0) * scene.light_v_half)[:, None] * scene.light_v_hat
        )
        rotation_d1 = uniform(self.seed, 0, 0, STREAM_PHOTON_INIT, 1, 0)
        rotation_d2 = uniform(self.seed, 0, 0, STREAM_PHOTON_INIT, 1, 1)
        d1, d2 = r2_2d(photon_dir_ids, rotation_d1, rotation_d2)
        normals = np.broadcast_to(scene.light_normal, (photons, 3)).copy()
        directions = cosine_hemisphere(normals, d1, d2)

        wavelength_jitter = uniform(self.seed, 0, 0, STREAM_PHOTON_INIT, 2, 0)
        wavelength = scene.min_nm + (
            scene.max_nm - scene.min_nm
        ) * ((photon_wave_ids + wavelength_jitter) / photons)
        rgb_weights = scene.spectral.rgb_weights(wavelength)
        photon_power = (
            scene.light_emission
            * scene.light_area
            * math.pi
            / photons
        )
        throughput = rgb_weights * photon_power

        medium = np.zeros(photons, dtype=np.int8)
        had_glass = np.zeros(photons, dtype=bool)
        had_tir = np.zeros(photons, dtype=bool)
        alive = np.ones(photons, dtype=bool)
        deposited_refract = 0
        deposited_tir = 0
        glass_transmissions = 0
        photon_max_depth = 16

        for bounce in range(photon_max_depth + 1):
            if not np.any(alive):
                break
            distance, kind, primitive = intersect_scene(
                scene, positions, directions, tmin=1.0e-5
            )
            hit = alive & (kind != KIND_NONE)
            if np.any(hit):
                hit_indices = np.nonzero(hit)[0]
                throughput[hit_indices] = _segment_absorption(
                    scene,
                    throughput[hit_indices],
                    medium[hit_indices],
                    wavelength[hit_indices],
                    distance[hit_indices],
                )

            dielectric = hit & ((kind == KIND_TRIANGLE) | (kind == KIND_WATER))
            if np.any(dielectric):
                idx = np.nonzero(dielectric)[0]
                p = positions[idx] + distance[idx, None] * directions[idx]
                branch = uniform(
                    self.seed, photon_ids[idx], bounce, STREAM_PHOTON_PATH, 0, 0
                )
                tri = kind[idx] == KIND_TRIANGLE
                wat = kind[idx] == KIND_WATER
                if np.any(tri):
                    ti = idx[tri]
                    assert scene.triangle_normals is not None
                    outward = scene.triangle_normals[primitive[ti]]
                    target = np.where(
                        medium[ti] == MEDIUM_GLASS, MEDIUM_AIR, MEDIUM_GLASS
                    )
                    new_d, new_m, _, tir_flag = _dielectric_response(
                        scene,
                        directions[ti],
                        outward,
                        medium[ti],
                        target,
                        wavelength[ti],
                        branch[tri],
                        radiance_mode=False,
                    )
                    transmitted = new_m != medium[ti]
                    had_glass[ti[transmitted]] = True
                    glass_transmissions += int(np.count_nonzero(transmitted))
                    # TIR can also happen on reflection; mark it as a TIR path.
                    had_tir[ti[tir_flag]] = True
                    medium[ti] = new_m
                    directions[ti] = new_d
                    positions[ti] = p[tri] + new_d * _EPS
                if np.any(wat):
                    wi = idx[wat]
                    outward = np.broadcast_to(_UP, (wi.size, 3)).copy()
                    target = np.where(
                        medium[wi] == MEDIUM_WATER, MEDIUM_AIR, MEDIUM_WATER
                    )
                    new_d, new_m, _, tir_flag = _dielectric_response(
                        scene,
                        directions[wi],
                        outward,
                        medium[wi],
                        target,
                        wavelength[wi],
                        branch[wat],
                        radiance_mode=False,
                    )
                    had_tir[wi[tir_flag]] = True
                    medium[wi] = new_m
                    directions[wi] = new_d
                    positions[wi] = p[wat] + new_d * _EPS

            ground = hit & (kind == KIND_GROUND)
            if np.any(ground):
                idx = np.nonzero(ground)[0]
                p = positions[idx] + distance[idx, None] * directions[idx]
                valid = had_glass[idx]
                refr_idx = idx[valid & ~had_tir[idx]]
                tir_idx = idx[valid & had_tir[idx]]
                if refr_idx.size:
                    p_refr = positions[refr_idx] + distance[refr_idx, None] * directions[refr_idx]
                    _add_bilinear(
                        refract,
                        p_refr[:, 0],
                        p_refr[:, 2],
                        throughput[refr_idx],
                        scene.ground_x0,
                        scene.ground_x1,
                        scene.ground_z0,
                        scene.ground_z1,
                    )
                    deposited_refract += refr_idx.size
                if tir_idx.size:
                    p_tir = positions[tir_idx] + distance[tir_idx, None] * directions[tir_idx]
                    _add_bilinear(
                        tir,
                        p_tir[:, 0],
                        p_tir[:, 2],
                        throughput[tir_idx],
                        scene.ground_x0,
                        scene.ground_x1,
                        scene.ground_z0,
                        scene.ground_z1,
                    )
                    deposited_tir += tir_idx.size
                alive[idx] = False

            light_hit = hit & (kind == KIND_LIGHT)
            if np.any(light_hit):
                alive[light_hit] = False
            missed = alive & (kind == KIND_NONE)
            alive[missed] = False

        smoothing_passes = 3
        refract = _smooth_map(refract, smoothing_passes)
        tir = _smooth_map(tir, smoothing_passes)
        self.caustic_map = CausticMap(
            scene.ground_x0,
            scene.ground_x1,
            scene.ground_z0,
            scene.ground_z1,
            resolution,
            refract,
            tir,
            photons,
            deposited_refract,
            deposited_tir,
            glass_transmissions,
            smoothing_passes,
        )
        return self.caustic_map

    # ------------------------------------------------------------------
    # Tiles
    # ------------------------------------------------------------------
    def render_tile(self, tile_x: int, tile_y: int) -> dict[str, np.ndarray]:
        scene = self.scene
        x0 = tile_x * self.tile
        y0 = tile_y * self.tile
        x1 = min(x0 + self.tile, self.width)
        y1 = min(y0 + self.tile, self.height)
        px = np.arange(x0, x1, dtype=np.int64)
        py = np.arange(y0, y1, dtype=np.int64)
        grid_x, grid_y = np.meshgrid(px, py)
        flat_x = grid_x.reshape(-1)
        flat_y = grid_y.reshape(-1)
        pixel_count = flat_x.size
        pixel_ids = np.repeat(flat_y * self.width + flat_x, self.spp)
        sample_ids = np.tile(np.arange(self.spp, dtype=np.int64), pixel_count)
        camera_strata_ids = _pixel_sample_permutation(
            self.seed,
            pixel_ids,
            sample_ids,
            pixel_count,
            self.spp,
            STREAM_CAMERA_STRATA,
        )
        light_strata_ids = _pixel_sample_permutation(
            self.seed,
            pixel_ids,
            sample_ids,
            pixel_count,
            self.spp,
            STREAM_LIGHT_STRATA,
        )
        material_strata_ids = _pixel_sample_permutation(
            self.seed,
            pixel_ids,
            sample_ids,
            pixel_count,
            self.spp,
            STREAM_MATERIAL_STRATA,
        )
        wave_strata_ids = _pixel_sample_permutation(
            self.seed,
            pixel_ids,
            sample_ids,
            pixel_count,
            self.spp,
            STREAM_SPECTRAL,
        )

        rotation_x = uniform(self.seed, pixel_ids, 0, STREAM_CAMERA, 0, 0)
        rotation_y = uniform(self.seed, pixel_ids, 0, STREAM_CAMERA, 0, 1)
        jitter_x, jitter_y = r2_2d(camera_strata_ids, rotation_x, rotation_y)
        aspect = self.width / self.height
        ndc_x = (
            (2.0 * (flat_x.repeat(self.spp) + jitter_x) / self.width - 1.0)
            * aspect
            * self.scene.camera_tan_half_fov
        )
        ndc_y = (
            1.0
            - 2.0 * (flat_y.repeat(self.spp) + jitter_y) / self.height
        ) * self.scene.camera_tan_half_fov
        directions = (
            self.scene.camera_forward[None, :]
            + ndc_x[:, None] * self.scene.camera_right[None, :]
            + ndc_y[:, None] * self.scene.camera_true_up[None, :]
        )
        directions = _normalize(directions)
        origins = np.broadcast_to(self.scene.camera_position, directions.shape).copy()

        wavelength_jitter = uniform(self.seed, pixel_ids, 0, STREAM_SPECTRAL, 0, 0)
        stratum = (wave_strata_ids + wavelength_jitter) / self.spp
        wavelength = scene.min_nm + (scene.max_nm - scene.min_nm) * stratum
        assert scene.spectral is not None
        rgb_weights = scene.spectral.rgb_weights(wavelength)

        total, direct, indirect, aux_contribution, aux_counts = self._trace_camera_rays(
            origins,
            directions,
            wavelength,
            rgb_weights,
            pixel_ids,
            sample_ids,
            light_strata_ids,
            material_strata_ids,
        )

        total_pixels = total.reshape(pixel_count, self.spp, 3)
        direct_pixels = direct.reshape(pixel_count, self.spp, 3)
        indirect_pixels = indirect.reshape(pixel_count, self.spp, 3)
        aux_pixels = aux_contribution.reshape(pixel_count, self.spp, 3)
        aux_count_pixels = aux_counts.reshape(pixel_count, self.spp, 3)

        return {
            "sum_rgb": np.sum(total_pixels, axis=1),
            "sum_sq_rgb": np.sum(total_pixels * total_pixels, axis=1),
            "sum_direct_rgb": np.sum(direct_pixels, axis=1),
            "sum_indirect_rgb": np.sum(indirect_pixels, axis=1),
            "aux_contribution": np.sum(aux_pixels, axis=1),
            "aux_hits": np.sum(aux_count_pixels, axis=1),
        }


def build_renderer(
    scene: Scene,
    width: int,
    height: int,
    spp: int,
    seed: int,
    tile: int,
    max_depth: int,
    use_indirect: bool,
    caustic_photons: int,
    caustic_map_resolution: int,
) -> Renderer:
    return Renderer(
        scene,
        width,
        height,
        spp,
        seed,
        tile,
        max_depth,
        use_indirect,
        caustic_photons,
        caustic_map_resolution,
    )

