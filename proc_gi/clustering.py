from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

from .config import ClusteringConfig, Light
from .mathutil import normalize, ray_segment_intersects_triangles, sphere_frustum_factor


def _quantized_vector(vector: Sequence[float], places: int = 5) -> tuple[float, ...]:
    return tuple(round(float(value), places) for value in vector)


def _member_id(member_ids: Iterable[str]) -> str:
    digest = hashlib.sha1("\x1f".join(sorted(member_ids)).encode("utf-8")).hexdigest()
    return f"vpl-{digest[:12]}"


@dataclass(frozen=True)
class VirtualPointLight:
    id: str
    position: tuple[float, float, float]
    color: tuple[float, float, float]
    intensity: float
    direction: tuple[float, float, float]
    direction_range_deg: float
    is_spot: bool
    range: float
    radius: float
    member_ids: tuple[str, ...]
    source_count: int
    signature: str

    @property
    def luminance(self) -> float:
        return float(np.dot(np.asarray(self.color), (0.2126, 0.7152, 0.0722)))

    @property
    def importance(self) -> float:
        return max(0.0, self.intensity) * self.luminance

    @property
    def cone_cos_outer(self) -> float:
        if not self.is_spot:
            return -1.0
        return float(np.cos(np.deg2rad(min(89.5, self.direction_range_deg))))

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "position": list(self.position),
            "color": list(self.color),
            "intensity": self.intensity,
            "direction": list(self.direction),
            "direction_range_deg": self.direction_range_deg,
            "is_spot": self.is_spot,
            "range": self.range,
            "radius": self.radius,
            "member_ids": list(self.member_ids),
            "source_count": self.source_count,
            "signature": self.signature,
        }


def build_vpl(lights: Sequence[Light]) -> VirtualPointLight:
    if not lights:
        raise ValueError("cannot build a VPL from zero lights")
    weights = np.asarray([max(1.0e-6, light.intensity) for light in lights], dtype=np.float64)
    total_weight = float(weights.sum())
    positions = np.asarray([light.position for light in lights], dtype=np.float64)
    colors = np.asarray([light.color for light in lights], dtype=np.float64)
    position = np.sum(positions * weights[:, None], axis=0) / total_weight
    color_energy = np.sum(colors * weights[:, None], axis=0) / total_weight
    directions = np.asarray(
        [
            normalize(np.asarray(light.direction), fallback=(0.0, -1.0, 0.0))
            if light.type == "spot"
            else np.asarray((0.0, -1.0, 0.0))
            for light in lights
        ]
    )
    direction = normalize(np.sum(directions * weights[:, None], axis=0), fallback=(0.0, -1.0, 0.0))
    angle_deviations = []
    for member_direction in directions:
        angle_deviations.append(
            math.degrees(math.acos(float(np.clip(np.dot(direction, member_direction), -1.0, 1.0))))
        )
    all_spots = all(light.type == "spot" for light in lights)
    spot_outer = max(
        (float(light.spot_angle_deg) for light in lights if light.type == "spot"),
        default=0.0,
    )
    direction_range = max(angle_deviations, default=0.0)
    if all_spots:
        direction_range += spot_outer
    distances = np.linalg.norm(positions - position[None, :], axis=1)
    cluster_radius = float(distances.max()) + 0.35
    light_range = max(float(light.range) for light in lights)
    vpl_range = max(light_range + float(distances.max()), 0.75)
    member_ids = tuple(sorted(light.id for light in lights))
    quantized_payload = (
        _quantized_vector(position, 4),
        _quantized_vector(color_energy, 4),
        round(total_weight, 4),
        _quantized_vector(direction, 4),
        round(direction_range, 3),
        all_spots,
        round(vpl_range, 4),
        member_ids,
    )
    signature = hashlib.sha1(repr(quantized_payload).encode("utf-8")).hexdigest()
    return VirtualPointLight(
        id=_member_id(member_ids),
        position=tuple(float(value) for value in position),
        color=tuple(float(value) for value in color_energy),
        intensity=total_weight,
        direction=tuple(float(value) for value in direction),
        direction_range_deg=float(direction_range),
        is_spot=all_spots,
        range=float(vpl_range),
        radius=float(cluster_radius),
        member_ids=member_ids,
        source_count=len(lights),
        signature=signature,
    )


def _orientation_weight(config: ClusteringConfig) -> float:
    return max(0.0001, config.orientation_weight) * 2.2


def _feature(light: Light, config: ClusteringConfig) -> np.ndarray:
    direction = (
        normalize(np.asarray(light.direction), fallback=(0.0, -1.0, 0.0))
        if light.type == "spot"
        else np.asarray((0.0, -1.0, 0.0))
    )
    return np.concatenate(
        [
            np.asarray(light.position, dtype=np.float64) * config.spatial_weight,
            direction * _orientation_weight(config),
        ]
    )


def _weighted_distance_sq(feature: np.ndarray, center: np.ndarray) -> float:
    delta = feature - center
    return float(np.dot(delta, delta))


def _initial_centers(
    lights: Sequence[Light],
    features: np.ndarray,
    count: int,
    config: ClusteringConfig,
) -> np.ndarray:
    importance = np.asarray([light.importance for light in lights], dtype=np.float64)
    first = int(np.lexsort((np.asarray([light.id for light in lights]), -importance))[0])
    chosen = [first]
    closest = np.full(len(lights), np.inf, dtype=np.float64)
    for _ in range(1, count):
        distances = np.asarray(
            [_weighted_distance_sq(feature, features[chosen[-1]]) for feature in features]
        )
        closest = np.minimum(closest, distances)
        weighted = closest * np.maximum(importance, 1.0e-6)
        weighted[chosen] = -1.0
        chosen.append(int(np.argmax(weighted)))
    return features[chosen].copy()


def _assign(
    lights: Sequence[Light],
    features: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    distance_sq = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(distance_sq, axis=1)


def _recompute_centers(
    lights: Sequence[Light],
    features: np.ndarray,
    assignments: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    updated = centers.copy()
    weights = np.asarray([max(1.0e-6, light.intensity) for light in lights], dtype=np.float64)
    for cluster in range(len(centers)):
        members = assignments == cluster
        if not np.any(members):
            farthest_light = int(np.argmax(np.min(np.sum((features[:, None, :] - centers) ** 2, axis=2), axis=1)))
            updated[cluster] = features[farthest_light]
            continue
        cluster_weights = weights[members]
        updated[cluster] = np.sum(features[members] * cluster_weights[:, None], axis=0) / float(
            cluster_weights.sum()
        )
    return updated


def cluster_lights(
    lights: Sequence[Light],
    config: ClusteringConfig,
    *,
    previous: Sequence[VirtualPointLight] | None = None,
    changed_ids: Iterable[str] | None = None,
) -> list[VirtualPointLight]:
    """Cluster lights, preserving untouched clusters when previous state is supplied."""
    if not lights:
        return []
    if len(lights) > config.max_lights:
        raise ValueError("light count exceeds clustering.max_lights")
    lights = sorted(lights, key=lambda light: light.id)
    current_by_id = {light.id: light for light in lights}
    changed = set(changed_ids or ())
    clean_clusters: list[VirtualPointLight] = []
    if previous and changed_ids is not None:
        for vpl in previous:
            if all(member_id in current_by_id for member_id in vpl.member_ids) and not any(
                member_id in changed for member_id in vpl.member_ids
            ):
                clean_clusters.append(vpl)
    clean_light_ids = {
        member_id for vpl in clean_clusters for member_id in vpl.member_ids
    }
    affected_lights = [light for light in lights if light.id not in clean_light_ids]
    available_slots = max(0, config.max_vpls - len(clean_clusters))
    if not affected_lights or available_slots == 0:
        result = list(clean_clusters)
    else:
        average_size = (
            len(lights) / max(1, len(previous))
            if previous
            else len(lights) / min(config.max_vpls, len(lights))
        )
        cluster_count = max(1, int(round(len(affected_lights) / max(1.0, average_size))))
        cluster_count = min(cluster_count, available_slots, len(affected_lights))
        features = np.asarray([_feature(light, config) for light in affected_lights])
        centers = _initial_centers(affected_lights, features, cluster_count, config)
        assignments = _assign(affected_lights, features, centers)
        for _ in range(max(1, config.iterations)):
            centers = _recompute_centers(affected_lights, features, assignments, centers)
            new_assignments = _assign(affected_lights, features, centers)
            if np.array_equal(new_assignments, assignments):
                break
            assignments = new_assignments
        clusters = []
        for index in range(cluster_count):
            members = [
                light
                for light, assignment in zip(affected_lights, assignments)
                if int(assignment) == index
            ]
            if members:
                clusters.append(build_vpl(members))
        result = clean_clusters + clusters
    result.sort(key=lambda vpl: vpl.id)
    return result


@dataclass(frozen=True)
class ImportanceResult:
    accepted: list[VirtualPointLight]
    rejected: list[VirtualPointLight]
    scores: dict[str, dict[str, float]]


def rank_vpl_importance(
    vpls: Sequence[VirtualPointLight],
    *,
    capacity: int,
    view_proj: np.ndarray,
    camera_position: np.ndarray,
    scene_triangles: np.ndarray,
) -> ImportanceResult:
    scores: dict[str, dict[str, float]] = {}
    camera = np.asarray(camera_position, dtype=np.float64)
    for vpl in vpls:
        frustum = sphere_frustum_factor(view_proj, np.asarray(vpl.position), vpl.radius)
        occluded = ray_segment_intersects_triangles(
            camera,
            np.asarray(vpl.position),
            scene_triangles,
            epsilon=0.02,
        )
        occlusion_rate = 1.0 if occluded else 0.0
        intensity = max(0.0, vpl.intensity) * max(1.0e-6, vpl.luminance)
        score = intensity * frustum * (1.0 - 0.65 * occlusion_rate)
        scores[vpl.id] = {
            "score": float(score),
            "intensity": float(vpl.intensity),
            "frustum_factor": float(frustum),
            "occlusion_rate": float(occlusion_rate),
        }
    ordered = sorted(vpls, key=lambda vpl: (-scores[vpl.id]["score"], vpl.id))
    accepted_ids = {vpl.id for vpl in ordered[: max(0, capacity)]}
    accepted = sorted([vpl for vpl in vpls if vpl.id in accepted_ids], key=lambda vpl: vpl.id)
    rejected = sorted([vpl for vpl in vpls if vpl.id not in accepted_ids], key=lambda vpl: vpl.id)
    return ImportanceResult(accepted=accepted, rejected=rejected, scores=scores)


def truncation_warning(
    requested_count: int,
    result: ImportanceResult,
    capacity: int,
) -> dict[str, object]:
    source_ids = sorted(
        {
            member_id
            for vpl in result.rejected
            for member_id in vpl.member_ids
        }
    )
    return {
        "code": "VPL_CAPACITY_TRUNCATED",
        "severity": "warning",
        "message": "Virtual point light budget exceeded; lowest-importance VPLs were truncated.",
        "context": {
            "requested_vpl_count": requested_count,
            "accepted_vpl_count": len(result.accepted),
            "capacity": capacity,
            "truncated_vpl_ids": [vpl.id for vpl in result.rejected],
            "truncated_source_light_ids": source_ids,
            "importance": {
                vpl.id: result.scores.get(vpl.id, {}) for vpl in result.rejected
            },
            "degraded_state": {
                "render_continues": True,
                "lighting_mode": "importance_truncated_vpl_accumulation",
            },
        },
    }


def replace_vpl_signature(vpl: VirtualPointLight, signature: str) -> VirtualPointLight:
    return replace(vpl, signature=signature)

