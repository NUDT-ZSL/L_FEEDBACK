"""Deterministic full and incremental probe baking."""

from __future__ import annotations

import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, Sequence

from .canonical import sha256_canonical
from .integrate import (
    ZERO,
    direct_irradiance,
    evaluate_probe,
    light_contribution,
    sample_directions,
    trace_scene,
)
from .layout import Probe, ProbeLayout
from .math3d import Vec3, add, dot, mul, normalize, ray_obb_interval, sub
from .scene import (
    DirectionalLight,
    PointLight,
    Scene,
    changed_component_ids,
    geometry_change_kinds,
)
from .snapshot import ProbeRecord, Snapshot

INTEGRATOR_VERSION = "one-bounce-cosine-hammersley-v1"
SAMPLING_VERSION = "hammersley-cranley-patterson-v1"


@dataclass(frozen=True, slots=True)
class BakeParams:
    samples: int = 64
    ray_epsilon: float = 1.0e-6
    workers: int = 1
    chunk_size: int = 16

    def __post_init__(self) -> None:
        if isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples <= 0:
            raise ValueError("samples must be a positive integer")
        if not math.isfinite(self.ray_epsilon) or self.ray_epsilon <= 0.0:
            raise ValueError("ray_epsilon must be finite and positive")
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers <= 0:
            raise ValueError("workers must be a positive integer")
        if self.workers > 256:
            raise ValueError("workers must not exceed 256")
        if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

    def evaluation_dict(self) -> dict[str, object]:
        return {
            "integrator": INTEGRATOR_VERSION,
            "sampling": SAMPLING_VERSION,
            "bounce_depth": 1,
            "samples": self.samples,
            "ray_epsilon": self.ray_epsilon,
            "integration_tolerance": self.ray_epsilon,
            "evaluation_hash": self.evaluation_hash,
        }

    @property
    def evaluation_hash(self) -> str:
        return sha256_canonical(
            {
                "integrator": INTEGRATOR_VERSION,
                "sampling": SAMPLING_VERSION,
                "bounce_depth": 1,
                "samples": self.samples,
                "ray_epsilon": self.ray_epsilon,
                "integration_tolerance": self.ray_epsilon,
            }
        )


@dataclass(frozen=True, slots=True)
class BakeOutput:
    scene: Scene
    layout: ProbeLayout
    irradiances: tuple[Vec3, ...]
    affected_probe_ids: tuple[int, ...]
    reused_probe_count: int
    new_probe_count: int
    reuse_allowed: bool
    changed_geometry_ids: tuple[int, ...]
    changed_light_ids: tuple[str, ...]
    evaluation: dict[str, object]
    elapsed_seconds: float
    chunk_count: int
    workers_used: int

    @property
    def affected_probe_count(self) -> int:
        return len(self.affected_probe_ids)


def _evaluate_chunk(
    scene: Scene,
    probes: tuple[Probe, ...],
    sample_count: int,
    ray_epsilon: float,
) -> tuple[tuple[int, Vec3], ...]:
    return tuple(
        (probe.probe_id, evaluate_probe(scene, probe, sample_count, ray_epsilon))
        for probe in probes
    )


def _make_chunks(probes: Sequence[Probe], chunk_size: int) -> list[tuple[Probe, ...]]:
    return [
        tuple(probes[start : start + chunk_size])
        for start in range(0, len(probes), chunk_size)
    ]


def _evaluate_probes(
    scene: Scene,
    probes: Sequence[Probe],
    params: BakeParams,
) -> tuple[dict[int, Vec3], int, int]:
    if not probes:
        return {}, 0, 0
    chunks = _make_chunks(tuple(sorted(probes, key=lambda probe: probe.probe_id)), params.chunk_size)
    results: dict[int, Vec3] = {}
    workers_used = 1

    if params.workers == 1 or len(chunks) == 1:
        for chunk in chunks:
            for probe_id, value in _evaluate_chunk(scene, chunk, params.samples, params.ray_epsilon):
                results[probe_id] = value
    else:
        workers_used = min(params.workers, len(chunks))
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers_used, mp_context=context) as executor:
            futures = [
                executor.submit(_evaluate_chunk, scene, chunk, params.samples, params.ray_epsilon)
                for chunk in chunks
            ]
            # Completion order is intentionally ignored.  Results are scattered
            # by stable probe id and reduced once in ascending id order.
            for future in as_completed(futures):
                for probe_id, value in future.result():
                    if probe_id in results:
                        raise RuntimeError(f"duplicate worker result for probe {probe_id}")
                    results[probe_id] = value

    expected = {probe.probe_id for probe in probes}
    if set(results) != expected:
        missing = sorted(expected - set(results))
        extra = sorted(set(results) - expected)
        raise RuntimeError(f"worker result mismatch; missing={missing}, extra={extra}")
    return results, len(chunks), workers_used


def _finish_output(
    *,
    scene: Scene,
    layout: ProbeLayout,
    values: list[Vec3 | None],
    affected_ids: Iterable[int],
    reused_count: int,
    new_count: int,
    reuse_allowed: bool,
    changed_geometry_ids: set[int],
    changed_light_ids: set[str],
    params: BakeParams,
    start_time: float,
    chunk_count: int,
    workers_used: int,
) -> BakeOutput:
    ordered_affected = tuple(sorted(affected_ids))
    if any(values[probe_id] is None for probe_id in range(len(values))):
        raise RuntimeError("internal error: some probe irradiances were not produced")
    return BakeOutput(
        scene=scene,
        layout=layout,
        irradiances=tuple(value for value in values if value is not None),
        affected_probe_ids=ordered_affected,
        reused_probe_count=reused_count,
        new_probe_count=new_count,
        reuse_allowed=reuse_allowed,
        changed_geometry_ids=tuple(sorted(changed_geometry_ids)),
        changed_light_ids=tuple(sorted(changed_light_ids)),
        evaluation=params.evaluation_dict(),
        elapsed_seconds=time.perf_counter() - start_time,
        chunk_count=chunk_count,
        workers_used=workers_used,
    )


def bake_full(scene: Scene, layout: ProbeLayout, params: BakeParams) -> BakeOutput:
    start_time = time.perf_counter()
    affected = tuple(layout.probes)
    result_map, chunk_count, workers_used = _evaluate_probes(scene, affected, params)
    values: list[Vec3 | None] = [result_map[probe.probe_id] for probe in layout.probes]
    return _finish_output(
        scene=scene,
        layout=layout,
        values=values,
        affected_ids=(probe.probe_id for probe in layout.probes),
        reused_count=0,
        new_count=len(layout.probes),
        reuse_allowed=False,
        changed_geometry_ids=set(),
        changed_light_ids=set(),
        params=params,
        start_time=start_time,
        chunk_count=chunk_count,
        workers_used=workers_used,
    )


def _light_by_id(scene: Scene, light_id: str) -> DirectionalLight | PointLight | None:
    if scene.directional.light_id == light_id:
        return scene.directional
    for light in scene.point_lights:
        if light.light_id == light_id:
            return light
    return None


def _light_ray(light: DirectionalLight | PointLight, origin: Vec3, ray_epsilon: float) -> tuple[Vec3, float] | None:
    if isinstance(light, DirectionalLight):
        direction = normalize(light.direction, "directional light direction")
        return (-direction[0], -direction[1], -direction[2]), math.inf
    to_light = sub(light.position, origin)
    distance = math.sqrt(dot(to_light, to_light))
    if distance <= ray_epsilon:
        return None
    return (
        (to_light[0] / distance, to_light[1] / distance, to_light[2] / distance),
        distance - ray_epsilon,
    )


def _ray_crosses_any(
    origin: Vec3,
    direction: Vec3,
    t_max: float,
    boxes: Sequence[tuple[int, Vec3, Vec3, Vec3]],
    ray_epsilon: float,
) -> bool:
    for _object_id, center, half_size, rotation_deg in boxes:
        interval = ray_obb_interval(origin, direction, center, half_size, rotation_deg)
        if interval is None:
            continue
        t_enter, t_exit, _entry_normal, _exit_normal = interval
        if t_exit > ray_epsilon and t_enter < t_max:
            return True
    return False


def _changed_boxes(scene: Scene, changed_ids: set[int]) -> tuple[tuple[int, Vec3, Vec3, Vec3], ...]:
    return tuple(
        (box.object_id, box.center, box.half_size, box.rotation_deg)
        for box in scene.geometry
        if box.object_id in changed_ids
    )


def _geometry_affects_probe(
    old_scene: Scene,
    new_scene: Scene,
    probe: Probe,
    changed_ids: set[int],
    shape_changed_ids: set[int],
    params: BakeParams,
) -> bool:
    old_changed = _changed_boxes(old_scene, shape_changed_ids)
    new_changed = _changed_boxes(new_scene, shape_changed_ids)

    # Direct visibility can change whenever a changed box crosses a
    # probe-to-light segment in either the old or new scene.
    for light in (old_scene.directional, *old_scene.point_lights):
        ray = _light_ray(light, probe.position, params.ray_epsilon)
        if ray is None:
            continue
        direction, t_max = ray
        if _ray_crosses_any(probe.position, direction, t_max, old_changed, params.ray_epsilon):
            return True
        if _ray_crosses_any(probe.position, direction, t_max, new_changed, params.ray_epsilon):
            return True

    for direction in sample_directions(probe, params.samples):
        old_hit = trace_scene(old_scene, probe.position, direction, params.ray_epsilon)
        new_hit = trace_scene(new_scene, probe.position, direction, params.ray_epsilon)
        old_id = None if old_hit is None else old_hit.object_id
        new_id = None if new_hit is None else new_hit.object_id
        if old_id != new_id:
            return True
        if new_hit is None:
            continue
        if new_id in changed_ids:
            return True

        # The nearest visible surface is unchanged.  Its reflected direct
        # light can still change if a changed box crosses one of its shadow
        # rays.
        shadow_origin = add(new_hit.point, mul(new_hit.normal, params.ray_epsilon))
        for light in (old_scene.directional, *old_scene.point_lights):
            ray = _light_ray(light, shadow_origin, params.ray_epsilon)
            if ray is None:
                continue
            light_direction, t_max = ray
            if _ray_crosses_any(shadow_origin, light_direction, t_max, old_changed, params.ray_epsilon):
                return True
            if _ray_crosses_any(shadow_origin, light_direction, t_max, new_changed, params.ray_epsilon):
                return True
    return False


def _material_affects_probe(
    old_scene: Scene,
    probe: Probe,
    changed_ids: set[int],
    params: BakeParams,
) -> bool:
    """Material-only changes cannot alter visibility; check actual hits."""

    changed_boxes = _changed_boxes(old_scene, changed_ids)
    for direction in sample_directions(probe, params.samples):
        # Cheap reject before performing the full nearest-hit query.
        if not _ray_crosses_any(
            probe.position,
            direction,
            math.inf,
            changed_boxes,
            params.ray_epsilon,
        ):
            continue
        hit = trace_scene(old_scene, probe.position, direction, params.ray_epsilon)
        if hit is not None and hit.object_id in changed_ids:
            return True
    return False


def _changed_light_contribution(
    old_scene: Scene,
    new_scene: Scene,
    point: Vec3,
    normal: Vec3,
    light_id: str,
    params: BakeParams,
    *,
    offset_surface: bool,
) -> bool:
    old_light = _light_by_id(old_scene, light_id)
    new_light = _light_by_id(new_scene, light_id)
    old_value = (
        ZERO
        if old_light is None
        else light_contribution(
            old_scene,
            point,
            normal,
            old_light,
            params.ray_epsilon,
            offset_surface=offset_surface,
        )
    )
    new_value = (
        ZERO
        if new_light is None
        else light_contribution(
            new_scene,
            point,
            normal,
            new_light,
            params.ray_epsilon,
            offset_surface=offset_surface,
        )
    )
    return old_value != new_value


def _lights_affect_probe(
    old_scene: Scene,
    new_scene: Scene,
    probe: Probe,
    changed_ids: set[str],
    params: BakeParams,
) -> bool:
    for light_id in sorted(changed_ids):
        if _changed_light_contribution(
            old_scene,
            new_scene,
            probe.position,
            probe.normal,
            light_id,
            params,
            offset_surface=False,
        ):
            return True

    for direction in sample_directions(probe, params.samples):
        hit = trace_scene(old_scene, probe.position, direction, params.ray_epsilon)
        if hit is None:
            continue
        toward_probe = (-direction[0], -direction[1], -direction[2])
        if dot(hit.normal, toward_probe) <= 0.0:
            continue
        shadow_origin = add(hit.point, mul(hit.normal, params.ray_epsilon))
        for light_id in sorted(changed_ids):
            if _changed_light_contribution(
                old_scene,
                new_scene,
                shadow_origin,
                hit.normal,
                light_id,
                params,
                offset_surface=False,
            ):
                return True
    return False


def bake_incremental(
    scene: Scene,
    layout: ProbeLayout,
    params: BakeParams,
    base: Snapshot | None,
) -> BakeOutput:
    if base is None:
        result = bake_full(scene, layout, params)
        return result

    start_time = time.perf_counter()
    old_evaluation = base.header.get("evaluation", {})
    reuse_allowed = (
        isinstance(old_evaluation, dict)
        and old_evaluation.get("evaluation_hash") == params.evaluation_hash
        and base.layout_hash == layout.layout_hash
    )

    changed_geometry_ids, changed_light_ids = changed_component_ids(base.scene, scene)
    _material_changed_ids, shape_changed_ids = geometry_change_kinds(base.scene, scene)
    both_categories_changed = bool(changed_geometry_ids) and bool(changed_light_ids)
    old_by_key = base.probes_by_key()
    affected_ids: set[int] = set()
    values: list[Vec3 | None] = [None] * len(layout.probes)
    reused_count = 0
    new_count = 0

    for probe in layout.probes:
        record = old_by_key.get(probe.key())
        if record is None:
            new_count += 1
            affected_ids.add(probe.probe_id)
            continue
        if not reuse_allowed or both_categories_changed:
            affected_ids.add(probe.probe_id)
            continue

        if changed_geometry_ids:
            if shape_changed_ids:
                affected = _geometry_affects_probe(
                    base.scene,
                    scene,
                    probe,
                    changed_geometry_ids,
                    shape_changed_ids,
                    params,
                )
            else:
                affected = _material_affects_probe(
                    base.scene,
                    probe,
                    changed_geometry_ids,
                    params,
                )
            if affected:
                affected_ids.add(probe.probe_id)
                continue
        elif changed_light_ids:
            affected = _lights_affect_probe(
                base.scene,
                scene,
                probe,
                changed_light_ids,
                params,
            )
            if affected:
                affected_ids.add(probe.probe_id)
                continue

        values[probe.probe_id] = record.irradiance
        reused_count += 1

    probes_to_bake = tuple(layout.probes[probe_id] for probe_id in sorted(affected_ids))
    result_map, chunk_count, workers_used = _evaluate_probes(scene, probes_to_bake, params)
    for probe_id, value in result_map.items():
        values[probe_id] = value

    return _finish_output(
        scene=scene,
        layout=layout,
        values=values,
        affected_ids=affected_ids,
        reused_count=reused_count,
        new_count=new_count,
        reuse_allowed=reuse_allowed,
        changed_geometry_ids=changed_geometry_ids,
        changed_light_ids=changed_light_ids,
        params=params,
        start_time=start_time,
        chunk_count=chunk_count,
        workers_used=workers_used,
    )

