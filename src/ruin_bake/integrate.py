"""Deterministic one-bounce irradiance integration.

Direct lighting is evaluated analytically from the directional light and the
point lights.  Indirect lighting uses a fixed-count, cosine-weighted
Hammersley sequence in the probe's normal hemisphere.  Every accumulation and
light iteration order is fixed, so a probe is a pure function of
(scene, probe geometry, integration parameters).
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache

from .layout import Probe
from .math3d import (
    Vec3,
    add,
    dot,
    length,
    mul,
    mul_components,
    normalize,
    orthonormal_hemisphere_basis,
    sub,
    trace_boxes,
)
from .scene import DirectionalLight, PointLight, Scene

PI = math.pi
ZERO: Vec3 = (0.0, 0.0, 0.0)


def _add3(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale3(a: Vec3, scale: float) -> Vec3:
    return (a[0] * scale, a[1] * scale, a[2] * scale)


def _radical_inverse_base2(index: int) -> float:
    value = 0.0
    denominator = 2.0
    while index:
        value += (index & 1) / denominator
        index >>= 1
        denominator *= 2.0
    return value


def _probe_shifts(probe: Probe) -> tuple[float, float]:
    digest = hashlib.sha256(probe.key().encode("ascii")).digest()
    shift_u = int.from_bytes(digest[:8], "big") / 18446744073709551616.0
    shift_v = int.from_bytes(digest[8:16], "big") / 18446744073709551616.0
    return shift_u, shift_v


@lru_cache(maxsize=65536)
def _sample_directions_cached(
    probe_key: str,
    position: tuple[float, float, float],
    normal: tuple[float, float, float],
    sample_count: int,
) -> tuple[Vec3, ...]:
    del probe_key
    probe = Probe(probe_id=0, position=position, normal=normal)
    shift_u, shift_v = _probe_shifts(probe)
    tangent, bitangent, normal_vec = orthonormal_hemisphere_basis(normal)
    result: list[Vec3] = []
    for index in range(sample_count):
        u1 = ((index + 0.5) / sample_count + shift_u) % 1.0
        u2 = (_radical_inverse_base2(index + 1) + shift_v) % 1.0
        radius = math.sqrt(u1)
        phi = 2.0 * PI * u2
        local_x = radius * math.cos(phi)
        local_y = radius * math.sin(phi)
        local_z = math.sqrt(max(0.0, 1.0 - u1))
        direction = (
            tangent[0] * local_x + bitangent[0] * local_y + normal_vec[0] * local_z,
            tangent[1] * local_x + bitangent[1] * local_y + normal_vec[1] * local_z,
            tangent[2] * local_x + bitangent[2] * local_y + normal_vec[2] * local_z,
        )
        result.append(direction)
    return tuple(result)


def sample_directions(probe: Probe, sample_count: int) -> tuple[Vec3, ...]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    return _sample_directions_cached(probe.key(), probe.position, probe.normal, sample_count)


def trace_scene(
    scene: Scene,
    origin: Vec3,
    direction: Vec3,
    ray_epsilon: float,
    t_max: float = math.inf,
):
    return trace_boxes(origin, direction, scene.box_tuples(), ray_epsilon, t_max)


def _shadow_origin(point: Vec3, normal: Vec3, ray_epsilon: float, offset_surface: bool) -> Vec3:
    if not offset_surface:
        return point
    return add(point, mul(normal, ray_epsilon))


def light_contribution(
    scene: Scene,
    point: Vec3,
    normal: Vec3,
    light: DirectionalLight | PointLight,
    ray_epsilon: float,
    *,
    offset_surface: bool = False,
) -> Vec3:
    """Per-light direct irradiance contribution.

    This function is also the shared reference used by incremental dependency
    analysis.  Keeping one implementation guarantees bit-identical values.
    """

    origin = _shadow_origin(point, normal, ray_epsilon, offset_surface)
    if isinstance(light, DirectionalLight):
        direction = normalize(light.direction, "directional light direction")
        incoming = (-direction[0], -direction[1], -direction[2])
        cosine = dot(normal, incoming)
        if cosine <= 0.0:
            return ZERO
        if trace_scene(scene, origin, incoming, ray_epsilon) is not None:
            return ZERO
        return _scale3(light.radiance, cosine)

    to_light = sub(light.position, origin)
    distance = length(to_light)
    if distance <= ray_epsilon:
        return ZERO
    incoming = (to_light[0] / distance, to_light[1] / distance, to_light[2] / distance)
    cosine = dot(normal, incoming)
    if cosine <= 0.0:
        return ZERO
    hit = trace_scene(scene, origin, incoming, ray_epsilon, distance - ray_epsilon)
    if hit is not None:
        return ZERO
    attenuation = cosine / (distance * distance)
    return _scale3(light.radiance, attenuation)


def direct_irradiance(
    scene: Scene,
    point: Vec3,
    normal: Vec3,
    ray_epsilon: float,
    *,
    offset_surface: bool = False,
) -> Vec3:
    total = ZERO
    total = _add3(
        total,
        light_contribution(scene, point, normal, scene.directional, ray_epsilon, offset_surface=offset_surface),
    )
    for light in scene.point_lights:
        total = _add3(
            total,
            light_contribution(scene, point, normal, light, ray_epsilon, offset_surface=offset_surface),
        )
    return total


def outgoing_radiance(
    scene: Scene,
    hit,
    incoming_direction: Vec3,
    ray_epsilon: float,
) -> Vec3:
    # The sampled ray points from the probe toward the hit surface.  A visible
    # front face must face back toward the probe.
    toward_probe = (
        -incoming_direction[0],
        -incoming_direction[1],
        -incoming_direction[2],
    )
    if dot(hit.normal, toward_probe) <= 0.0:
        return ZERO

    box = None
    for candidate in scene.geometry:
        if candidate.object_id == hit.object_id:
            box = candidate
            break
    if box is None:
        raise ValueError(f"hit references unknown geometry id {hit.object_id}")
    reflected = _scale3(
        mul_components(
            box.albedo,
            direct_irradiance(
                scene,
                hit.point,
                hit.normal,
                ray_epsilon,
                offset_surface=True,
            ),
        ),
        1.0 / PI,
    )
    return _add3(box.emission, reflected)


def evaluate_probe(scene: Scene, probe: Probe, sample_count: int, ray_epsilon: float) -> Vec3:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not math.isfinite(ray_epsilon) or ray_epsilon <= 0.0:
        raise ValueError("ray_epsilon must be finite and positive")

    direct = direct_irradiance(scene, probe.position, probe.normal, ray_epsilon)
    accumulated = ZERO
    for direction in sample_directions(probe, sample_count):
        hit = trace_scene(scene, probe.position, direction, ray_epsilon)
        if hit is None:
            continue
        accumulated = _add3(accumulated, outgoing_radiance(scene, hit, direction, ray_epsilon))

    # Cosine-weighted pdf is cos/pi, so E = pi * mean(radiance).
    indirect = _scale3(accumulated, PI / sample_count)
    return _add3(direct, indirect)

