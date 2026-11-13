"""Stable uniform-grid probe layout with deterministic culling and ids."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any

from .canonical import SerializationError, sha256_canonical
from .math3d import Vec3, box_signed_distance
from .scene import Scene

LAYOUT_POLICY_VERSION = "centered-grid-z-y-x-cull-v1"


@dataclass(frozen=True, slots=True)
class LayoutParams:
    spacing: float = 2.0
    cull_distance: float = 0.4
    normal_mode: str = "up"

    def __post_init__(self) -> None:
        if not math.isfinite(self.spacing) or self.spacing <= 0.0:
            raise ValueError("layout spacing must be finite and positive")
        if not math.isfinite(self.cull_distance) or self.cull_distance < 0.0:
            raise ValueError("cull_distance must be finite and non-negative")
        if self.normal_mode != "up":
            raise ValueError("normal_mode must be 'up'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": LAYOUT_POLICY_VERSION,
            "spacing": self.spacing,
            "cull_distance": self.cull_distance,
            "normal_mode": self.normal_mode,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "LayoutParams":
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise SerializationError("layout params must be an object")
        allowed = {"spacing", "cull_distance", "normal_mode", "policy_version"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise SerializationError(f"unknown layout parameter keys: {unknown}")
        if "policy_version" in value and value["policy_version"] != LAYOUT_POLICY_VERSION:
            raise SerializationError("unsupported layout policy version")
        defaults = cls()
        return cls(
            spacing=value.get("spacing", defaults.spacing),
            cull_distance=value.get("cull_distance", defaults.cull_distance),
            normal_mode=value.get("normal_mode", defaults.normal_mode),
        )

    def layout_hash(self, bounds_min: Vec3, bounds_max: Vec3) -> str:
        return sha256_canonical(
            {
                "params": self.to_dict(),
                "bounds_min": list(bounds_min),
                "bounds_max": list(bounds_max),
            }
        )


@dataclass(frozen=True, slots=True)
class Probe:
    probe_id: int
    position: Vec3
    normal: Vec3

    def key(self) -> str:
        return struct.pack(
            "<6d",
            self.position[0],
            self.position[1],
            self.position[2],
            self.normal[0],
            self.normal[1],
            self.normal[2],
        ).hex()


@dataclass(frozen=True, slots=True)
class ProbeLayout:
    params: LayoutParams
    probes: tuple[Probe, ...]
    generated_count: int
    culled_count: int
    layout_hash: str


def _axis_centers(low: float, high: float, spacing: float) -> tuple[float, ...]:
    extent = high - low
    count = int(math.floor(extent / spacing + 1.0e-12))
    if count <= 0:
        return ((low + high) * 0.5,)
    return tuple(low + (index + 0.5) * spacing for index in range(count))


def build_probe_layout(scene: Scene, params: LayoutParams | None = None) -> ProbeLayout:
    if params is None:
        params = LayoutParams()

    xs = _axis_centers(scene.bounds_min[0], scene.bounds_max[0], params.spacing)
    ys = _axis_centers(scene.bounds_min[1], scene.bounds_max[1], params.spacing)
    zs = _axis_centers(scene.bounds_min[2], scene.bounds_max[2], params.spacing)
    boxes = scene.box_tuples()

    kept: list[Probe] = []
    generated = 0
    culled = 0
    normal: Vec3 = (0.0, 0.0, 1.0)
    # Stable enumeration: z slowest, y middle, x fastest.  Culling happens
    # before id assignment, so ids are contiguous and independent of workers.
    for z in zs:
        for y in ys:
            for x in xs:
                generated += 1
                position = (x, y, z)
                reject = False
                for _object_id, center, half_size, rotation_deg in boxes:
                    if box_signed_distance(position, center, half_size, rotation_deg) < params.cull_distance:
                        reject = True
                        break
                if reject:
                    culled += 1
                    continue
                kept.append(Probe(probe_id=len(kept), position=position, normal=normal))

    return ProbeLayout(
        params=params,
        probes=tuple(kept),
        generated_count=generated,
        culled_count=culled,
        layout_hash=params.layout_hash(scene.bounds_min, scene.bounds_max),
    )

