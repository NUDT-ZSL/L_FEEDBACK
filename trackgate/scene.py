"""Deterministic procedural scene generator for track gate segments.

Same (seed, num_segments) always produces byte-identical canonical JSON.
Segments may share material *ids* (a shared immutable material table), but
every segment carries its own immutable parameter tuple; no mutable state
is ever shared between segments.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from typing import Tuple

from .errors import EmptySceneError

Color = Tuple[int, int, int]

# Fixed palette keeps stripe colors deterministic and visually distinct.
_STRIPE_PALETTE: Tuple[Color, ...] = (
    (255, 64, 32),
    (32, 200, 255),
    (255, 220, 40),
    (120, 255, 90),
    (255, 80, 200),
    (180, 140, 255),
)


@dataclass(frozen=True)
class Material:
    material_id: int
    base_color: Color


@dataclass(frozen=True)
class StripeParams:
    """Direction-indicator light band, parameterized by arc length."""

    count: int          # number of stripe periods along the arch arc
    duty: float         # fraction of each period that is lit, in (0, 1)
    color: Color        # lit-stripe color


@dataclass(frozen=True)
class Segment:
    index: int
    translation: Tuple[float, float, float]
    yaw_deg: float
    arch_height: float
    arch_width: float
    thickness: float
    depth: float
    material_id: int
    stripe: StripeParams


@dataclass(frozen=True)
class Scene:
    seed: int
    materials: Tuple[Material, ...]
    segments: Tuple[Segment, ...]

    def canonical_dict(self) -> dict:
        return {
            "seed": self.seed,
            "materials": [
                {"material_id": m.material_id, "base_color": list(m.base_color)}
                for m in self.materials
            ],
            "segments": [
                {
                    "index": s.index,
                    "translation": [round(v, 9) for v in s.translation],
                    "yaw_deg": round(s.yaw_deg, 9),
                    "arch_height": round(s.arch_height, 9),
                    "arch_width": round(s.arch_width, 9),
                    "thickness": round(s.thickness, 9),
                    "depth": round(s.depth, 9),
                    "material_id": s.material_id,
                    "stripe": {
                        "count": s.stripe.count,
                        "duty": round(s.stripe.duty, 9),
                        "color": list(s.stripe.color),
                    },
                }
                for s in self.segments
            ],
        }

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def version(self) -> str:
        return hashlib.sha1(self.canonical_json().encode("utf-8")).hexdigest()[:16]

    def material_color(self, material_id: int) -> Color:
        for m in self.materials:
            if m.material_id == material_id:
                return m.base_color
        raise KeyError(f"unknown material_id {material_id}")

    def mutate_segment(self, index: int, **fields) -> "Scene":
        """Return a new Scene with one segment replaced (new version id)."""
        segs = list(self.segments)
        if not (0 <= index < len(segs)):
            raise KeyError(f"segment index {index} out of range")
        segs[index] = replace(segs[index], **fields)
        return Scene(seed=self.seed, materials=self.materials, segments=tuple(segs))

    def mutate_material(self, material_id: int, base_color: Color) -> "Scene":
        """Return a new Scene with a shared material recolored."""
        mats = tuple(
            replace(m, base_color=tuple(base_color)) if m.material_id == material_id else m
            for m in self.materials
        )
        return Scene(seed=self.seed, materials=mats, segments=self.segments)


def generate_scene(seed: int, num_segments: int) -> Scene:
    """Build a deterministic scene. Raises EmptySceneError for num_segments <= 0."""
    if num_segments <= 0:
        raise EmptySceneError(
            "scene must contain at least one gate segment",
            seed=seed,
            num_segments=num_segments,
        )
    rng = random.Random(seed)

    num_materials = 3
    materials = tuple(
        Material(
            material_id=k,
            base_color=(
                rng.randrange(40, 220),
                rng.randrange(40, 220),
                rng.randrange(40, 220),
            ),
        )
        for k in range(num_materials)
    )

    segments = []
    for i in range(num_segments):
        translation = (
            round(i * 9.0 + rng.uniform(-1.5, 1.5), 6),
            0.0,
            round(rng.uniform(-2.0, 2.0), 6),
        )
        stripe = StripeParams(
            count=rng.randrange(4, 13),
            duty=round(rng.uniform(0.3, 0.7), 6),
            color=_STRIPE_PALETTE[rng.randrange(len(_STRIPE_PALETTE))],
        )
        segments.append(
            Segment(
                index=i,
                translation=translation,
                yaw_deg=round(rng.uniform(-18.0, 18.0), 6),
                arch_height=round(rng.uniform(3.0, 5.0), 6),
                arch_width=round(rng.uniform(4.0, 6.0), 6),
                thickness=round(rng.uniform(0.30, 0.60), 6),
                depth=round(rng.uniform(0.5, 1.0), 6),
                material_id=rng.randrange(num_materials),
                stripe=stripe,
            )
        )
    return Scene(seed=seed, materials=materials, segments=tuple(segments))
