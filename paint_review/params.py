from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Dict, Tuple


ColorRGB = Tuple[float, float, float]


def _q(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


@dataclass(frozen=True)
class PaintParams:
    base_color: ColorRGB = (0.32, 0.13, 0.10)
    metallic: float = 0.0
    roughness: float = 0.28
    clearcoat_thickness: float = 1.0
    grain_density: float = 0.0
    pearl_shift: float = 0.0
    exposure: float = 1.35
    grain_seed: int = 107
    paint_type: str = "solid"

    def normalized(self) -> "PaintParams":
        color = tuple(min(1.0, max(0.0, _q(c))) for c in self.base_color)
        return replace(
            self,
            base_color=color,  # type: ignore[arg-type]
            metallic=_q(min(1.0, max(0.0, self.metallic))),
            roughness=_q(min(1.0, max(0.005, self.roughness))),
            clearcoat_thickness=_q(min(4.0, max(0.0, self.clearcoat_thickness))),
            grain_density=_q(min(1.0, max(0.0, self.grain_density))),
            pearl_shift=_q(min(1.0, max(-1.0, self.pearl_shift))),
            exposure=_q(min(8.0, max(0.05, self.exposure))),
            grain_seed=int(self.grain_seed) & 0xFFFFFFFF,
            paint_type=self.paint_type
            if self.paint_type in ("solid", "metallic", "pearl")
            else "solid",
        )

    def canonical_dict(self) -> Dict[str, object]:
        p = self.normalized()
        return {
            "base_color": [p.base_color[0], p.base_color[1], p.base_color[2]],
            "clearcoat_thickness": p.clearcoat_thickness,
            "exposure": p.exposure,
            "grain_density": p.grain_density,
            "grain_seed": p.grain_seed,
            "metallic": p.metallic,
            "paint_type": p.paint_type,
            "pearl_shift": p.pearl_shift,
            "roughness": p.roughness,
        }

    def material_hash(self) -> str:
        payload = {
            "schema": "layered-paint-material-v1",
            "params": self.canonical_dict(),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OrbitCamera:
    yaw_deg: float = 0.0
    elevation_deg: float = 38.0
    distance: float = 4.15
    fov_deg: float = 36.0
    target: Tuple[float, float, float] = (0.0, 0.08, 0.03)

    def normalized(self) -> "OrbitCamera":
        return replace(
            self,
            yaw_deg=_q(self.yaw_deg, 3) % 360.0,
            elevation_deg=_q(min(86.0, max(4.0, self.elevation_deg)), 3),
            distance=_q(min(8.0, max(2.55, self.distance)), 3),
            fov_deg=_q(min(70.0, max(18.0, self.fov_deg)), 2),
        )

    def camera_hash(self) -> str:
        c = self.normalized()
        payload = {
            "distance": c.distance,
            "elevation_deg": c.elevation_deg,
            "fov_deg": c.fov_deg,
            "target": [round(v, 4) for v in c.target],
            "yaw_deg": c.yaw_deg,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def render_hash(params: PaintParams, camera: OrbitCamera, width: int, height: int) -> str:
    payload = {
        "camera": camera.normalized().camera_hash(),
        "height": int(height),
        "material": params.normalized().material_hash(),
        "schema": "layered-paint-render-v1",
        "width": int(width),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def camera_from_orbit(
    yaw_deg: float,
    elevation_deg: float,
    distance: float,
    fov_deg: float = 36.0,
) -> OrbitCamera:
    return OrbitCamera(yaw_deg, elevation_deg, distance, fov_deg).normalized()


def camera_eye(camera: OrbitCamera) -> Tuple[float, float, float]:
    c = camera.normalized()
    yaw = math.radians(c.yaw_deg)
    elev = math.radians(c.elevation_deg)
    tx, ty, tz = c.target
    return (
        tx + c.distance * math.cos(elev) * math.sin(yaw),
        ty + c.distance * math.sin(elev),
        tz + c.distance * math.cos(elev) * math.cos(yaw),
    )


PRESETS: Dict[str, Dict[str, object]] = {
    "素色漆": {
        "params": PaintParams(
            base_color=(0.42, 0.055, 0.035),
            metallic=0.0,
            roughness=0.24,
            clearcoat_thickness=1.45,
            grain_density=0.0,
            pearl_shift=0.0,
            exposure=1.25,
            grain_seed=107,
            paint_type="solid",
        )
    },
    "金属漆": {
        "params": PaintParams(
            base_color=(0.12, 0.27, 0.52),
            metallic=0.88,
            roughness=0.22,
            clearcoat_thickness=1.10,
            grain_density=0.42,
            pearl_shift=0.0,
            exposure=1.30,
            grain_seed=211,
            paint_type="metallic",
        )
    },
    "珠光漆": {
        "params": PaintParams(
            base_color=(0.48, 0.075, 0.16),
            metallic=0.42,
            roughness=0.18,
            clearcoat_thickness=1.25,
            grain_density=0.36,
            pearl_shift=0.32,
            exposure=1.32,
            grain_seed=503,
            paint_type="pearl",
        )
    },
}


def preset_params(name: str) -> PaintParams:
    if name not in PRESETS:
        raise KeyError("Unknown preset: {}".format(name))
    params = PRESETS[name]["params"]
    assert isinstance(params, PaintParams)
    return params

