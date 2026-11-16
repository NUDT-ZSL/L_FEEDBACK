from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LOGGER = logging.getLogger("proc_gi")

KNOWN_MATERIAL_TYPES = {
    "Lambertian": 0,
    "OrenNayar": 1,
    "BlinnPhong": 2,
}
MATERIAL_FALLBACK_TYPE = "Lambertian"
MAX_LIGHTS = 256


def _as_float3(values: Iterable[float], name: str) -> tuple[float, float, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite numbers")
    return tuple(float(value) for value in array)


def _luminance(color: Iterable[float]) -> float:
    rgb = np.asarray(tuple(color), dtype=np.float64)
    return float(np.dot(rgb, (0.2126, 0.7152, 0.0722)))


@dataclass
class DiagnosticLog:
    records: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        code: str,
        message: str,
        *,
        severity: str = "warning",
        context: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "code": code,
            "severity": severity,
            "message": message,
            "context": context or {},
        }
        if record not in self.records:
            self.records.append(record)
        LOGGER.warning("[BUSINESS:%s] %s %s", code, message, record["context"])

    def extend(self, records: Iterable[dict[str, Any]]) -> None:
        for record in records:
            self.add(
                str(record.get("code", "UNKNOWN")),
                str(record.get("message", "")),
                severity=str(record.get("severity", "warning")),
                context=dict(record.get("context", {})),
            )

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "degraded" if self.records else "ok",
            "warning_count": len(self.records),
            "warnings": self.records,
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass
class Material:
    id: int
    name: str
    type: str
    albedo: tuple[float, float, float]
    roughness: float = 0.8
    specular: float = 0.02
    shininess: float = 32.0
    original_type: str | None = None

    @property
    def shader_type(self) -> int:
        return KNOWN_MATERIAL_TYPES[self.type]

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        index: int,
        diagnostics: DiagnosticLog | None = None,
    ) -> "Material":
        material_id = int(payload.get("id", index))
        requested_type = str(payload.get("type", "Lambertian"))
        original_type = None
        if requested_type not in KNOWN_MATERIAL_TYPES:
            original_type = requested_type
            if diagnostics is not None:
                diagnostics.add(
                    "MAT_UNKNOWN_TYPE",
                    "Unknown material type; rendering continues with Lambertian fallback.",
                    context={
                        "material_id": material_id,
                        "material_name": str(payload.get("name", f"material-{index}")),
                        "unknown_type": requested_type,
                        "fallback_type": MATERIAL_FALLBACK_TYPE,
                    },
                )
            requested_type = MATERIAL_FALLBACK_TYPE
        return cls(
            id=material_id,
            name=str(payload.get("name", f"material-{index}")),
            type=requested_type,
            albedo=_as_float3(payload.get("albedo", (0.7, 0.7, 0.7)), "albedo"),
            roughness=float(np.clip(payload.get("roughness", 0.8), 0.0, 1.0)),
            specular=float(np.clip(payload.get("specular", 0.02), 0.0, 1.0)),
            shininess=max(1.0, float(payload.get("shininess", 32.0))),
            original_type=original_type,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "albedo": list(self.albedo),
            "roughness": self.roughness,
            "specular": self.specular,
            "shininess": self.shininess,
        }


@dataclass
class Light:
    id: str
    type: str
    position: tuple[float, float, float]
    direction: tuple[float, float, float]
    color: tuple[float, float, float]
    intensity: float
    range: float = 6.0
    spot_angle_deg: float = 35.0
    spot_softness_deg: float = 12.0

    @property
    def luminance(self) -> float:
        return _luminance(self.color)

    @property
    def importance(self) -> float:
        return max(0.0, self.intensity) * self.luminance

    @classmethod
    def from_dict(cls, payload: dict[str, Any], index: int) -> "Light":
        light_type = str(payload.get("type", "point")).lower()
        if light_type not in {"point", "spot"}:
            raise ValueError(f"light {index} has unsupported type {light_type!r}")
        intensity = float(payload.get("intensity", 1.0))
        light_range = float(payload.get("range", 6.0))
        if intensity < 0.0 or light_range <= 0.0:
            raise ValueError(f"light {index} has invalid intensity/range")
        return cls(
            id=str(payload.get("id", f"light-{index:03d}")),
            type=light_type,
            position=_as_float3(payload["position"], "light.position"),
            direction=_as_float3(payload.get("direction", (0.0, -1.0, 0.0)), "light.direction"),
            color=_as_float3(payload.get("color", (1.0, 1.0, 1.0)), "light.color"),
            intensity=intensity,
            range=light_range,
            spot_angle_deg=float(payload.get("spot_angle_deg", 35.0)),
            spot_softness_deg=float(payload.get("spot_softness_deg", 12.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "position": list(self.position),
            "direction": list(self.direction),
            "color": list(self.color),
            "intensity": self.intensity,
            "range": self.range,
            "spot_angle_deg": self.spot_angle_deg,
            "spot_softness_deg": self.spot_softness_deg,
        }


@dataclass(frozen=True)
class ClusteringConfig:
    max_vpls: int = 16
    max_lights: int = MAX_LIGHTS
    spatial_weight: float = 1.0
    orientation_weight: float = 0.75
    iterations: int = 12
    cube_size: int = 128
    tile_size: int = 16
    vpl_capacity: int = 16

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClusteringConfig":
        config = cls(
            max_vpls=int(payload.get("max_vpls", 16)),
            max_lights=int(payload.get("max_lights", MAX_LIGHTS)),
            spatial_weight=float(payload.get("spatial_weight", 1.0)),
            orientation_weight=float(payload.get("orientation_weight", 0.75)),
            iterations=int(payload.get("iterations", 12)),
            cube_size=int(payload.get("cube_size", 128)),
            tile_size=int(payload.get("tile_size", 16)),
            vpl_capacity=int(payload.get("vpl_capacity", payload.get("max_vpls", 16))),
        )
        if config.max_vpls < 1 or config.max_vpls > 64:
            raise ValueError("clustering.max_vpls must be in [1, 64]")
        if config.max_lights < 1 or config.max_lights > MAX_LIGHTS:
            raise ValueError("clustering.max_lights must be in [1, 256]")
        if config.vpl_capacity < 1 or config.vpl_capacity > 64:
            raise ValueError("clustering.vpl_capacity must be in [1, 64]")
        if config.cube_size not in {64, 128, 256, 512}:
            raise ValueError("clustering.cube_size must be one of 64/128/256/512")
        if config.tile_size < 4 or config.tile_size > 64:
            raise ValueError("clustering.tile_size must be in [4, 64]")
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_vpls": self.max_vpls,
            "max_lights": self.max_lights,
            "spatial_weight": self.spatial_weight,
            "orientation_weight": self.orientation_weight,
            "iterations": self.iterations,
            "cube_size": self.cube_size,
            "tile_size": self.tile_size,
            "vpl_capacity": self.vpl_capacity,
        }


@dataclass(frozen=True)
class RenderConfig:
    width: int = 960
    height: int = 540
    exposure: float = 0.20
    screenshot_path: str = "artifacts/frame_tonemapped.png"
    debug_screenshot_path: str = "artifacts/frame_linear.png"
    warning_path: str = "artifacts/render_warnings.json"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RenderConfig":
        config = cls(
            width=int(payload.get("width", 960)),
            height=int(payload.get("height", 540)),
            exposure=float(payload.get("exposure", 0.20)),
            screenshot_path=str(payload.get("screenshot_path", "artifacts/frame_tonemapped.png")),
            debug_screenshot_path=str(
                payload.get("debug_screenshot_path", "artifacts/frame_linear.png")
            ),
            warning_path=str(payload.get("warning_path", "artifacts/render_warnings.json")),
        )
        if config.width < 64 or config.height < 64:
            raise ValueError("render resolution must be at least 64x64")
        if config.exposure <= 0.0:
            raise ValueError("render.exposure must be positive")
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "exposure": self.exposure,
            "screenshot_path": self.screenshot_path,
            "debug_screenshot_path": self.debug_screenshot_path,
            "warning_path": self.warning_path,
        }


@dataclass
class SceneConfig:
    seed: int
    materials: list[Material]
    lights: list[Light]
    clustering: ClusteringConfig
    render: RenderConfig
    camera: dict[str, Any]
    room: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        diagnostics: DiagnosticLog | None = None,
    ) -> "SceneConfig":
        diagnostics = diagnostics or DiagnosticLog()
        materials = [
            Material.from_dict(item, index, diagnostics)
            for index, item in enumerate(payload.get("materials", []))
        ]
        if not materials:
            raise ValueError("scene config must contain at least one material")
        material_ids = [material.id for material in materials]
        if len(set(material_ids)) != len(material_ids):
            raise ValueError("material ids must be unique")

        clustering = ClusteringConfig.from_dict(payload.get("clustering", {}))
        raw_lights = list(payload.get("lights", []))
        if not raw_lights:
            raise ValueError("scene config must contain at least one light")
        lights = [Light.from_dict(item, index) for index, item in enumerate(raw_lights)]
        light_ids = [light.id for light in lights]
        if len(set(light_ids)) != len(light_ids):
            raise ValueError("light ids must be unique")
        if len(lights) > clustering.max_lights:
            lights.sort(key=lambda light: (light.importance, light.id), reverse=True)
            dropped = lights[clustering.max_lights :]
            lights = lights[: clustering.max_lights]
            diagnostics.add(
                "LIGHT_LIMIT_TRUNCATED",
                "Scene contains more lights than allowed; weakest lights were truncated.",
                context={
                    "limit": clustering.max_lights,
                    "truncated_light_ids": [light.id for light in dropped],
                    "rendering_continues": True,
                },
            )
        return cls(
            seed=int(payload.get("seed", 0)),
            materials=materials,
            lights=lights,
            clustering=clustering,
            render=RenderConfig.from_dict(payload.get("render", {})),
            camera=dict(payload.get("camera", {})),
            room=dict(payload.get("scene", {})),
            raw=payload,
        )

    def material_by_id(self, material_id: int) -> Material:
        for material in self.materials:
            if material.id == material_id:
                return material
        raise KeyError(f"unknown material id {material_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "seed": self.seed,
            "scene": self.room,
            "camera": self.camera,
            "materials": [material.to_dict() for material in self.materials],
            "lights": [light.to_dict() for light in self.lights],
            "clustering": self.clustering.to_dict(),
            "render": self.render.to_dict(),
        }


def load_scene_config(
    path: str | Path,
    diagnostics: DiagnosticLog | None = None,
) -> SceneConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SceneConfig.from_dict(payload, diagnostics)


def write_scene_config(path: str | Path, config: SceneConfig) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

