from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np

from .errors import ValidationError


def _triple(value: Sequence[float], name: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} 必须是三个有限数值") from exc
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValidationError(f"{name} 必须是三个有限数值")
    return values


@dataclass(frozen=True)
class Camera:
    eye: tuple[float, float, float] = (0.0, 6.8, 9.5)
    target: tuple[float, float, float] = (0.0, 0.4, 0.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_y_deg: float = 48.0
    near: float = 0.15
    far: float = 60.0

    def normalized(self, width: int, height: int) -> "Camera":
        eye = _triple(self.eye, "camera.eye")
        target = _triple(self.target, "camera.target")
        up = _triple(self.up, "camera.up")
        if width <= 0 or height <= 0:
            raise ValidationError("分辨率必须为正整数")
        if not math.isfinite(self.fov_y_deg) or not 1.0 <= self.fov_y_deg <= 178.0:
            raise ValidationError("camera.fov_y_deg 必须在 [1, 178] 度内")
        if not math.isfinite(self.near) or not math.isfinite(self.far):
            raise ValidationError("相机近远裁剪面必须有限")
        if self.near <= 0.0 or self.far <= self.near:
            raise ValidationError("必须满足 0 < near < far")
        forward = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
        if float(np.linalg.norm(forward)) < 1e-9:
            raise ValidationError("相机 eye 与 target 不能重合")
        if float(np.linalg.norm(np.asarray(up))) < 1e-9:
            raise ValidationError("相机 up 不能为零向量")
        if abs(float(np.dot(forward / np.linalg.norm(forward), np.asarray(up) / np.linalg.norm(up)))) > 1.0 - 1e-7:
            raise ValidationError("相机 up 不能与视线平行")
        return Camera(eye, target, up, float(self.fov_y_deg), float(self.near), float(self.far))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DirectionalLight:
    direction: tuple[float, float, float] = (-0.42, -1.0, -0.31)
    color: tuple[float, float, float] = (1.0, 0.96, 0.88)
    intensity: float = 1.05

    def normalized(self) -> "DirectionalLight":
        direction = _triple(self.direction, "directional_light.direction")
        color = _triple(self.color, "directional_light.color")
        if float(np.linalg.norm(np.asarray(direction))) < 1e-9:
            raise ValidationError("方向光方向不能为零向量")
        if not math.isfinite(self.intensity) or self.intensity < 0.0:
            raise ValidationError("方向光强度必须为非负有限数")
        if any(c < 0.0 for c in color):
            raise ValidationError("方向光颜色不能为负")
        return DirectionalLight(direction, color, float(self.intensity))


@dataclass(frozen=True)
class PointLight:
    position: tuple[float, float, float]
    color: tuple[float, float, float] = (1.0, 0.82, 0.62)
    intensity: float = 2.0
    range: float = 9.0

    def normalized(self, index: int = 0) -> "PointLight":
        position = _triple(self.position, f"point_lights[{index}].position")
        color = _triple(self.color, f"point_lights[{index}].color")
        if not math.isfinite(self.intensity) or self.intensity < 0.0:
            raise ValidationError(f"point_lights[{index}].intensity 必须为非负有限数")
        if not math.isfinite(self.range) or self.range <= 0.0:
            raise ValidationError(f"point_lights[{index}].range 必须为正有限数")
        if any(c < 0.0 for c in color):
            raise ValidationError(f"point_lights[{index}].color 不能为负")
        return PointLight(position, color, float(self.intensity), float(self.range))


@dataclass(frozen=True)
class Material:
    name: str
    albedo: tuple[float, float, float]
    roughness: float
    normal_strength: float
    detail_scale: float
    detail_phase: float

    def normalized(self, index: int) -> "Material":
        if not self.name or not isinstance(self.name, str):
            raise ValidationError(f"materials[{index}].name 不能为空")
        albedo = _triple(self.albedo, f"materials[{index}].albedo")
        if any(c < 0.0 for c in albedo):
            raise ValidationError(f"materials[{index}].albedo 不能为负")
        if not math.isfinite(self.roughness) or not 0.0 < self.roughness <= 1.0:
            raise ValidationError(f"materials[{index}].roughness 必须在 (0, 1] 内")
        if not math.isfinite(self.normal_strength) or self.normal_strength < 0.0:
            raise ValidationError(f"materials[{index}].normal_strength 必须为非负有限数")
        if not math.isfinite(self.detail_scale) or self.detail_scale <= 0.0:
            raise ValidationError(f"materials[{index}].detail_scale 必须为正有限数")
        if not math.isfinite(self.detail_phase):
            raise ValidationError(f"materials[{index}].detail_phase 必须有限")
        return Material(
            self.name,
            albedo,
            float(self.roughness),
            float(self.normal_strength),
            float(self.detail_scale),
            float(self.detail_phase),
        )


def default_materials() -> tuple[Material, ...]:
    return (
        Material("meadow", (0.17, 0.30, 0.11), 0.82, 0.10, 0.70, 0.0),
        Material("rock", (0.34, 0.32, 0.29), 0.68, 0.22, 1.65, 1.1),
        Material("sand", (0.53, 0.43, 0.25), 0.92, 0.07, 1.15, 2.3),
        Material("snow", (0.78, 0.83, 0.87), 0.58, 0.04, 0.90, 3.7),
    )


def default_point_lights() -> tuple[PointLight, ...]:
    return (
        PointLight((-3.2, 3.8, -2.4), (1.0, 0.73, 0.46), 2.4, 10.0),
        PointLight((3.1, 2.8, 2.0), (0.48, 0.68, 1.0), 2.0, 9.0),
    )


@dataclass(frozen=True)
class CascadeSettings:
    count: int = 3
    shadow_size: int = 512
    split_lambda: float = 0.68
    overlap_fraction: float = 0.12

    def normalized(self) -> "CascadeSettings":
        if isinstance(self.count, bool) or not isinstance(self.count, (int, np.integer)) or self.count <= 0:
            raise ValidationError("级联数量必须为正整数")
        if isinstance(self.shadow_size, bool) or not isinstance(self.shadow_size, (int, np.integer)) or self.shadow_size <= 0:
            raise ValidationError("阴影贴图尺寸必须为正整数")
        if self.shadow_size > 4096:
            raise ValidationError("阴影贴图尺寸不能超过 4096")
        if not math.isfinite(self.split_lambda) or not 0.0 <= self.split_lambda <= 1.0:
            raise ValidationError("级联 split_lambda 必须在 [0, 1] 内")
        if not math.isfinite(self.overlap_fraction) or not 0.0 <= self.overlap_fraction < 0.5:
            raise ValidationError("级联重叠比例必须在 [0, 0.5) 内")
        return CascadeSettings(int(self.count), int(self.shadow_size), float(self.split_lambda), float(self.overlap_fraction))


@dataclass(frozen=True)
class ToneSettings:
    exposure: float = 1.0
    gamma: float = 2.2

    def normalized(self) -> "ToneSettings":
        if not math.isfinite(self.exposure) or self.exposure <= 0.0:
            raise ValidationError("色调映射曝光必须为正有限数")
        if not math.isfinite(self.gamma) or not 0.5 <= self.gamma <= 4.0:
            raise ValidationError("色调映射 gamma 必须在 [0.5, 4] 内")
        return ToneSettings(float(self.exposure), float(self.gamma))


@dataclass(frozen=True)
class RenderSettings:
    width: int = 320
    height: int = 240
    tile_size: int = 32
    cascades: CascadeSettings = CascadeSettings()
    ambient: tuple[float, float, float] = (0.095, 0.115, 0.15)
    tone: ToneSettings = ToneSettings()
    backface_culling: bool = True
    shadow_constant_bias: float = 0.006
    shadow_slope_bias: float = 1.35
    shadow_normal_offset: float = 0.8
    shadow_pcf_radius: int = 1

    def normalized(self) -> "RenderSettings":
        for name, value in (("width", self.width), ("height", self.height), ("tile_size", self.tile_size)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValidationError(f"{name} 必须为正整数")
        if self.width > 8192 or self.height > 8192:
            raise ValidationError("分辨率不能超过 8192")
        if self.tile_size > 1024:
            raise ValidationError("tile_size 不能超过 1024")
        if isinstance(self.shadow_pcf_radius, bool) or not isinstance(self.shadow_pcf_radius, (int, np.integer)):
            raise ValidationError("shadow_pcf_radius 必须为整数")
        if not 0 <= self.shadow_pcf_radius <= 4:
            raise ValidationError("shadow_pcf_radius 必须在 [0, 4] 内")
        if self.shadow_constant_bias < 0.0 or not math.isfinite(self.shadow_constant_bias):
            raise ValidationError("shadow_constant_bias 必须为非负有限数")
        if self.shadow_slope_bias < 0.0 or not math.isfinite(self.shadow_slope_bias):
            raise ValidationError("shadow_slope_bias 必须为非负有限数")
        if self.shadow_normal_offset < 0.0 or not math.isfinite(self.shadow_normal_offset):
            raise ValidationError("shadow_normal_offset 必须为非负有限数")
        ambient = _triple(self.ambient, "ambient")
        if any(c < 0.0 for c in ambient):
            raise ValidationError("ambient 不能为负")
        return RenderSettings(
            width=int(self.width),
            height=int(self.height),
            tile_size=int(self.tile_size),
            cascades=self.cascades.normalized(),
            ambient=ambient,
            tone=self.tone.normalized(),
            backface_culling=bool(self.backface_culling),
            shadow_constant_bias=float(self.shadow_constant_bias),
            shadow_slope_bias=float(self.shadow_slope_bias),
            shadow_normal_offset=float(self.shadow_normal_offset),
            shadow_pcf_radius=int(self.shadow_pcf_radius),
        )
