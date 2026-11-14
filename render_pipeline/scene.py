from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ErrorCode, PipelineError, diagnostic
from .util import as_finite_float, require_dict, require_list, safe_id, sha256_bytes


@dataclass(frozen=True)
class MeshSpec:
    u_segments: int
    v_segments: int
    radius_x: float
    radius_z: float
    height: float
    profile_amplitude: float
    profile_waves: float


@dataclass(frozen=True)
class UVLayout:
    width: int
    height: int
    mesh: MeshSpec


@dataclass(frozen=True)
class CameraSpec:
    id: str
    yaw_deg: float
    ortho_height: float
    width: int
    height_px: int


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    uv_layout: UVLayout
    material: dict[str, Any]
    cameras: tuple[CameraSpec, ...]
    lighting_presets: tuple[dict[str, Any], ...]
    input_sha256: str
    canonical_sha256: str


@dataclass(frozen=True)
class LightSpec:
    id: str
    key_direction: tuple[float, float, float]
    key_irradiance: tuple[float, float, float]
    ambient_irradiance: tuple[float, float, float]
    tone_operator: str
    exposure: float
    white_point: float


def _expect_int(value: Any, *, path: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PipelineError(
            code=ErrorCode.UV_LAYOUT_INVALID,
            message=f"{path} must be an integer",
            scope="uv_layout",
            path=path,
        )
    if value < minimum or value > maximum:
        raise PipelineError(
            code=ErrorCode.UV_LAYOUT_INVALID,
            message=f"{path} must be between {minimum} and {maximum}",
            scope="uv_layout",
            path=path,
            context={"value": value},
        )
    return value


def _parse_uv_layout(raw: Any) -> UVLayout:
    layout = require_dict(
        raw,
        code=ErrorCode.SCENE_FIELD_MISSING,
        path="uv_layout",
        message="uv_layout must be an object",
    )
    for key in ("width", "height", "mesh"):
        if key not in layout:
            raise PipelineError(
                code=ErrorCode.SCENE_FIELD_MISSING,
                message=f"uv_layout.{key} is required",
                scope="uv_layout",
                path=f"uv_layout.{key}",
            )
    width = _expect_int(layout["width"], path="uv_layout.width", minimum=8, maximum=4096)
    height = _expect_int(layout["height"], path="uv_layout.height", minimum=8, maximum=4096)
    mesh_raw = require_dict(
        layout["mesh"],
        code=ErrorCode.UV_LAYOUT_INVALID,
        path="uv_layout.mesh",
        message="uv_layout.mesh must be an object",
    )
    required = (
        "u_segments",
        "v_segments",
        "radius_x",
        "radius_z",
        "height",
        "profile_amplitude",
        "profile_waves",
    )
    for key in required:
        if key not in mesh_raw:
            raise PipelineError(
                code=ErrorCode.SCENE_FIELD_MISSING,
                message=f"uv_layout.mesh.{key} is required",
                scope="uv_layout",
                path=f"uv_layout.mesh.{key}",
            )
    u_segments = _expect_int(mesh_raw["u_segments"], path="uv_layout.mesh.u_segments", minimum=8, maximum=512)
    v_segments = _expect_int(mesh_raw["v_segments"], path="uv_layout.mesh.v_segments", minimum=8, maximum=512)
    scalar_values = {}
    for key in ("radius_x", "radius_z", "height", "profile_amplitude", "profile_waves"):
        path = f"uv_layout.mesh.{key}"
        value = as_finite_float(mesh_raw[key], path=path, code=ErrorCode.UV_LAYOUT_INVALID)
        if key.startswith("radius") or key == "height":
            if value <= 0.0:
                raise PipelineError(
                    code=ErrorCode.UV_LAYOUT_INVALID,
                    message=f"{path} must be positive",
                    scope="uv_layout",
                    path=path,
                )
        if key == "profile_amplitude" and not (-0.5 <= value <= 0.5):
            raise PipelineError(
                code=ErrorCode.UV_LAYOUT_INVALID,
                message=f"{path} must be in [-0.5, 0.5]",
                scope="uv_layout",
                path=path,
            )
        scalar_values[key] = value
    return UVLayout(
        width=width,
        height=height,
        mesh=MeshSpec(
            u_segments=u_segments,
            v_segments=v_segments,
            radius_x=scalar_values["radius_x"],
            radius_z=scalar_values["radius_z"],
            height=scalar_values["height"],
            profile_amplitude=scalar_values["profile_amplitude"],
            profile_waves=scalar_values["profile_waves"],
        ),
    )


def _default_cameras() -> list[Any]:
    return [
        {
            "id": "front",
            "yaw_deg": 0.0,
            "ortho_height": 1.35,
            "resolution": [160, 200],
        },
        {
            "id": "side",
            "yaw_deg": 90.0,
            "ortho_height": 1.35,
            "resolution": [160, 200],
        },
    ]


def _parse_cameras(raw: Any) -> tuple[CameraSpec, ...]:
    entries = _default_cameras() if raw is None else require_list(
        raw,
        code=ErrorCode.CAMERA_INVALID,
        path="cameras",
        message="cameras must be an array",
    )
    if not entries:
        raise PipelineError(
            code=ErrorCode.CAMERA_INVALID,
            message="at least one camera is required",
            scope="camera",
            path="cameras",
        )
    cameras: list[CameraSpec] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        path = f"cameras[{index}]"
        camera = require_dict(entry, code=ErrorCode.CAMERA_INVALID, path=path, message=f"{path} must be an object")
        for key in ("id", "yaw_deg", "ortho_height", "resolution"):
            if key not in camera:
                raise PipelineError(
                    code=ErrorCode.SCENE_FIELD_MISSING,
                    message=f"{path}.{key} is required",
                    scope="camera",
                    path=f"{path}.{key}",
                )
        camera_id = safe_id(camera["id"], path=f"{path}.id")
        if camera_id in seen:
            raise PipelineError(
                code=ErrorCode.SCENE_ID_COLLISION,
                message=f"duplicate camera id {camera_id!r}",
                scope="camera",
                path=f"{path}.id",
                context={"id": camera_id},
            )
        seen.add(camera_id)
        yaw = as_finite_float(camera["yaw_deg"], path=f"{path}.yaw_deg", code=ErrorCode.CAMERA_INVALID)
        ortho_height = as_finite_float(camera["ortho_height"], path=f"{path}.ortho_height", code=ErrorCode.CAMERA_INVALID)
        if ortho_height <= 0.0:
            raise PipelineError(
                code=ErrorCode.CAMERA_INVALID,
                message=f"{path}.ortho_height must be positive",
                scope="camera",
                path=f"{path}.ortho_height",
            )
        resolution = require_list(
            camera["resolution"],
            code=ErrorCode.CAMERA_INVALID,
            path=f"{path}.resolution",
            message=f"{path}.resolution must be [width, height]",
        )
        if len(resolution) != 2:
            raise PipelineError(
                code=ErrorCode.CAMERA_INVALID,
                message=f"{path}.resolution must have two elements",
                scope="camera",
                path=f"{path}.resolution",
            )
        width = _expect_int(resolution[0], path=f"{path}.resolution[0]", minimum=8, maximum=2048)
        height_px = _expect_int(resolution[1], path=f"{path}.resolution[1]", minimum=8, maximum=2048)
        cameras.append(CameraSpec(camera_id, yaw, ortho_height, width, height_px))
    return tuple(cameras)


def load_scene_bytes(data: bytes) -> SceneSpec:
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PipelineError(
            code=ErrorCode.SCENE_JSON_INVALID,
            message=f"scene JSON cannot be decoded: {exc}",
            scope="scene",
            path="$",
        ) from exc
    root = require_dict(
        raw,
        code=ErrorCode.SCENE_JSON_INVALID,
        path="$",
        message="scene root must be an object",
    )
    for key in ("scene_id", "uv_layout", "material", "lighting_presets"):
        if key not in root:
            raise PipelineError(
                code=ErrorCode.SCENE_FIELD_MISSING,
                message=f"{key} is required",
                scope="scene",
                path=key,
            )
    scene_id = safe_id(root["scene_id"], path="scene_id")
    uv_layout = _parse_uv_layout(root["uv_layout"])
    material = require_dict(
        root["material"],
        code=ErrorCode.MATERIAL_FIELD_MISSING,
        path="material",
        message="material must be an object",
    )
    presets = require_list(
        root["lighting_presets"],
        code=ErrorCode.SCENE_FIELD_MISSING,
        path="lighting_presets",
        message="lighting_presets must be an array",
    )
    if not presets:
        raise PipelineError(
            code=ErrorCode.SCENE_FIELD_INVALID,
            message="lighting_presets must not be empty",
            scope="scene",
            path="lighting_presets",
        )
    preset_ids: set[str] = set()
    for index, value in enumerate(presets):
        if not isinstance(value, dict):
            raise PipelineError(
                code=ErrorCode.SCENE_FIELD_INVALID,
                message=f"lighting_presets[{index}] must be an object",
                scope="lighting_preset",
                path=f"lighting_presets[{index}]",
            )
        if "id" in value:
            preset_id = safe_id(value["id"], path=f"lighting_presets[{index}].id")
            if preset_id in preset_ids:
                raise PipelineError(
                    code=ErrorCode.SCENE_ID_COLLISION,
                    message=f"duplicate lighting preset id {preset_id!r}",
                    scope="lighting_preset",
                    path=f"lighting_presets[{index}].id",
                    context={"id": preset_id},
                )
            preset_ids.add(preset_id)
    cameras = _parse_cameras(root.get("cameras"))
    canonical = json.dumps(
        raw,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return SceneSpec(
        scene_id=scene_id,
        uv_layout=uv_layout,
        material=material,
        cameras=cameras,
        lighting_presets=tuple(presets),
        input_sha256=sha256_bytes(data),
        canonical_sha256=sha256_bytes(canonical),
    )


def load_scene(path: Path) -> SceneSpec:
    return load_scene_bytes(path.read_bytes())


def inspect_lighting_preset(raw: Any, index: int) -> tuple[LightSpec | None, list[dict[str, Any]]]:
    """Validate a lighting preset without aborting siblings.

    Returns ``(None, diagnostics)`` when the preset cannot be rendered. The
    validation order is explicit so diagnostics are stable.
    """

    prefix = f"lighting_presets[{index}]"
    errors: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        errors.append(
            diagnostic(
                ErrorCode.LIGHT_VALUE_INVALID,
                f"{prefix} must be an object",
                "lighting_preset",
                prefix,
            )
        )
        return None, errors

    preset_id_value = raw.get("id")
    try:
        preset_id = safe_id(preset_id_value, path=f"{prefix}.id")
    except PipelineError as exc:
        errors.append(
            diagnostic(
                ErrorCode.LIGHT_FIELD_MISSING if preset_id_value is None else ErrorCode.LIGHT_VALUE_INVALID,
                exc.message,
                "lighting_preset",
                f"{prefix}.id",
            )
        )
        return None, errors

    def missing_field(field: str) -> None:
        errors.append(
            diagnostic(
                ErrorCode.LIGHT_FIELD_MISSING,
                f"{field} is required",
                "lighting_preset",
                f"{prefix}.{field}",
                preset_id=preset_id,
            )
        )

    key_light = raw.get("key_light")
    if not isinstance(key_light, dict):
        missing_field("key_light")
        key_light = {}
    if "direction" not in key_light:
        missing_field("key_light.direction")
    if "irradiance" not in key_light:
        missing_field("key_light.irradiance")
    if "ambient_irradiance" not in raw:
        missing_field("ambient_irradiance")
    tone = raw.get("tone_mapping")
    if not isinstance(tone, dict):
        missing_field("tone_mapping")
        tone = {}
    for field in ("operator", "exposure", "white_point"):
        if field not in tone:
            missing_field(f"tone_mapping.{field}")
    if errors:
        return None, errors

    direction: tuple[float, float, float] | None = None
    irradiance: tuple[float, float, float] | None = None
    ambient: tuple[float, float, float] | None = None
    exposure: float | None = None
    white_point: float | None = None
    try:
        direction_raw = key_light["direction"]
        if not isinstance(direction_raw, (list, tuple)) or len(direction_raw) != 3:
            raise ValueError("direction must contain three numbers")
        direction_values = tuple(
            as_finite_float(item, path=f"{prefix}.key_light.direction[{i}]", code=ErrorCode.LIGHT_VALUE_INVALID)
            for i, item in enumerate(direction_raw)
        )
        norm = math.sqrt(sum(component * component for component in direction_values))
        if norm <= 1e-12:
            raise ValueError("direction must be non-zero")
        direction = tuple(component / norm for component in direction_values)  # type: ignore[assignment]

        irradiance_raw = key_light["irradiance"]
        if not isinstance(irradiance_raw, (list, tuple)) or len(irradiance_raw) != 3:
            raise ValueError("irradiance must contain three numbers")
        irradiance_values = tuple(
            as_finite_float(item, path=f"{prefix}.key_light.irradiance[{i}]", code=ErrorCode.LIGHT_VALUE_INVALID)
            for i, item in enumerate(irradiance_raw)
        )
        if min(irradiance_values) < 0.0:
            raise ValueError("irradiance must be non-negative")
        irradiance = irradiance_values  # type: ignore[assignment]

        ambient_raw = raw["ambient_irradiance"]
        if not isinstance(ambient_raw, (list, tuple)) or len(ambient_raw) != 3:
            raise ValueError("ambient_irradiance must contain three numbers")
        ambient_values = tuple(
            as_finite_float(item, path=f"{prefix}.ambient_irradiance[{i}]", code=ErrorCode.LIGHT_VALUE_INVALID)
            for i, item in enumerate(ambient_raw)
        )
        if min(ambient_values) < 0.0:
            raise ValueError("ambient_irradiance must be non-negative")
        ambient = ambient_values  # type: ignore[assignment]

        exposure = as_finite_float(tone["exposure"], path=f"{prefix}.tone_mapping.exposure", code=ErrorCode.LIGHT_VALUE_INVALID)
        if exposure <= 0.0:
            raise ValueError("exposure must be positive")
        white_point = as_finite_float(tone["white_point"], path=f"{prefix}.tone_mapping.white_point", code=ErrorCode.LIGHT_VALUE_INVALID)
        if white_point <= 0.0:
            raise ValueError("white_point must be positive")
    except (PipelineError, ValueError) as exc:
        errors.append(
            diagnostic(
                ErrorCode.LIGHT_VALUE_INVALID,
                str(exc),
                "lighting_preset",
                prefix,
                preset_id=preset_id,
            )
        )
        return None, errors

    operator = tone["operator"]
    if not isinstance(operator, str) or operator.lower() not in {"reinhard", "aces"}:
        errors.append(
            diagnostic(
                ErrorCode.TONE_MAP_OPERATOR_UNSUPPORTED,
                f"unsupported tone-mapping operator {operator!r}",
                "lighting_preset",
                f"{prefix}.tone_mapping.operator",
                preset_id=preset_id,
            )
        )
        return None, errors
    return (
        LightSpec(
            id=preset_id,
            key_direction=direction,
            key_irradiance=irradiance,
            ambient_irradiance=ambient,
            tone_operator=operator.lower(),
            exposure=exposure,
            white_point=white_point,
        ),
        errors,
    )
