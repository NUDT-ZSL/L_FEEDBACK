"""Scene JSON parsing and validation.

The scene format is intentionally explicit. All color values are linear RGB
unless a field name says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class SceneValidationError(ValueError):
    """A business-level scene error, reported as E_INVALID_SCENE."""

    code = "E_INVALID_SCENE"


@dataclass(frozen=True)
class Camera:
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    fov_y_deg: float
    near: float
    far: float


@dataclass(frozen=True)
class DirectionalLight:
    direction: tuple[float, float, float]
    color: tuple[float, float, float]
    intensity: float


@dataclass(frozen=True)
class Sphere:
    index: int
    name: str
    center: tuple[float, float, float]
    radius: float
    base_color: tuple[float, float, float]
    metallic: float
    roughness: float
    normal_perturbation: float


@dataclass(frozen=True)
class Scene:
    name: str
    width: int
    height: int
    camera: Camera
    light: DirectionalLight
    background_color: tuple[float, float, float]
    ambient: float
    ambient_color: tuple[float, float, float]
    spheres: tuple[Sphere, ...]
    ssao_enabled: bool
    ssao_radius: float
    ssao_intensity: float
    ssao_samples: int
    source: str


def _fail(where: str, message: str) -> None:
    raise SceneValidationError(f"{where}: {message}")


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(where, "must be a JSON object")
    return value


def _required(obj: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in obj:
        _fail(where, f"missing required field '{key}'")
    return obj[key]


def _number(
    value: Any,
    where: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(where, "must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(where, "must be a finite number")
    if minimum is not None:
        if exclusive_minimum and result <= minimum:
            _fail(where, f"must be greater than {minimum:g}")
        if not exclusive_minimum and result < minimum:
            _fail(where, f"must be at least {minimum:g}")
    if maximum is not None and result > maximum:
        _fail(where, f"must be at most {maximum:g}")
    return result


def _integer(
    value: Any,
    where: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(where, "must be an integer")
    if minimum is not None and value < minimum:
        _fail(where, f"must be at least {minimum}")
    if maximum is not None and value > maximum:
        _fail(where, f"must be at most {maximum}")
    return value


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        _fail(where, "must be a boolean")
    return value


def _vec3(
    value: Any,
    where: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(where, "must be an array of three numbers")
    if len(value) != 3:
        _fail(where, "must contain exactly three numbers")
    return (
        _number(value[0], f"{where}[0]", minimum=minimum, maximum=maximum),
        _number(value[1], f"{where}[1]", minimum=minimum, maximum=maximum),
        _number(value[2], f"{where}[2]", minimum=minimum, maximum=maximum),
    )


def _norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def parse_scene(payload: Any, source: str = "<memory>") -> Scene:
    """Validate a decoded JSON value and return an immutable scene."""

    root = _mapping(payload, source)
    name_value = root.get("scene_name", "untitled")
    if not isinstance(name_value, str) or not name_value.strip():
        _fail(f"{source}.scene_name", "must be a non-empty string")

    resolution = _mapping(
        _required(root, "resolution", source), f"{source}.resolution"
    )
    width = _integer(
        _required(resolution, "width", f"{source}.resolution"),
        f"{source}.resolution.width",
        minimum=1,
        maximum=8192,
    )
    height = _integer(
        _required(resolution, "height", f"{source}.resolution"),
        f"{source}.resolution.height",
        minimum=1,
        maximum=8192,
    )

    camera_obj = _mapping(
        _required(root, "camera", source), f"{source}.camera"
    )
    camera = Camera(
        position=_vec3(
            _required(camera_obj, "position", f"{source}.camera"),
            f"{source}.camera.position",
        ),
        target=_vec3(
            _required(camera_obj, "target", f"{source}.camera"),
            f"{source}.camera.target",
        ),
        up=_vec3(
            camera_obj.get("up", [0.0, 1.0, 0.0]),
            f"{source}.camera.up",
        ),
        fov_y_deg=_number(
            _required(camera_obj, "fov_y_deg", f"{source}.camera"),
            f"{source}.camera.fov_y_deg",
            minimum=1.0,
            maximum=179.0,
        ),
        near=_number(
            camera_obj.get("near", 1.0e-4),
            f"{source}.camera.near",
            minimum=0.0,
            exclusive_minimum=True,
        ),
        far=_number(
            camera_obj.get("far", 1000.0),
            f"{source}.camera.far",
            minimum=0.0,
            exclusive_minimum=True,
        ),
    )
    if camera.far <= camera.near:
        _fail(f"{source}.camera", "far must be greater than near")
    if _norm(camera.up) == 0.0:
        _fail(f"{source}.camera.up", "must be non-zero")
    fwd = tuple(camera.target[i] - camera.position[i] for i in range(3))
    if _norm(fwd) == 0.0:
        _fail(f"{source}.camera", "position and target must differ")
    up_len = _norm(camera.up)
    fwd_len = _norm(fwd)
    cosine = sum(fwd[i] * camera.up[i] for i in range(3)) / (fwd_len * up_len)
    if abs(cosine) > 1.0 - 1.0e-8:
        _fail(f"{source}.camera.up", "must not be parallel to the view direction")

    light_obj = _mapping(
        _required(root, "light", source), f"{source}.light"
    )
    light_direction = _vec3(
        _required(light_obj, "direction", f"{source}.light"),
        f"{source}.light.direction",
    )
    if _norm(light_direction) == 0.0:
        _fail(f"{source}.light.direction", "must be non-zero")
    light = DirectionalLight(
        direction=light_direction,
        color=_vec3(
            light_obj.get("color", [1.0, 1.0, 1.0]),
            f"{source}.light.color",
            minimum=0.0,
        ),
        intensity=_number(
            light_obj.get("intensity", 1.0),
            f"{source}.light.intensity",
            minimum=0.0,
        ),
    )

    background_obj = _mapping(root.get("background", {}), f"{source}.background")
    background_color = _vec3(
        background_obj.get("color", [0.01, 0.012, 0.016]),
        f"{source}.background.color",
        minimum=0.0,
    )
    ambient = _number(
        background_obj.get("ambient", 0.20),
        f"{source}.background.ambient",
        minimum=0.0,
    )
    ambient_color = _vec3(
        background_obj.get("ambient_color", [1.0, 1.0, 1.0]),
        f"{source}.background.ambient_color",
        minimum=0.0,
    )

    ssao_obj = _mapping(root.get("ssao", {}), f"{source}.ssao")
    ssao_enabled = _boolean(
        ssao_obj.get("enabled", True), f"{source}.ssao.enabled"
    )
    ssao_radius = _number(
        ssao_obj.get("radius", 0.35),
        f"{source}.ssao.radius",
        minimum=0.0,
        exclusive_minimum=True,
    )
    ssao_intensity = _number(
        ssao_obj.get("intensity", 1.0),
        f"{source}.ssao.intensity",
        minimum=0.0,
    )
    ssao_samples = _integer(
        ssao_obj.get("samples", 16),
        f"{source}.ssao.samples",
        minimum=1,
        maximum=64,
    )

    # Accept "materials" as a convenience alias, but the sphere fields remain
    # flat as specified by the scene contract.
    sphere_key = "spheres" if "spheres" in root else "materials"
    spheres_value = _required(root, sphere_key, source)
    if (
        not isinstance(spheres_value, Sequence)
        or isinstance(spheres_value, (str, bytes))
        or len(spheres_value) == 0
    ):
        _fail(f"{source}.{sphere_key}", "must be a non-empty array")

    spheres: list[Sphere] = []
    names: set[str] = set()
    for index, item in enumerate(spheres_value):
        where = f"{source}.{sphere_key}[{index}]"
        obj = _mapping(item, where)
        default_name = f"sphere_{index:03d}"
        raw_name = obj.get("name", default_name)
        if not isinstance(raw_name, str) or not raw_name.strip():
            _fail(f"{where}.name", "must be a non-empty string")
        sphere_name = raw_name.strip()
        if sphere_name in names:
            _fail(f"{where}.name", f"duplicate sphere name '{sphere_name}'")
        names.add(sphere_name)
        spheres.append(
            Sphere(
                index=index,
                name=sphere_name,
                center=_vec3(
                    _required(obj, "center", where), f"{where}.center"
                ),
                radius=_number(
                    _required(obj, "radius", where),
                    f"{where}.radius",
                    minimum=0.0,
                    exclusive_minimum=True,
                ),
                base_color=_vec3(
                    _required(obj, "base_color", where),
                    f"{where}.base_color",
                    minimum=0.0,
                    maximum=1.0,
                ),
                metallic=_number(
                    _required(obj, "metallic", where),
                    f"{where}.metallic",
                    minimum=0.0,
                    maximum=1.0,
                ),
                roughness=_number(
                    _required(obj, "roughness", where),
                    f"{where}.roughness",
                    minimum=0.0,
                    maximum=1.0,
                    exclusive_minimum=True,
                ),
                normal_perturbation=_number(
                    _required(obj, "normal_perturbation", where),
                    f"{where}.normal_perturbation",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        )

    return Scene(
        name=name_value.strip(),
        width=width,
        height=height,
        camera=camera,
        light=light,
        background_color=background_color,
        ambient=ambient,
        ambient_color=ambient_color,
        spheres=tuple(spheres),
        ssao_enabled=ssao_enabled,
        ssao_radius=ssao_radius,
        ssao_intensity=ssao_intensity,
        ssao_samples=ssao_samples,
        source=source,
    )


def load_scene(path: str | Path) -> Scene:
    """Load and validate a scene JSON file."""

    scene_path = Path(path)
    try:
        text = scene_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SceneValidationError(f"{scene_path}: cannot read scene: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SceneValidationError(
            f"{scene_path}: invalid JSON at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc
    return parse_scene(payload, str(scene_path))

