"""The built-in deterministic scene and its structural validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from .errors import business_error
from .spectral import SpectralModel


def _prism_mesh() -> dict[str, list[list[float]] | list[list[int]]]:
    # A closed triangular prism.  The triangle is in XY, extruded along Z.
    vertices = [
        [-0.55, 0.55, 0.55],   # 0 A front
        [0.55, 0.55, 0.55],    # 1 B front
        [0.00, 1.32, 0.55],    # 2 C front
        [-0.55, 0.55, -0.55],  # 3 A back
        [0.55, 0.55, -0.55],   # 4 B back
        [0.00, 1.32, -0.55],   # 5 C back
    ]
    triangles = [
        [0, 1, 2],  # front +Z
        [3, 5, 4],  # back -Z
        [0, 3, 4],  # bottom
        [0, 4, 1],
        [1, 4, 5],  # right slanted side
        [1, 5, 2],
        [2, 5, 3],  # left slanted side
        [2, 3, 0],
    ]
    return {"vertices": vertices, "triangles": triangles}


def scene_definition(name: str) -> dict[str, Any]:
    if name != "prism":
        raise business_error("E_SCENE", f"unknown built-in scene {name!r}; only 'prism' exists")

    def table(nm: list[float], values: list[float]) -> dict[str, list[float]]:
        return {"nm": nm, "values": values}

    return {
        "name": "prism",
        "description": "triangular glass prism above a finite water layer and diffuse floor",
        "spectral": {"min_nm": 400.0, "max_nm": 700.0, "sampling": "uniform hero wavelength"},
        "camera": {
            "position": [2.65, 1.55, 3.25],
            "target": [-0.05, 0.50, 0.0],
            "up": [0.0, 1.0, 0.0],
            "vfov_deg": 42.0,
        },
        "water": {
            "surface_y": 0.22,
            "extent_x": [-2.30, 2.30],
            "extent_z": [-2.30, 2.30],
            "ior": table(
                [400, 450, 500, 550, 600, 650, 700],
                [1.342, 1.338, 1.336, 1.333, 1.331, 1.330, 1.329],
            ),
            "absorption": table(
                [400, 450, 500, 550, 600, 650, 700],
                [0.045, 0.040, 0.035, 0.030, 0.032, 0.036, 0.040],
            ),
        },
        "ground": {
            "y": 0.0,
            "extent_x": [-2.50, 2.50],
            "extent_z": [-2.50, 2.50],
            "albedo": [0.62, 0.60, 0.55],
        },
        "prism": {
            "mesh": _prism_mesh(),
            "ior": table(
                [400, 450, 500, 550, 600, 650, 700],
                [1.505, 1.502, 1.500, 1.499, 1.498, 1.497, 1.496],
            ),
            "absorption": table(
                [400, 450, 500, 550, 600, 650, 700],
                [0.025, 0.022, 0.020, 0.018, 0.017, 0.016, 0.015],
            ),
        },
        "light": {
            "center": [-1.10, 3.10, 0.55],
            "u": [1.35, 0.0, 0.0],
            "v": [0.0, 0.0, 0.85],
            "emission": 24.0,
        },
    }


def _require(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise business_error("E_SCENE_FIELDS", f"missing scene field {path}.{key}")
    return mapping[key]


def _require_vector(
    mapping: dict[str, Any], key: str, path: str, length: int
) -> list[float]:
    value = _require(mapping, key, path)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise business_error(
            "E_SCENE_VALUE", f"scene field {path}.{key} must contain {length} numbers"
        )
    try:
        out = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise business_error(
            "E_SCENE_VALUE", f"scene field {path}.{key} must contain numbers"
        ) from exc
    if not all(np.isfinite(out)):
        raise business_error("E_SCENE_VALUE", f"scene field {path}.{key} is not finite")
    return out


def _require_scalar(mapping: dict[str, Any], key: str, path: str) -> float:
    value = _require(mapping, key, path)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise business_error(
            "E_SCENE_VALUE", f"scene field {path}.{key} must be numeric"
        ) from exc
    if not np.isfinite(out):
        raise business_error("E_SCENE_VALUE", f"scene field {path}.{key} is not finite")
    return out


def _validate_table(
    mapping: dict[str, Any],
    key: str,
    path: str,
    *,
    positive: bool,
) -> None:
    table = _require(mapping, key, path)
    if not isinstance(table, dict):
        raise business_error("E_SCENE_VALUE", f"scene field {path}.{key} must be an object")
    nm = _require(table, "nm", f"{path}.{key}")
    raw_values = _require(table, "values", f"{path}.{key}")
    if not isinstance(nm, (list, tuple)) or not isinstance(raw_values, (list, tuple)):
        raise business_error(
            "E_SCENE_VALUE", f"scene field {path}.{key} needs nm and values lists"
        )
    try:
        nm_values = [float(item) for item in nm]
        values = [float(item) for item in raw_values]
    except (TypeError, ValueError) as exc:
        raise business_error(
            "E_SCENE_VALUE", f"scene field {path}.{key} tables must contain numbers"
        ) from exc
    if len(nm_values) != len(values) or len(values) < 2:
        raise business_error(
            "E_SCENE_VALUE", f"scene field {path}.{key} needs matching nm/values tables"
        )
    if not all(np.isfinite(nm_values)) or not all(np.isfinite(values)):
        raise business_error("E_SCENE_VALUE", f"scene field {path}.{key} has non-finite values")
    if any(b <= a for a, b in zip(nm_values, nm_values[1:])):
        raise business_error("E_SCENE_VALUE", f"scene field {path}.{key}.nm must increase")
    if positive and any(v <= 0.0 for v in values):
        raise business_error(
            "E_SCENE_VALUE", f"scene field {path}.{key}.values must be positive"
        )
    if not positive and any(v < 0.0 for v in values):
        raise business_error(
            "E_SCENE_VALUE", f"scene field {path}.{key}.values must be non-negative"
        )


def validate_scene_data(data: dict[str, Any]) -> None:
    """Validate all fields needed by the integrator.

    The CLI never accepts an external scene file.  This function is public so
    regression tests can exercise missing-field and malformed-value failures.
    """

    if not isinstance(data, dict):
        raise business_error("E_SCENE_FIELDS", "scene description must be an object")
    if _require(data, "name", "scene") != "prism":
        raise business_error("E_SCENE_VALUE", "scene.name must be 'prism'")

    spectral = _require(data, "spectral", "scene")
    lo = _require_scalar(spectral, "min_nm", "scene.spectral")
    hi = _require_scalar(spectral, "max_nm", "scene.spectral")
    if not (350.0 <= lo < hi <= 750.0):
        raise business_error(
            "E_SCENE_VALUE", "scene.spectral must satisfy 350 <= min_nm < max_nm <= 750"
        )

    camera = _require(data, "camera", "scene")
    _require_vector(camera, "position", "scene.camera", 3)
    _require_vector(camera, "target", "scene.camera", 3)
    _require_vector(camera, "up", "scene.camera", 3)
    fov = _require_scalar(camera, "vfov_deg", "scene.camera")
    if not (1.0 <= fov < 179.0):
        raise business_error("E_SCENE_VALUE", "scene.camera.vfov_deg must be in [1, 179)")

    water = _require(data, "water", "scene")
    water_surface_y = _require_scalar(water, "surface_y", "scene.water")
    x0, x1 = _require_vector(water, "extent_x", "scene.water", 2)
    z0, z1 = _require_vector(water, "extent_z", "scene.water", 2)
    if not (x0 < x1 and z0 < z1):
        raise business_error("E_SCENE_VALUE", "scene.water extents must have min < max")
    _validate_table(water, "ior", "scene.water", positive=True)
    _validate_table(water, "absorption", "scene.water", positive=False)

    ground = _require(data, "ground", "scene")
    ground_y = _require_scalar(ground, "y", "scene.ground")
    gx0, gx1 = _require_vector(ground, "extent_x", "scene.ground", 2)
    gz0, gz1 = _require_vector(ground, "extent_z", "scene.ground", 2)
    if not (gx0 < gx1 and gz0 < gz1):
        raise business_error("E_SCENE_VALUE", "scene.ground extents must have min < max")
    albedo = _require_vector(ground, "albedo", "scene.ground", 3)
    if any(v < 0.0 or v > 1.0 for v in albedo):
        raise business_error("E_SCENE_VALUE", "scene.ground.albedo must be in [0, 1]")

    prism = _require(data, "prism", "scene")
    mesh = _require(prism, "mesh", "scene.prism")
    if not isinstance(mesh, dict):
        raise business_error("E_SCENE_VALUE", "scene.prism.mesh must be an object")
    vertices = _require(mesh, "vertices", "scene.prism.mesh")
    triangles = _require(mesh, "triangles", "scene.prism.mesh")
    if not isinstance(vertices, (list, tuple)) or len(vertices) < 3:
        raise business_error("E_SCENE_VALUE", "scene.prism.mesh.vertices needs >= 3 points")
    if not isinstance(triangles, (list, tuple)) or len(triangles) < 1:
        raise business_error("E_SCENE_VALUE", "scene.prism.mesh.triangles needs >= 1 face")
    for index, vertex in enumerate(vertices):
        if not isinstance(vertex, (list, tuple)) or len(vertex) != 3:
            raise business_error(
                "E_SCENE_VALUE", f"scene.prism.mesh.vertices[{index}] must have 3 numbers"
            )
        try:
            vertex_values = [float(item) for item in vertex]
        except (TypeError, ValueError) as exc:
            raise business_error(
                "E_SCENE_VALUE", f"scene.prism.mesh.vertices[{index}] must contain numbers"
            ) from exc
        if not all(np.isfinite(vertex_values)):
            raise business_error(
                "E_SCENE_VALUE", f"scene.prism.mesh.vertices[{index}] is not finite"
            )
    for index, triangle in enumerate(triangles):
        if not isinstance(triangle, (list, tuple)) or len(triangle) != 3:
            raise business_error(
                "E_SCENE_VALUE", f"scene.prism.mesh.triangles[{index}] must have 3 indices"
            )
        for vertex_index in triangle:
            if not isinstance(vertex_index, int) or not (0 <= vertex_index < len(vertices)):
                raise business_error(
                    "E_SCENE_VALUE",
                    f"scene.prism.mesh.triangles[{index}] contains an invalid index",
                )
    _validate_table(prism, "ior", "scene.prism", positive=True)
    _validate_table(prism, "absorption", "scene.prism", positive=False)

    light = _require(data, "light", "scene")
    _require_vector(light, "center", "scene.light", 3)
    u = _require_vector(light, "u", "scene.light", 3)
    v = _require_vector(light, "v", "scene.light", 3)
    emission = _require_scalar(light, "emission", "scene.light")
    if emission <= 0.0:
        raise business_error("E_SCENE_VALUE", "scene.light.emission must be positive")
    if np.linalg.norm(np.cross(u, v)) <= 1.0e-9:
        raise business_error("E_SCENE_VALUE", "scene.light.u and scene.light.v must not be parallel")

    if water_surface_y <= ground_y:
        raise business_error("E_SCENE_VALUE", "water surface must lie above the ground plane")


@dataclass
class Scene:
    data: dict[str, Any]
    min_nm: float
    max_nm: float
    camera_position: np.ndarray
    camera_target: np.ndarray
    camera_up: np.ndarray
    camera_forward: np.ndarray
    camera_right: np.ndarray
    camera_true_up: np.ndarray
    camera_tan_half_fov: float
    width: int = 0
    height: int = 0
    vertices: np.ndarray | None = None
    triangles: np.ndarray | None = None
    triangle_v0: np.ndarray | None = None
    triangle_e1: np.ndarray | None = None
    triangle_e2: np.ndarray | None = None
    triangle_normals: np.ndarray | None = None
    water_y: float = 0.0
    water_x0: float = 0.0
    water_x1: float = 0.0
    water_z0: float = 0.0
    water_z1: float = 0.0
    ground_y: float = 0.0
    ground_x0: float = 0.0
    ground_x1: float = 0.0
    ground_z0: float = 0.0
    ground_z1: float = 0.0
    ground_albedo: np.ndarray | None = None
    light_center: np.ndarray | None = None
    light_u: np.ndarray | None = None
    light_v: np.ndarray | None = None
    light_u_hat: np.ndarray | None = None
    light_v_hat: np.ndarray | None = None
    light_u_half: float = 0.0
    light_v_half: float = 0.0
    light_area: float = 0.0
    light_normal: np.ndarray | None = None
    light_emission: float = 0.0
    spectral: SpectralModel | None = None

    def __post_init__(self) -> None:
        self._populate_geometry()

    def _populate_geometry(self) -> None:
        data = self.data
        camera = data["camera"]
        water = data["water"]
        ground = data["ground"]
        prism = data["prism"]

        self.vertices = np.asarray(prism["mesh"]["vertices"], dtype=np.float64)
        self.triangles = np.asarray(prism["mesh"]["triangles"], dtype=np.int64)
        self.triangle_v0 = self.vertices[self.triangles[:, 0]]
        v1 = self.vertices[self.triangles[:, 1]]
        v2 = self.vertices[self.triangles[:, 2]]
        self.triangle_e1 = v1 - self.triangle_v0
        self.triangle_e2 = v2 - self.triangle_v0
        normals = np.cross(self.triangle_e1, self.triangle_e2)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-30)
        self.triangle_normals = normals

        self.water_y = float(water["surface_y"])
        self.water_x0, self.water_x1 = map(float, water["extent_x"])
        self.water_z0, self.water_z1 = map(float, water["extent_z"])
        self.ground_y = float(ground["y"])
        self.ground_x0, self.ground_x1 = map(float, ground["extent_x"])
        self.ground_z0, self.ground_z1 = map(float, ground["extent_z"])
        self.ground_albedo = np.asarray(ground["albedo"], dtype=np.float64)

        light = data["light"]
        self.light_center = np.asarray(light["center"], dtype=np.float64)
        self.light_u = np.asarray(light["u"], dtype=np.float64)
        self.light_v = np.asarray(light["v"], dtype=np.float64)
        self.light_u_half = 0.5 * float(np.linalg.norm(self.light_u))
        self.light_v_half = 0.5 * float(np.linalg.norm(self.light_v))
        self.light_u_hat = self.light_u / (2.0 * self.light_u_half)
        self.light_v_hat = self.light_v / (2.0 * self.light_v_half)
        self.light_area = 4.0 * self.light_u_half * self.light_v_half
        self.light_normal = np.cross(self.light_u_hat, self.light_v_hat)
        self.light_normal /= np.linalg.norm(self.light_normal)
        self.light_emission = float(light["emission"])

        self.spectral = SpectralModel(
            self.min_nm,
            self.max_nm,
            {
                "water": water,
                "glass": prism,
            },
        )

        self.camera_position = np.asarray(camera["position"], dtype=np.float64)
        self.camera_target = np.asarray(camera["target"], dtype=np.float64)
        self.camera_up = np.asarray(camera["up"], dtype=np.float64)
        forward = self.camera_target - self.camera_position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, self.camera_up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)
        self.camera_forward = forward
        self.camera_right = right
        self.camera_true_up = true_up
        self.camera_tan_half_fov = float(np.tan(np.deg2rad(float(camera["vfov_deg"])) * 0.5))

    def fingerprint(self) -> str:
        payload = json.dumps(self.data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def point_inside_water(self, point: np.ndarray) -> bool:
        return bool(
            point[1] < self.water_y - 1.0e-8
            and self.water_x0 < point[0] < self.water_x1
            and self.water_z0 < point[2] < self.water_z1
        )

    def point_inside_prism(self, point: np.ndarray) -> bool:
        z_min = float(np.min(self.vertices[:, 2]))
        z_max = float(np.max(self.vertices[:, 2]))
        if not (z_min < point[2] < z_max):
            return False
        a = self.vertices[0]
        b = self.vertices[1]
        c = self.vertices[2]
        p = point
        v0 = c - a
        v1 = b - a
        v2 = p - a
        dot00 = float(np.dot(v0, v0))
        dot01 = float(np.dot(v0, v1))
        dot02 = float(np.dot(v0, v2))
        dot11 = float(np.dot(v1, v1))
        dot12 = float(np.dot(v1, v2))
        denom = dot00 * dot11 - dot01 * dot01
        u = (dot11 * dot02 - dot01 * dot12) / denom
        v = (dot00 * dot12 - dot01 * dot02) / denom
        return bool(u > 0.0 and v > 0.0 and u + v < 1.0)

    def validate_camera(self) -> None:
        position = self.camera_position
        if np.any(~np.isfinite(position)):
            raise business_error("E_SCENE_VALUE", "camera position is not finite")
        if position[1] <= self.ground_y:
            raise business_error(
                "E_CAMERA_MEDIUM", "camera is at or below the ground/water volume"
            )
        if self.point_inside_water(position):
            raise business_error("E_CAMERA_MEDIUM", "camera starts inside the water medium")
        if self.point_inside_prism(position):
            raise business_error("E_CAMERA_MEDIUM", "camera starts inside the glass prism")


def build_scene(name: str) -> Scene:
    data = scene_definition(name)
    validate_scene_data(data)
    spectral = data["spectral"]
    scene = Scene(
        data=data,
        min_nm=float(spectral["min_nm"]),
        max_nm=float(spectral["max_nm"]),
        camera_position=np.zeros(3),
        camera_target=np.zeros(3),
        camera_up=np.zeros(3),
        camera_forward=np.zeros(3),
        camera_right=np.zeros(3),
        camera_true_up=np.zeros(3),
        camera_tan_half_fov=0.0,
    )
    scene.validate_camera()
    return scene

