from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import ClusteringConfig, RenderConfig, SceneConfig
from .geometry import MeshBuilder, MeshData

DEFAULT_SEED = 196


def _polygon_area(points: np.ndarray) -> float:
    following = np.roll(points, -1, axis=0)
    return 0.5 * float(np.sum(points[:, 0] * following[:, 1] - points[:, 1] * following[:, 0]))


def _ensure_ccw(points: np.ndarray) -> np.ndarray:
    return points if _polygon_area(points) > 0.0 else points[::-1].copy()


def _point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    def cross(u: np.ndarray, v: np.ndarray) -> float:
        return float(u[0] * v[1] - u[1] * v[0])

    ab = cross(b - a, p - a)
    bc = cross(c - b, p - b)
    ca = cross(a - c, p - c)
    return ab >= -1.0e-10 and bc >= -1.0e-10 and ca >= -1.0e-10


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    x, z = point
    inside = False
    for index in range(len(polygon)):
        x1, z1 = polygon[index]
        x2, z2 = polygon[(index + 1) % len(polygon)]
        if (z1 > z) != (z2 > z):
            crossing_x = (x2 - x1) * (z - z1) / (z2 - z1) + x1
            if x < crossing_x:
                inside = not inside
    return inside


def _inset_point_in_polygon(
    point: tuple[float, float],
    polygon: np.ndarray,
    margin: float,
) -> bool:
    if not point_in_polygon(point, polygon):
        return False
    x, z = point
    for index in range(len(polygon)):
        x1, z1 = polygon[index]
        x2, z2 = polygon[(index + 1) % len(polygon)]
        edge_x = x2 - x1
        edge_z = z2 - z1
        edge_length = float(np.hypot(edge_x, edge_z))
        if edge_length <= 1.0e-9:
            continue
        if abs(edge_z * (x - x1) - edge_x * (z - z1)) / edge_length < margin:
            return False
    return True


def triangulate_polygon(points: np.ndarray) -> list[tuple[int, int, int]]:
    polygon = _ensure_ccw(np.asarray(points, dtype=np.float64))
    indices = list(range(len(polygon)))
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(indices) > 3:
        guard += 1
        if guard > 10000:
            raise ValueError("polygon triangulation did not converge")
        clipped = False
        for cursor in range(len(indices)):
            previous = indices[(cursor - 1) % len(indices)]
            current = indices[cursor]
            following = indices[(cursor + 1) % len(indices)]
            a, b, c = polygon[previous], polygon[current], polygon[following]
            turn = float((b - a)[0] * (c - b)[1] - (b - a)[1] * (c - b)[0])
            if turn <= 1.0e-10:
                continue
            contains = False
            for candidate in indices:
                if candidate in {previous, current, following}:
                    continue
                if _point_in_triangle(polygon[candidate], a, b, c):
                    contains = True
                    break
            if contains:
                continue
            triangles.append((previous, current, following))
            indices.pop(cursor)
            clipped = True
            break
        if not clipped:
            raise ValueError("polygon is not a simple polygon")
    triangles.append(tuple(indices))
    return triangles


def _rotate_y(points: np.ndarray, yaw: float) -> np.ndarray:
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    rotation = np.asarray(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
        dtype=np.float64,
    )
    return points @ rotation.T


def add_box(
    builder: MeshBuilder,
    material_index: int,
    material_name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    yaw: float = 0.0,
) -> None:
    center_array = np.asarray(center, dtype=np.float64)
    half = np.asarray(size, dtype=np.float64) * 0.5
    signs = np.asarray(
        [
            (-1, -1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
        ],
        dtype=np.float64,
    )
    corners = signs * half
    corners = _rotate_y(corners, yaw) + center_array
    faces = (
        ((0, 1, 2, 3), (0.0, 0.0, -1.0)),
        ((4, 7, 6, 5), (0.0, 0.0, 1.0)),
        ((0, 4, 5, 1), (0.0, -1.0, 0.0)),
        ((3, 2, 6, 7), (0.0, 1.0, 0.0)),
        ((0, 3, 7, 4), (-1.0, 0.0, 0.0)),
        ((1, 5, 6, 2), (1.0, 0.0, 0.0)),
    )
    for face, expected in faces:
        builder.add_quad(
            material_index,
            material_name,
            corners[list(face)],
            _rotate_y(np.asarray(expected)[None, :], yaw)[0],
        )


def add_cylinder(
    builder: MeshBuilder,
    material_index: int,
    material_name: str,
    base_center: tuple[float, float, float],
    radius: float,
    height: float,
    segments: int = 16,
) -> None:
    base = np.asarray(base_center, dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = np.column_stack([np.cos(angles) * radius, np.zeros(segments), np.sin(angles) * radius])
    for index in range(segments):
        next_index = (index + 1) % segments
        lower_a = base + ring[index]
        lower_b = base + ring[next_index]
        upper_a = lower_a + np.asarray((0.0, height, 0.0))
        upper_b = lower_b + np.asarray((0.0, height, 0.0))
        normal_a = np.asarray((ring[index, 0], 0.0, ring[index, 2]))
        normal_b = np.asarray((ring[next_index, 0], 0.0, ring[next_index, 2]))
        normal_a /= max(1.0e-12, float(np.linalg.norm(normal_a)))
        normal_b /= max(1.0e-12, float(np.linalg.norm(normal_b)))
        builder.add_triangle(
            material_index,
            material_name,
            np.asarray((lower_a, lower_b, upper_b)),
            np.asarray((normal_a, normal_b, normal_b)),
        )
        builder.add_triangle(
            material_index,
            material_name,
            np.asarray((lower_a, upper_b, upper_a)),
            np.asarray((normal_a, normal_b, normal_a)),
        )
    top = base + np.asarray((0.0, height, 0.0))
    for index in range(segments):
        next_index = (index + 1) % segments
        builder.add_triangle(
            material_index,
            material_name,
            np.asarray((top, top + ring[index], top + ring[next_index])),
            np.tile(np.asarray((0.0, 1.0, 0.0)), (3, 1)),
        )


def _room_polygon(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed ^ 0x5A17)
    base = np.asarray(
        (
            (-4.2, -3.2),
            (3.8, -3.2),
            (4.6, -1.6),
            (4.6, 2.8),
            (1.5, 3.4),
            (1.0, 2.4),
            (-2.0, 3.4),
            (-4.4, 1.2),
        ),
        dtype=np.float64,
    )
    jitter = rng.uniform(-0.08, 0.08, size=base.shape)
    polygon = base + jitter
    return _ensure_ccw(np.round(polygon, 5))


def _build_room(builder: MeshBuilder, polygon: np.ndarray, height: float) -> None:
    floor_triangles = triangulate_polygon(polygon)
    for triangle in floor_triangles:
        points = np.column_stack(
            [polygon[list(triangle), 0], np.zeros(3), polygon[list(triangle), 1]]
        )
        builder.add_triangle(
            0,
            "plaster",
            points,
            np.tile(np.asarray((0.0, 1.0, 0.0)), (3, 1)),
        )
        ceiling_points = points.copy()
        ceiling_points[:, 1] = height
        builder.add_triangle(
            0,
            "plaster",
            ceiling_points,
            np.tile(np.asarray((0.0, -1.0, 0.0)), (3, 1)),
        )

    for index in range(len(polygon)):
        point_a = polygon[index]
        point_b = polygon[(index + 1) % len(polygon)]
        edge = point_b - point_a
        inward = np.asarray((-edge[1], edge[0]), dtype=np.float64)
        inward /= max(1.0e-12, float(np.linalg.norm(inward)))
        points = np.asarray(
            (
                (point_a[0], 0.0, point_a[1]),
                (point_b[0], 0.0, point_b[1]),
                (point_b[0], height, point_b[1]),
                (point_a[0], height, point_a[1]),
            )
        )
        builder.add_quad(0, "plaster", points, np.asarray((inward[0], 0.0, inward[1])))


def _build_furniture(builder: MeshBuilder, seed: int) -> list[dict[str, Any]]:
    rng = np.random.RandomState(seed ^ 0xC0FFEE)
    objects: list[dict[str, Any]] = []

    rug_center = (0.1, 0.012, 0.2)
    add_box(builder, 2, "fabric", rug_center, (3.8, 0.024, 2.7), 0.08)
    objects.append({"name": "rug", "type": "box", "material": "fabric"})

    table_center = (-0.15, 0.0, 0.25)
    table_yaw = float(rng.uniform(-0.15, 0.15))
    add_box(builder, 3, "lacquer", (table_center[0], 0.74, table_center[2]), (2.2, 0.12, 1.15), table_yaw)
    for x_offset, z_offset in ((-0.88, -0.40), (0.88, -0.40), (-0.88, 0.40), (0.88, 0.40)):
        rotated = _rotate_y(np.asarray(((x_offset, 0.0, z_offset),)), table_yaw)[0]
        add_box(
            builder,
            1,
            "wood",
            (table_center[0] + rotated[0], 0.37, table_center[2] + rotated[2]),
            (0.13, 0.72, 0.13),
            table_yaw,
        )
    objects.append({"name": "table", "type": "box-group", "material": "lacquer/wood"})

    chair_positions = (
        (-2.05, 0.15, 0.15, -0.55),
        (1.85, 0.15, 0.05, 0.50),
        (-0.25, 0.15, -1.30, 0.05),
        (0.10, 0.15, 1.65, np.pi),
    )
    for chair_index, (x, y, z, yaw) in enumerate(chair_positions):
        yaw += float(rng.uniform(-0.08, 0.08))
        add_box(builder, 1, "wood", (x, y + 0.43, z), (0.62, 0.10, 0.62), yaw)
        back_offset = _rotate_y(np.asarray(((0.0, 0.0, -0.26),)), yaw)[0]
        add_box(
            builder,
            1,
            "wood",
            (x + back_offset[0], y + 0.77, z + back_offset[2]),
            (0.62, 0.62, 0.09),
            yaw,
        )
        for local_x, local_z in ((-0.24, -0.24), (0.24, -0.24), (-0.24, 0.24), (0.24, 0.24)):
            offset = _rotate_y(np.asarray(((local_x, 0.0, local_z),)), yaw)[0]
            add_box(
                builder,
                1,
                "wood",
                (x + offset[0], y + 0.21, z + offset[2]),
                (0.075, 0.42, 0.075),
                yaw,
            )
        objects.append({"name": f"chair-{chair_index}", "type": "box-group", "material": "wood"})

    sofa_center = (2.55, 0.0, 1.95)
    sofa_yaw = -0.42
    add_box(builder, 2, "fabric", (sofa_center[0], 0.38, sofa_center[2]), (2.2, 0.52, 0.88), sofa_yaw)
    add_box(builder, 2, "fabric", (sofa_center[0], 0.72, sofa_center[2] - 0.36), (2.2, 1.05, 0.18), sofa_yaw)
    objects.append({"name": "sofa", "type": "box-group", "material": "fabric"})

    add_box(builder, 1, "wood", (-3.62, 1.10, -0.75), (0.62, 2.20, 1.65), 0.06)
    objects.append({"name": "cabinet", "type": "box", "material": "wood"})

    lamp_base = (-0.15 + float(rng.uniform(-0.05, 0.05)), 0.80, 0.25)
    add_cylinder(builder, 4, "brass", lamp_base, 0.035, 0.42, 12)
    add_cylinder(builder, 4, "brass", (lamp_base[0], 1.22, lamp_base[2]), 0.32, 0.10, 20)
    objects.append({"name": "table-lamp", "type": "cylinder-group", "material": "brass"})
    return objects


def _make_materials() -> list[dict[str, Any]]:
    return [
        {
            "id": 0,
            "name": "plaster",
            "type": "Lambertian",
            "albedo": [0.72, 0.69, 0.62],
            "roughness": 0.92,
            "specular": 0.015,
            "shininess": 18.0,
        },
        {
            "id": 1,
            "name": "wood",
            "type": "Lambertian",
            "albedo": [0.31, 0.15, 0.075],
            "roughness": 0.78,
            "specular": 0.03,
            "shininess": 24.0,
        },
        {
            "id": 2,
            "name": "fabric",
            "type": "OrenNayar",
            "albedo": [0.18, 0.24, 0.30],
            "roughness": 0.86,
            "specular": 0.0,
            "shininess": 8.0,
        },
        {
            "id": 3,
            "name": "lacquer",
            "type": "BlinnPhong",
            "albedo": [0.30, 0.10, 0.045],
            "roughness": 0.32,
            "specular": 0.55,
            "shininess": 96.0,
        },
        {
            "id": 4,
            "name": "brass",
            "type": "BlinnPhong",
            "albedo": [0.72, 0.47, 0.13],
            "roughness": 0.22,
            "specular": 0.9,
            "shininess": 180.0,
        },
    ]


def _make_lights(seed: int, polygon: np.ndarray) -> list[dict[str, Any]]:
    rng = np.random.RandomState(seed ^ 0x11A57)
    palette = (
        (1.0, 0.72, 0.43),
        (1.0, 0.86, 0.67),
        (0.58, 0.74, 1.0),
        (0.86, 0.92, 1.0),
    )
    lights: list[dict[str, Any]] = []
    for index in range(24):
        if index < 16:
            for _attempt in range(512):
                angle = 2.0 * np.pi * index / 16.0 + float(rng.uniform(-0.18, 0.18))
                radius = float(rng.uniform(1.6, 3.6))
                candidate = (radius * np.cos(angle), radius * np.sin(angle))
                if _inset_point_in_polygon(candidate, polygon, 0.28):
                    break
            position = (
                round(float(candidate[0]), 5),
                round(float(rng.uniform(2.25, 2.75)), 5),
                round(float(candidate[1]), 5),
            )
            light_type = "point"
            direction = (0.0, -1.0, 0.0)
            spot_angle = 35.0
            spot_softness = 12.0
        else:
            for _attempt in range(512):
                angle = 2.0 * np.pi * (index - 16) / 8.0 + 0.35 + float(rng.uniform(-0.16, 0.16))
                radius = float(rng.uniform(1.4, 3.0))
                candidate = (radius * np.cos(angle), radius * np.sin(angle))
                if _inset_point_in_polygon(candidate, polygon, 0.28):
                    break
            position = (
                round(float(candidate[0]), 5),
                round(float(rng.uniform(2.40, 2.80)), 5),
                round(float(candidate[1]), 5),
            )
            light_type = "spot"
            direction = (
                round(-position[0] * 0.28, 5),
                -1.0,
                round(-position[2] * 0.28, 5),
            )
            spot_angle = 28.0
            spot_softness = 14.0
        color = palette[(index + int(seed)) % len(palette)]
        lights.append(
            {
                "id": f"light-{index:03d}",
                "type": light_type,
                "position": list(position),
                "direction": list(direction),
                "color": list(color),
                "intensity": round(float(rng.uniform(8.0, 24.0)), 5),
                "range": round(float(rng.uniform(3.8, 6.4)), 5),
                "spot_angle_deg": spot_angle,
                "spot_softness_deg": spot_softness,
            }
        )
    return lights


def generate_config_dict(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    polygon = _room_polygon(seed)
    return {
        "schema_version": 1,
        "seed": seed,
        "scene": {
            "room_polygon": [[round(float(x), 5), round(float(z), 5)] for x, z in polygon],
            "wall_height": 3.2,
            "description": "Deterministic concave polygon room with box/cylinder furniture",
        },
        "camera": {
            "position": [0.0, 1.65, 2.65],
            "target": [0.0, 1.05, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov_y_deg": 58.0,
            "near": 0.05,
            "far": 25.0,
        },
        "materials": _make_materials(),
        "lights": _make_lights(seed, polygon),
        "clustering": ClusteringConfig(
            max_vpls=16,
            max_lights=256,
            spatial_weight=1.0,
            orientation_weight=0.75,
            iterations=12,
            cube_size=128,
            tile_size=16,
            vpl_capacity=16,
        ).to_dict(),
        "render": RenderConfig().to_dict(),
    }


def generate_scene(
    seed: int = DEFAULT_SEED,
    diagnostics=None,
) -> tuple[SceneConfig, MeshData]:
    payload = generate_config_dict(seed)
    config = SceneConfig.from_dict(payload, diagnostics)
    polygon = np.asarray(payload["scene"]["room_polygon"], dtype=np.float64)
    builder = MeshBuilder()
    _build_room(builder, polygon, float(payload["scene"]["wall_height"]))
    _build_furniture(builder, seed)
    return config, builder.finalize()


def write_generated_scene(path: str | Path, seed: int = DEFAULT_SEED) -> SceneConfig:
    payload = generate_config_dict(seed)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return SceneConfig.from_dict(payload)

