from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .mathutil import normalize_single
from .scene import Camera, Scene


@dataclass
class ViewCamera:
    position: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    tan_half_fov: float
    aspect: float
    width: int
    height: int
    near: float
    far: float

    def view_points(self, points: np.ndarray) -> np.ndarray:
        delta = np.asarray(points, dtype=np.float64) - self.position
        return np.stack(
            [
                np.sum(delta * self.right, axis=-1),
                np.sum(delta * self.up, axis=-1),
                np.sum(delta * self.forward, axis=-1),
            ],
            axis=-1,
        )

    def project_view_point(self, view_point: np.ndarray) -> np.ndarray:
        depth = float(view_point[2])
        ndc_x = float(view_point[0]) / (depth * self.tan_half_fov * self.aspect)
        ndc_y = float(view_point[1]) / (depth * self.tan_half_fov)
        return np.array(
            [
                (ndc_x * 0.5 + 0.5) * self.width,
                (0.5 - ndc_y * 0.5) * self.height,
            ],
            dtype=np.float64,
        )

    def ndc_to_pixels(self, ndc_x: np.ndarray, ndc_y: np.ndarray) -> np.ndarray:
        x = (np.asarray(ndc_x) * 0.5 + 0.5) * self.width
        y = (0.5 - np.asarray(ndc_y) * 0.5) * self.height
        return np.stack([x, y], axis=-1)

    def view_direction_to_face_uv(
        self, direction: np.ndarray
    ) -> np.ndarray:
        direction = np.asarray(direction, dtype=np.float64)
        depth = np.sum(direction * self.forward, axis=-1)
        x = np.sum(direction * self.right, axis=-1) / np.maximum(depth, 1.0e-12)
        y = np.sum(direction * self.up, axis=-1) / np.maximum(depth, 1.0e-12)
        ndc_x = x / (self.tan_half_fov * self.aspect)
        ndc_y = y / self.tan_half_fov
        return self.ndc_to_pixels(ndc_x, ndc_y)


def make_view_camera(camera: Camera, width: int, height: int) -> ViewCamera:
    forward = normalize_single(camera.target - camera.position)
    right = normalize_single(np.cross(forward, camera.up))
    up = normalize_single(np.cross(right, forward))
    return ViewCamera(
        position=np.asarray(camera.position, dtype=np.float64),
        right=right,
        up=up,
        forward=forward,
        tan_half_fov=float(np.tan(np.deg2rad(camera.fov_y_deg) * 0.5)),
        aspect=float(width) / float(height),
        width=width,
        height=height,
        near=camera.near,
        far=camera.far,
    )


def _interpolate_vertex(
    first: Dict[str, np.ndarray], second: Dict[str, np.ndarray], amount: float
) -> Dict[str, np.ndarray]:
    result = {
        key: first[key] + (second[key] - first[key]) * amount
        for key in ("world", "normal", "uv", "view", "depth")
    }
    return result


def _clip_near(vertices: Sequence[Dict[str, np.ndarray]], near: float) -> List[Dict[str, np.ndarray]]:
    if not vertices:
        return []
    output: List[Dict[str, np.ndarray]] = []
    previous = vertices[-1]
    previous_distance = float(previous["depth"]) - near
    for current in vertices:
        current_distance = float(current["depth"]) - near
        if current_distance >= 0.0:
            if previous_distance < 0.0:
                amount = previous_distance / (previous_distance - current_distance)
                output.append(_interpolate_vertex(previous, current, amount))
            output.append(current)
        elif previous_distance >= 0.0:
            amount = previous_distance / (previous_distance - current_distance)
            output.append(_interpolate_vertex(previous, current, amount))
        previous = current
        previous_distance = current_distance
    return output


def prepare_screen_triangles(
    scene: Scene,
    camera: ViewCamera,
    material_ids: Optional[Iterable[int]] = None,
) -> List[Dict[str, object]]:
    selected = None if material_ids is None else set(int(value) for value in material_ids)
    result: List[Dict[str, object]] = []
    for triangle_index, triangle in enumerate(scene.triangles):
        material_id = int(scene.triangle_materials[triangle_index])
        if selected is not None and material_id not in selected:
            continue
        indices = [int(triangle[0]), int(triangle[1]), int(triangle[2])]
        world = scene.positions[indices]
        normals = scene.normals[indices]
        uvs = scene.uvs[indices]
        view = camera.view_points(world)
        vertices = [
            {
                "world": world[i],
                "normal": normals[i],
                "uv": uvs[i],
                "view": view[i],
                "depth": view[i, 2],
            }
            for i in range(3)
        ]
        clipped = _clip_near(vertices, camera.near)
        if len(clipped) < 3:
            continue
        for fan_index in range(1, len(clipped) - 1):
            polygon = [clipped[0], clipped[fan_index], clipped[fan_index + 1]]
            screen = np.asarray(
                [camera.project_view_point(vertex["view"]) for vertex in polygon],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(screen)):
                continue
            result.append(
                {
                    "screen": screen,
                    "depth": np.asarray([vertex["depth"] for vertex in polygon], dtype=np.float64),
                    "world": np.asarray([vertex["world"] for vertex in polygon], dtype=np.float64),
                    "normal": np.asarray([vertex["normal"] for vertex in polygon], dtype=np.float64),
                    "uv": np.asarray([vertex["uv"] for vertex in polygon], dtype=np.float64),
                    "material": material_id,
                    "order": triangle_index,
                }
            )
    return result


def barycentric_coordinates(
    screen: np.ndarray, pixel_x: np.ndarray, pixel_y: np.ndarray
) -> np.ndarray:
    a, b, c = screen[0], screen[1], screen[2]
    denominator = (
        (b[1] - c[1]) * (a[0] - c[0])
        + (c[0] - b[0]) * (a[1] - c[1])
    )
    if abs(float(denominator)) <= 1.0e-14:
        return np.zeros(pixel_x.shape + (3,), dtype=np.float64)
    inverse = 1.0 / denominator
    w0 = (
        (b[1] - c[1]) * (pixel_x - c[0])
        + (c[0] - b[0]) * (pixel_y - c[1])
    ) * inverse
    w1 = (
        (c[1] - a[1]) * (pixel_x - c[0])
        + (a[0] - c[0]) * (pixel_y - c[1])
    ) * inverse
    w2 = 1.0 - w0 - w1
    return np.stack([w0, w1, w2], axis=-1)


def perspective_weights(screen_weights: np.ndarray, depth: np.ndarray) -> np.ndarray:
    inverse_depth = 1.0 / depth
    denominator = np.sum(screen_weights * inverse_depth, axis=-1, keepdims=True)
    return (screen_weights * inverse_depth) / np.maximum(denominator, 1.0e-30)
