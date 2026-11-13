"""Deterministic directional and point-light shadow maps with fixed-order PCF."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .math3d import (
    Vec3,
    aabb_corners,
    add,
    dot,
    length,
    make_basis_from_forward,
    normalize,
    scale,
    sub,
)
from .scene import Light, Scene


@dataclass(frozen=True)
class DirectionalShadowMap:
    light_id: str
    resolution: int
    eye: Vec3
    right: Vec3
    up: Vec3
    forward: Vec3
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    depths: tuple[float, ...]

    def project(self, world_point: Vec3) -> tuple[float, float, float]:
        relative = sub(world_point, self.eye)
        camera_x = dot(relative, self.right)
        camera_y = dot(relative, self.up)
        camera_z = dot(relative, self.forward)
        inverse_width = 1.0 / (self.max_x - self.min_x)
        inverse_height = 1.0 / (self.max_y - self.min_y)
        inverse_depth = 1.0 / (self.max_z - self.min_z)
        u = (camera_x - self.min_x) * inverse_width
        v = (self.max_y - camera_y) * inverse_height
        depth = (camera_z - self.min_z) * inverse_depth
        return u, v, depth


@dataclass(frozen=True)
class PointShadowFace:
    forward: Vec3
    right: Vec3
    up: Vec3
    depths: tuple[float, ...]


@dataclass(frozen=True)
class PointShadowMap:
    light_id: str
    position: Vec3
    far: float
    resolution: int
    faces: tuple[PointShadowFace, ...]

    def select_face(self, world_point: Vec3) -> tuple[PointShadowFace, float, float, float] | None:
        direction = normalize(sub(world_point, self.position))
        selected = 0
        selected_dot = -2.0
        for face_index, face in enumerate(self.faces):
            face_dot = dot(direction, face.forward)
            if face_dot > selected_dot:
                selected_dot = face_dot
                selected = face_index
        face = self.faces[selected]
        relative = sub(world_point, self.position)
        camera_x = dot(relative, face.right)
        camera_y = dot(relative, face.up)
        camera_z = dot(relative, face.forward)
        if camera_z <= 0.0:
            return None
        u = 0.5 + camera_x / camera_z * 0.5
        v = 0.5 - camera_y / camera_z * 0.5
        return face, u, v, camera_z / self.far


ShadowMaps = dict[str, DirectionalShadowMap | PointShadowMap]


def estimate_tile_shadow_bytes(scene: Scene) -> int:
    total = 0
    for light in scene.lights:
        resolution = scene.shadow.effective_resolution(light)
        face_count = 6 if light.kind == "point" else 1
        total += resolution * resolution * face_count * 8
    return total


def _scene_center(scene: Scene) -> Vec3:
    return (
        (scene.bounds_min[0] + scene.bounds_max[0]) * 0.5,
        (scene.bounds_min[1] + scene.bounds_max[1]) * 0.5,
        (scene.bounds_min[2] + scene.bounds_max[2]) * 0.5,
    )


def _scene_radius(scene: Scene) -> float:
    return max(0.5, length(sub(scene.bounds_max, scene.bounds_min)) * 0.5)


@dataclass(frozen=True)
class _ShadowVertex:
    camera: Vec3
    depth: float


def _clip_shadow_polygon(vertices: list[_ShadowVertex], minimum: float, maximum: float) -> list[_ShadowVertex]:
    def clip(plane_depth: float, keep_greater: bool) -> list[_ShadowVertex]:
        if not vertices:
            return []
        result: list[_ShadowVertex] = []
        previous = vertices[-1]
        previous_inside = previous.camera[2] >= plane_depth if keep_greater else previous.camera[2] <= plane_depth
        for current in vertices:
            current_inside = current.camera[2] >= plane_depth if keep_greater else current.camera[2] <= plane_depth
            if current_inside != previous_inside:
                denominator = previous.camera[2] - current.camera[2]
                amount = 0.0 if abs(denominator) <= 1.0e-15 else (previous.camera[2] - plane_depth) / denominator
                interpolated = _ShadowVertex(
                    camera=(
                        previous.camera[0] + (current.camera[0] - previous.camera[0]) * amount,
                        previous.camera[1] + (current.camera[1] - previous.camera[1]) * amount,
                        plane_depth,
                    ),
                    depth=previous.depth + (current.depth - previous.depth) * amount,
                )
                result.append(interpolated)
            if current_inside:
                result.append(current)
            previous = current
            previous_inside = current_inside
        return result

    clipped = clip(minimum, True)
    local_vertices = clipped
    if not local_vertices:
        return []
    result = []
    previous = local_vertices[-1]
    previous_inside = previous.camera[2] <= maximum
    for current in local_vertices:
        current_inside = current.camera[2] <= maximum
        if current_inside != previous_inside:
            denominator = previous.camera[2] - current.camera[2]
            amount = 0.0 if abs(denominator) <= 1.0e-15 else (previous.camera[2] - maximum) / denominator
            result.append(
                _ShadowVertex(
                    camera=(
                        previous.camera[0] + (current.camera[0] - previous.camera[0]) * amount,
                        previous.camera[1] + (current.camera[1] - previous.camera[1]) * amount,
                        maximum,
                    ),
                    depth=previous.depth + (current.depth - previous.depth) * amount,
                )
            )
        if current_inside:
            result.append(current)
        previous = current
        previous_inside = current_inside
    return result


def _edge(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> float:
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax)


def _rasterize_shadow_polygon(
    vertices: list[_ShadowVertex],
    resolution: int,
    project,
    perspective: bool,
    depth_buffer: list[float],
) -> None:
    projected = []
    for vertex in vertices:
        screen_x, screen_y = project(vertex)
        projected.append((screen_x, screen_y, vertex.camera[2], vertex.depth))
    for fan_offset in range(1, len(projected) - 1):
        first = projected[0]
        second = projected[fan_offset]
        third = projected[fan_offset + 1]
        area = _edge(first[0], first[1], second[0], second[1], third[0], third[1])
        if abs(area) <= 1.0e-12:
            continue
        min_x = max(0, int(math.floor(min(first[0], second[0], third[0]))))
        max_x = min(resolution - 1, int(math.ceil(max(first[0], second[0], third[0]))))
        min_y = max(0, int(math.floor(min(first[1], second[1], third[1]))))
        max_y = min(resolution - 1, int(math.ceil(max(first[1], second[1], third[1]))))
        if min_x > max_x or min_y > max_y:
            continue
        positive = area > 0.0
        inverse_area = 1.0 / area
        for pixel_y in range(min_y, max_y + 1):
            sample_y = float(pixel_y) + 0.5
            row_offset = pixel_y * resolution
            for pixel_x in range(min_x, max_x + 1):
                sample_x = float(pixel_x) + 0.5
                edge_a = _edge(second[0], second[1], third[0], third[1], sample_x, sample_y)
                edge_b = _edge(third[0], third[1], first[0], first[1], sample_x, sample_y)
                edge_c = _edge(first[0], first[1], second[0], second[1], sample_x, sample_y)
                if positive:
                    if edge_a < 0.0 or edge_b < 0.0 or edge_c < 0.0:
                        continue
                elif edge_a > 0.0 or edge_b > 0.0 or edge_c > 0.0:
                    continue
                bary_a = edge_a * inverse_area
                bary_b = edge_b * inverse_area
                bary_c = edge_c * inverse_area
                if perspective:
                    inverse_depth = bary_a / first[2] + bary_b / second[2] + bary_c / third[2]
                    if inverse_depth <= 1.0e-15:
                        continue
                    depth = (
                        bary_a * first[3] / first[2]
                        + bary_b * second[3] / second[2]
                        + bary_c * third[3] / third[2]
                    ) / inverse_depth
                else:
                    depth = bary_a * first[3] + bary_b * second[3] + bary_c * third[3]
                buffer_index = row_offset + pixel_x
                if depth < depth_buffer[buffer_index]:
                    depth_buffer[buffer_index] = depth


def _build_directional(scene: Scene, light: Light) -> DirectionalShadowMap:
    resolution = scene.shadow.effective_resolution(light)
    center = _scene_center(scene)
    radius = _scene_radius(scene)
    eye = sub(center, scale(light.direction, radius * 3.0 + 2.0))
    up_hint = (0.0, 1.0, 0.0)
    if abs(dot(light.direction, up_hint)) > 0.99:
        up_hint = (0.0, 0.0, 1.0)
    basis = make_basis_from_forward(eye, light.direction, up_hint)
    camera_corners = [basis.to_camera(corner) for corner in aabb_corners(scene.bounds_min, scene.bounds_max)]
    minimum_x = min(corner[0] for corner in camera_corners)
    maximum_x = max(corner[0] for corner in camera_corners)
    minimum_y = min(corner[1] for corner in camera_corners)
    maximum_y = max(corner[1] for corner in camera_corners)
    minimum_z = min(corner[2] for corner in camera_corners)
    maximum_z = max(corner[2] for corner in camera_corners)
    padding_xy = max(0.05, radius * 0.03)
    padding_z = max(0.05, radius * 0.03)
    minimum_x -= padding_xy
    maximum_x += padding_xy
    minimum_y -= padding_xy
    maximum_y += padding_xy
    minimum_z -= padding_z
    maximum_z += padding_z
    if maximum_x - minimum_x <= 1.0e-9:
        maximum_x = minimum_x + 1.0
    if maximum_y - minimum_y <= 1.0e-9:
        maximum_y = minimum_y + 1.0
    if maximum_z - minimum_z <= 1.0e-9:
        maximum_z = minimum_z + 1.0
    if minimum_z <= 0.0:
        minimum_z = 1.0e-4
        maximum_z = max(maximum_z, minimum_z + 1.0)
    depth_buffer = [1.0] * (resolution * resolution)
    width = maximum_x - minimum_x
    height = maximum_y - minimum_y
    depth_range = maximum_z - minimum_z

    def project(vertex: _ShadowVertex) -> tuple[float, float]:
        camera_x, camera_y, camera_z = vertex.camera
        screen_x = (camera_x - minimum_x) / width * resolution
        screen_y = (maximum_y - camera_y) / height * resolution
        return screen_x, screen_y

    for triangle in scene.triangles:
        camera_vertices = [
            _ShadowVertex(
                camera=basis.to_camera(position),
                depth=0.0,
            )
            for position in triangle.positions
        ]
        vertices = []
        for vertex in camera_vertices:
            normalized_depth = (vertex.camera[2] - minimum_z) / depth_range
            vertices.append(_ShadowVertex(vertex.camera, normalized_depth))
        clipped = _clip_shadow_polygon(vertices, minimum_z, maximum_z)
        if len(clipped) >= 3:
            _rasterize_shadow_polygon(clipped, resolution, project, False, depth_buffer)
    return DirectionalShadowMap(
        light_id=light.light_id,
        resolution=resolution,
        eye=eye,
        right=basis.right,
        up=basis.up,
        forward=basis.forward,
        min_x=minimum_x,
        max_x=maximum_x,
        min_y=minimum_y,
        max_y=maximum_y,
        min_z=minimum_z,
        max_z=maximum_z,
        depths=tuple(depth_buffer),
    )


_POINT_FACE_BASES: tuple[tuple[Vec3, Vec3, Vec3], ...] = (
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
    ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
)


def _point_far_distance(scene: Scene, light: Light) -> float:
    corners = aabb_corners(scene.bounds_min, scene.bounds_max)
    maximum_distance = max(length(sub(corner, light.position)) for corner in corners)
    if light.range is not None:
        return max(1.0e-4, light.range)
    return max(1.0e-4, maximum_distance * 1.05 + 0.1)


def _build_point(scene: Scene, light: Light) -> PointShadowMap:
    resolution = scene.shadow.effective_resolution(light)
    far = _point_far_distance(scene, light)
    near = max(1.0e-4, far * 0.0005)
    faces: list[PointShadowFace] = []
    for forward, right, up in _POINT_FACE_BASES:
        depth_buffer = [1.0] * (resolution * resolution)

        def project(vertex: _ShadowVertex, face_right: Vec3 = right, face_up: Vec3 = up) -> tuple[float, float]:
            camera_x, camera_y, camera_z = vertex.camera
            screen_x = (0.5 + camera_x / camera_z * 0.5) * resolution
            screen_y = (0.5 - camera_y / camera_z * 0.5) * resolution
            return screen_x, screen_y

        for triangle in scene.triangles:
            relative_vertices = []
            for position in triangle.positions:
                relative = sub(position, light.position)
                camera_point = (
                    dot(relative, right),
                    dot(relative, up),
                    dot(relative, forward),
                )
                relative_vertices.append(
                    _ShadowVertex(
                        camera=camera_point,
                        depth=dot(relative, forward) / far,
                    )
                )
            clipped = _clip_shadow_polygon(relative_vertices, near, far)
            if len(clipped) >= 3:
                _rasterize_shadow_polygon(clipped, resolution, project, True, depth_buffer)
        faces.append(PointShadowFace(forward=forward, right=right, up=up, depths=tuple(depth_buffer)))
    return PointShadowMap(
        light_id=light.light_id,
        position=light.position,
        far=far,
        resolution=resolution,
        faces=tuple(faces),
    )


def build_shadow_maps(scene: Scene) -> ShadowMaps:
    maps: ShadowMaps = {}
    for light in scene.lights:
        if light.kind == "point":
            maps[light.light_id] = _build_point(scene, light)
        else:
            maps[light.light_id] = _build_directional(scene, light)
    return maps


def _sample_depth(depths: tuple[float, ...], resolution: int, x: int, y: int) -> float:
    clamped_x = min(resolution - 1, max(0, x))
    clamped_y = min(resolution - 1, max(0, y))
    return depths[clamped_y * resolution + clamped_x]


def _pcf_visibility(
    depths: tuple[float, ...],
    resolution: int,
    u: float,
    v: float,
    compare_depth: float,
    kernel: int,
) -> float:
    center_x = int(math.floor(u * resolution))
    center_y = int(math.floor(v * resolution))
    radius = kernel // 2
    lit_count = 0
    inverse_sample_count = 1.0 / float(kernel * kernel)
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            stored_depth = _sample_depth(depths, resolution, center_x + offset_x, center_y + offset_y)
            if compare_depth <= stored_depth:
                lit_count += 1
    return lit_count * inverse_sample_count


def _directional_visibility(
    scene: Scene,
    light: Light,
    shadow_map: DirectionalShadowMap,
    world_position: Vec3,
    world_normal: Vec3,
    normal_dot_light: float,
) -> float:
    normal_bias = scene.shadow.effective_normal_bias(light)
    sample_position = add(world_position, scale(world_normal, normal_bias))
    u, v, fragment_depth = shadow_map.project(sample_position)
    if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
        return 1.0
    depth_extent = max(1.0e-12, shadow_map.max_z - shadow_map.min_z)
    kernel = scene.shadow.effective_pcf_kernel(light)
    texel_world_size = max(
        (shadow_map.max_x - shadow_map.min_x) / float(shadow_map.resolution),
        (shadow_map.max_y - shadow_map.min_y) / float(shadow_map.resolution),
    )
    clamped_ndotl = max(0.12, min(1.0, normal_dot_light))
    slope = math.sqrt(max(0.0, 1.0 - clamped_ndotl * clamped_ndotl)) / clamped_ndotl
    slope_bias = texel_world_size * (float(kernel // 2) + 0.75) * slope / depth_extent
    depth_bias = (
        scene.shadow.effective_bias(light) / depth_extent
        + scene.shadow.depth_epsilon
        + (1.0 - max(0.0, min(1.0, normal_dot_light))) * scene.shadow.depth_epsilon
        + slope_bias
    )
    return _pcf_visibility(
        shadow_map.depths,
        shadow_map.resolution,
        u,
        v,
        fragment_depth - depth_bias,
        kernel,
    )


def _point_visibility(
    scene: Scene,
    light: Light,
    shadow_map: PointShadowMap,
    world_position: Vec3,
    world_normal: Vec3,
    normal_dot_light: float,
) -> float:
    normal_bias = scene.shadow.effective_normal_bias(light)
    sample_position = add(world_position, scale(world_normal, normal_bias))
    selected = shadow_map.select_face(sample_position)
    if selected is None:
        return 1.0
    face, u, v, fragment_depth = selected
    if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
        return 1.0
    depth_bias = (
        scene.shadow.effective_bias(light) / shadow_map.far
        + scene.shadow.depth_epsilon
        + (1.0 - max(0.0, min(1.0, normal_dot_light))) * scene.shadow.depth_epsilon
    )
    kernel = scene.shadow.effective_pcf_kernel(light)
    distance_from_light = length(sub(sample_position, shadow_map.position))
    texel_world_size = 2.0 * distance_from_light / float(shadow_map.resolution)
    clamped_ndotl = max(0.12, min(1.0, normal_dot_light))
    slope = math.sqrt(max(0.0, 1.0 - clamped_ndotl * clamped_ndotl)) / clamped_ndotl
    depth_bias += texel_world_size * (float(kernel // 2) + 0.75) * slope / shadow_map.far
    return _pcf_visibility(
        face.depths,
        shadow_map.resolution,
        u,
        v,
        fragment_depth - depth_bias,
        kernel,
    )


def shadow_visibility(
    scene: Scene,
    light: Light,
    shadow_maps: ShadowMaps,
    world_position: Vec3,
    world_normal: Vec3,
    normal_dot_light: float,
) -> float:
    shadow_map = shadow_maps.get(light.light_id)
    if shadow_map is None:
        return 1.0
    if isinstance(shadow_map, DirectionalShadowMap):
        return _directional_visibility(scene, light, shadow_map, world_position, world_normal, normal_dot_light)
    return _point_visibility(scene, light, shadow_map, world_position, world_normal, normal_dot_light)
