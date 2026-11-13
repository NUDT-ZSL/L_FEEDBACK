"""Tile-local deferred G-buffer storage and deterministic software rasterization."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .errors import E_TILE_FAILED, PipelineError
from .math3d import Vec3, cross, normalize, sub, add, scale
from .scene import Scene, TileSpec, Triangle


INF_DEPTH = float("inf")


@dataclass
class GBufferTile:
    x: int
    y: int
    width: int
    height: int
    albedo: list[float]
    normal: list[float]
    depth: list[float]
    material_ids: list[int]
    primitive_ids: list[int]

    @classmethod
    def empty(cls, x: int, y: int, width: int, height: int) -> "GBufferTile":
        pixel_count = width * height
        return cls(
            x=x,
            y=y,
            width=width,
            height=height,
            albedo=[0.0] * (pixel_count * 3),
            normal=[0.0] * (pixel_count * 3),
            depth=[INF_DEPTH] * pixel_count,
            material_ids=[-1] * pixel_count,
            primitive_ids=[-1] * pixel_count,
        )

    def pixel_index(self, local_x: int, local_y: int) -> int:
        return local_y * self.width + local_x

    def coverage_indices(self) -> list[int]:
        return [index for index, value in enumerate(self.depth) if value != INF_DEPTH]

    def material_usage(self) -> tuple[int, ...]:
        used = sorted({material_id for material_id in self.material_ids if material_id >= 0})
        return tuple(used)

    def has_coverage(self) -> bool:
        for value in self.depth:
            if value != INF_DEPTH:
                return True
        return False

    def world_at(self, scene: Scene, local_x: int, local_y: int) -> Vec3:
        depth = self.depth[self.pixel_index(local_x, local_y)]
        screen_x = float(self.x + local_x) + 0.5
        screen_y = float(self.y + local_y) + 0.5
        return scene.camera.screen_to_world(screen_x, screen_y, depth)

    def memory_bytes(self) -> int:
        pixel_count = self.width * self.height
        return pixel_count * (3 * 4 + 3 * 4 + 4 + 4 + 4)

    def immutable_arrays(self) -> dict[str, tuple[float | int, ...]]:
        return {
            "albedo": tuple(self.albedo),
            "normal": tuple(self.normal),
            "depth": tuple(self.depth),
            "material_ids": tuple(self.material_ids),
            "primitive_ids": tuple(self.primitive_ids),
        }


@dataclass(frozen=True)
class RasterVertex:
    camera: Vec3
    world: Vec3
    normal: Vec3


def _intersect_camera_plane(left: RasterVertex, right: RasterVertex, plane_depth: float) -> RasterVertex:
    left_delta = left.camera[2] - plane_depth
    right_delta = right.camera[2] - plane_depth
    denominator = left_delta - right_delta
    if abs(denominator) <= 1.0e-15:
        amount = 0.0
    else:
        amount = left_delta / denominator
    return RasterVertex(
        camera=(
            left.camera[0] + (right.camera[0] - left.camera[0]) * amount,
            left.camera[1] + (right.camera[1] - left.camera[1]) * amount,
            plane_depth,
        ),
        world=(
            left.world[0] + (right.world[0] - left.world[0]) * amount,
            left.world[1] + (right.world[1] - left.world[1]) * amount,
            left.world[2] + (right.world[2] - left.world[2]) * amount,
        ),
        normal=normalize(
            (
                left.normal[0] + (right.normal[0] - left.normal[0]) * amount,
                left.normal[1] + (right.normal[1] - left.normal[1]) * amount,
                left.normal[2] + (right.normal[2] - left.normal[2]) * amount,
            )
        ),
    )


def _clip_polygon(vertices: list[RasterVertex], near: float, far: float) -> list[RasterVertex]:
    for plane_kind, plane_depth in (("near", near), ("far", far)):
        if not vertices:
            return []
        source = vertices
        result: list[RasterVertex] = []
        previous = source[-1]
        previous_inside = previous.camera[2] >= plane_depth if plane_kind == "near" else previous.camera[2] <= plane_depth
        for current in source:
            current_inside = current.camera[2] >= plane_depth if plane_kind == "near" else current.camera[2] <= plane_depth
            if current_inside != previous_inside:
                result.append(_intersect_camera_plane(previous, current, plane_depth))
            if current_inside:
                result.append(current)
            previous = current
            previous_inside = current_inside
        vertices = result
    return vertices


def _edge(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> float:
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax)


def _screen_triangle(scene: Scene, triangle: Triangle) -> list[tuple[float, float, float, RasterVertex]]:
    original = [
        RasterVertex(
            camera=scene.camera.basis.to_camera(triangle.positions[index]),
            world=triangle.positions[index],
            normal=triangle.normals[index],
        )
        for index in range(3)
    ]
    clipped = _clip_polygon(original, scene.camera.near, scene.camera.far)
    if len(clipped) < 3:
        return []
    projected: list[tuple[float, float, float, RasterVertex]] = []
    for vertex in clipped:
        screen_x, screen_y, depth = scene.camera.camera_to_screen_depth(vertex.camera)
        projected.append((screen_x, screen_y, depth, vertex))
    return projected


def rasterize_tile(scene: Scene, tile: TileSpec) -> GBufferTile:
    target = GBufferTile.empty(tile.x, tile.y, tile.width, tile.height)
    target_left = float(tile.x)
    target_top = float(tile.y)
    target_right = float(tile.x + tile.width)
    target_bottom = float(tile.y + tile.height)

    for triangle in scene.triangles:
        projected = _screen_triangle(scene, triangle)
        if len(projected) < 3:
            continue
        for fan_offset in range(1, len(projected) - 1):
            vertex_a = projected[0]
            vertex_b = projected[fan_offset]
            vertex_c = projected[fan_offset + 1]
            screen_a = (vertex_a[0], vertex_a[1])
            screen_b = (vertex_b[0], vertex_b[1])
            screen_c = (vertex_c[0], vertex_c[1])
            area = _edge(screen_a[0], screen_a[1], screen_b[0], screen_b[1], screen_c[0], screen_c[1])
            if abs(area) <= 1.0e-12:
                continue
            min_x = min(screen_a[0], screen_b[0], screen_c[0])
            max_x = max(screen_a[0], screen_b[0], screen_c[0])
            min_y = min(screen_a[1], screen_b[1], screen_c[1])
            max_y = max(screen_a[1], screen_b[1], screen_c[1])
            x_start = max(tile.x, int(math.floor(min_x)))
            x_end = min(tile.x + tile.width - 1, int(math.ceil(max_x)))
            y_start = max(tile.y, int(math.floor(min_y)))
            y_end = min(tile.y + tile.height - 1, int(math.ceil(max_y)))
            if x_start > x_end or y_start > y_end:
                continue
            if x_end < target_left or x_start >= target_right or y_end < target_top or y_start >= target_bottom:
                continue
            positive = area > 0.0
            inverse_area = 1.0 / area
            depth_a = vertex_a[2]
            depth_b = vertex_b[2]
            depth_c = vertex_c[2]
            normal_a = vertex_a[3].normal
            normal_b = vertex_b[3].normal
            normal_c = vertex_c[3].normal
            perspective = scene.camera.projection_kind == "perspective"

            for pixel_y in range(y_start, y_end + 1):
                sample_y = float(pixel_y) + 0.5
                for pixel_x in range(x_start, x_end + 1):
                    sample_x = float(pixel_x) + 0.5
                    edge_a = _edge(screen_b[0], screen_b[1], screen_c[0], screen_c[1], sample_x, sample_y)
                    edge_b = _edge(screen_c[0], screen_c[1], screen_a[0], screen_a[1], sample_x, sample_y)
                    edge_c = _edge(screen_a[0], screen_a[1], screen_b[0], screen_b[1], sample_x, sample_y)
                    if positive:
                        if edge_a < 0.0 or edge_b < 0.0 or edge_c < 0.0:
                            continue
                    elif edge_a > 0.0 or edge_b > 0.0 or edge_c > 0.0:
                        continue
                    bary_a = edge_a * inverse_area
                    bary_b = edge_b * inverse_area
                    bary_c = edge_c * inverse_area
                    if perspective:
                        inverse_depth = bary_a / depth_a + bary_b / depth_b + bary_c / depth_c
                        if inverse_depth <= 1.0e-15:
                            continue
                        pixel_depth = 1.0 / inverse_depth
                        normal = (
                            (bary_a * normal_a[0] / depth_a
                             + bary_b * normal_b[0] / depth_b
                             + bary_c * normal_c[0] / depth_c) * pixel_depth,
                            (bary_a * normal_a[1] / depth_a
                             + bary_b * normal_b[1] / depth_b
                             + bary_c * normal_c[1] / depth_c) * pixel_depth,
                            (bary_a * normal_a[2] / depth_a
                             + bary_b * normal_b[2] / depth_b
                             + bary_c * normal_c[2] / depth_c) * pixel_depth,
                        )
                    else:
                        pixel_depth = bary_a * depth_a + bary_b * depth_b + bary_c * depth_c
                        normal = (
                            bary_a * normal_a[0] + bary_b * normal_b[0] + bary_c * normal_c[0],
                            bary_a * normal_a[1] + bary_b * normal_b[1] + bary_c * normal_c[1],
                            bary_a * normal_a[2] + bary_b * normal_b[2] + bary_c * normal_c[2],
                        )
                    if pixel_depth < scene.camera.near or pixel_depth > scene.camera.far:
                        continue
                    local_x = pixel_x - tile.x
                    local_y = pixel_y - tile.y
                    index = target.pixel_index(local_x, local_y)
                    if pixel_depth < target.depth[index]:
                        target.depth[index] = pixel_depth
                        world_normal = normalize(normal)
                        target.normal[index * 3] = world_normal[0]
                        target.normal[index * 3 + 1] = world_normal[1]
                        target.normal[index * 3 + 2] = world_normal[2]
                        target.material_ids[index] = triangle.material_index
                        target.primitive_ids[index] = triangle.index
    return target


def material_shading(scene: Scene, gbuffer: GBufferTile) -> tuple[int, ...]:
    material_usage: set[int] = set()
    for local_y in range(gbuffer.height):
        for local_x in range(gbuffer.width):
            index = gbuffer.pixel_index(local_x, local_y)
            material_index = gbuffer.material_ids[index]
            if material_index < 0:
                continue
            material = scene.material(material_index)
            if material is None:
                raise PipelineError(E_TILE_FAILED, f"pixel references missing material {material_index}", "material_shading")
            if material.problem is not None:
                raise PipelineError(
                    E_TILE_FAILED,
                    f"invalid material {material.material_id!r}: {material.problem}",
                    "material_shading",
                )
            material_usage.add(material_index)
            gbuffer.albedo[index * 3] = material.albedo[0]
            gbuffer.albedo[index * 3 + 1] = material.albedo[1]
            gbuffer.albedo[index * 3 + 2] = material.albedo[2]
    return tuple(sorted(material_usage))


def geometric_normal(world_a: Vec3, world_b: Vec3, world_c: Vec3) -> Vec3:
    return normalize(cross(sub(world_b, world_a), sub(world_c, world_a)))


def offset_along_normal(position: Vec3, normal: Vec3, distance: float) -> Vec3:
    return add(position, scale(normal, distance))
