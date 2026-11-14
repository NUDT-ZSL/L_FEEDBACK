from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene import CameraSpec, MeshSpec, UVLayout


@dataclass(frozen=True)
class Mesh:
    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    triangles: np.ndarray


@dataclass(frozen=True)
class GBuffer:
    coverage: np.ndarray
    uv: np.ndarray
    normal: np.ndarray
    position: np.ndarray
    depth: np.ndarray
    camera_view_direction: tuple[float, float, float]


def _profile(mesh: MeshSpec, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wave = np.sin(2.0 * np.pi * mesh.profile_waves * v)
    shoulder = 0.78 + 0.22 * np.sin(np.pi * v)
    scale = shoulder * (1.0 + mesh.profile_amplitude * wave)
    radius_x = mesh.radius_x * scale
    radius_z = mesh.radius_z * scale
    d_shoulder = 0.22 * np.pi * np.cos(np.pi * v)
    d_scale = (
        d_shoulder * (1.0 + mesh.profile_amplitude * wave)
        + shoulder * mesh.profile_amplitude * 2.0 * np.pi * mesh.profile_waves * np.cos(2.0 * np.pi * mesh.profile_waves * v)
    )
    return radius_x, radius_z, mesh.radius_x * d_scale, mesh.radius_z * d_scale


def build_uv_mesh(mesh: MeshSpec) -> Mesh:
    """Build a closed-seam cylindrical character shell in UV parameter space.

    The last UV ring duplicates the first position with ``u == 1`` so texture
    maps have a deterministic seam without modulo interpolation artifacts.
    """

    u = np.linspace(0.0, 1.0, mesh.u_segments + 1, dtype=np.float64)
    v = np.linspace(0.0, 1.0, mesh.v_segments + 1, dtype=np.float64)
    # Store vertices with v as the outer axis so a flat index is
    # ``iv * (u_segments + 1) + iu``. Triangle construction below relies on
    # exactly this layout.
    vv, uu = np.meshgrid(v, u, indexing="ij")
    theta = 2.0 * np.pi * uu
    radius_x, radius_z, drx_dv, drz_dv = _profile(mesh, vv)
    y = (vv - 0.5) * mesh.height

    positions = np.stack(
        (
            radius_x * np.sin(theta),
            y,
            radius_z * np.cos(theta),
        ),
        axis=-1,
    )

    dtheta = np.stack(
        (
            radius_x * np.cos(theta),
            np.zeros_like(theta),
            -radius_z * np.sin(theta),
        ),
        axis=-1,
    )
    dv = np.stack(
        (
            drx_dv * np.sin(theta),
            np.full_like(theta, mesh.height),
            drz_dv * np.cos(theta),
        ),
        axis=-1,
    )
    normals = np.cross(dtheta, dv)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-15)
    uvs = np.stack((uu, vv), axis=-1)

    def vertex_index(iu: int, iv: int) -> int:
        return iv * (mesh.u_segments + 1) + iu

    triangles: list[tuple[int, int, int]] = []
    for iv in range(mesh.v_segments):
        for iu in range(mesh.u_segments):
            a = vertex_index(iu, iv)
            b = vertex_index(iu + 1, iv)
            c = vertex_index(iu, iv + 1)
            d = vertex_index(iu + 1, iv + 1)
            triangles.append((a, b, d))
            triangles.append((a, d, c))
    return Mesh(
        positions=positions.reshape(-1, 3),
        normals=normals.reshape(-1, 3),
        uvs=uvs.reshape(-1, 2),
        triangles=np.asarray(triangles, dtype=np.int32),
    )


def camera_basis(camera: CameraSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yaw = np.deg2rad(camera.yaw_deg)
    forward = np.array((-np.sin(yaw), 0.0, -np.cos(yaw)), dtype=np.float64)
    right = np.array((np.cos(yaw), 0.0, -np.sin(yaw)), dtype=np.float64)
    up = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    view_to_camera = -forward
    return forward, right, up, view_to_camera


def project_positions(mesh: Mesh, camera: CameraSpec) -> tuple[np.ndarray, np.ndarray]:
    _, right, up, view_to_camera = camera_basis(camera)
    half_height = camera.ortho_height * 0.5
    half_width = half_height * camera.width / camera.height_px
    x_cam = mesh.positions @ right
    y_cam = mesh.positions @ up
    depth = mesh.positions @ view_to_camera
    px = (x_cam / half_width * 0.5 + 0.5) * (camera.width - 1)
    py = (0.5 - y_cam / half_height * 0.5) * (camera.height_px - 1)
    return np.stack((px, py), axis=-1), depth


def rasterize(mesh: Mesh, camera: CameraSpec) -> GBuffer:
    """Rasterize with a deterministic orthographic z-buffer.

    Work is vectorized inside each triangle; the Python loop only walks the
    small, fixed triangle list. The camera angles are offline presets, not an
    interactive renderer.
    """

    height, width = camera.height_px, camera.width
    screen, vertex_depth = project_positions(mesh, camera)
    depth_buffer = np.full((height, width), -np.inf, dtype=np.float64)
    coverage = np.zeros((height, width), dtype=bool)
    uv_buffer = np.zeros((height, width, 2), dtype=np.float64)
    normal_buffer = np.zeros((height, width, 3), dtype=np.float64)
    position_buffer = np.zeros((height, width, 3), dtype=np.float64)

    for triangle in mesh.triangles:
        points = screen[triangle]
        depths = vertex_depth[triangle]
        min_x = max(0, int(np.floor(points[:, 0].min())))
        max_x = min(width - 1, int(np.ceil(points[:, 0].max())))
        min_y = max(0, int(np.floor(points[:, 1].min())))
        max_y = min(height - 1, int(np.ceil(points[:, 1].max())))
        if min_x > max_x or min_y > max_y:
            continue

        x0, y0 = points[0]
        x1, y1 = points[1]
        x2, y2 = points[2]
        area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if abs(area) < 1e-18:
            continue

        xs = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)
        inv_area = 1.0 / area
        w0 = ((x1 - grid_x) * (y2 - grid_y) - (y1 - grid_y) * (x2 - grid_x)) * inv_area
        w1 = ((x2 - grid_x) * (y0 - grid_y) - (y2 - grid_y) * (x0 - grid_x)) * inv_area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-10) & (w1 >= -1e-10) & (w2 >= -1e-10)
        if not np.any(inside):
            continue
        depth = w0 * depths[0] + w1 * depths[1] + w2 * depths[2]
        window = depth_buffer[min_y : max_y + 1, min_x : max_x + 1]
        update = inside & (depth > window)
        if not np.any(update):
            continue
        window[update] = depth[update]
        depth_buffer[min_y : max_y + 1, min_x : max_x + 1] = window
        coverage[min_y : max_y + 1, min_x : max_x + 1][update] = True

        weights = np.stack((w0, w1, w2), axis=-1)
        uv_values = np.einsum("...k,kj->...j", weights, mesh.uvs[triangle])
        normal_values = np.einsum("...k,kj->...j", weights, mesh.normals[triangle])
        normal_values /= np.maximum(np.linalg.norm(normal_values, axis=-1, keepdims=True), 1e-15)
        position_values = np.einsum("...k,kj->...j", weights, mesh.positions[triangle])
        uv_window = uv_buffer[min_y : max_y + 1, min_x : max_x + 1]
        normal_window = normal_buffer[min_y : max_y + 1, min_x : max_x + 1]
        position_window = position_buffer[min_y : max_y + 1, min_x : max_x + 1]
        uv_window[update] = uv_values[update]
        normal_window[update] = normal_values[update]
        position_window[update] = position_values[update]
        uv_buffer[min_y : max_y + 1, min_x : max_x + 1] = uv_window
        normal_buffer[min_y : max_y + 1, min_x : max_x + 1] = normal_window
        position_buffer[min_y : max_y + 1, min_x : max_x + 1] = position_window

    _, _, _, view_to_camera = camera_basis(camera)
    return GBuffer(
        coverage=coverage,
        uv=uv_buffer,
        normal=normal_buffer,
        position=position_buffer,
        depth=depth_buffer,
        camera_view_direction=tuple(float(v) for v in view_to_camera),
    )
