from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .environment import HdrEnvironment
from .geometry import HoodMesh, hood_mesh
from .grains import GrainCache, GrainTexture
from .params import OrbitCamera, PaintParams, camera_eye, render_hash
from .tone import aces_tonemap, linear_to_srgb


ProgressCallback = Callable[[int, str], None]


class RenderCancelled(Exception):
    pass


@dataclass(frozen=True)
class CameraFrame:
    eye: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    tan_half_fov: float


@dataclass
class RasterData:
    depth: np.ndarray
    uv: np.ndarray
    normal: np.ndarray
    tangent: np.ndarray
    covered_pixels: int
    rasterized_triangles: int


SUN_DIRECTION = np.array(
    (
        math.cos(0.48) * math.sin(0.62),
        math.sin(0.48),
        math.cos(0.48) * math.cos(0.62),
    ),
    dtype=np.float64,
)
SUN_DIRECTION /= np.linalg.norm(SUN_DIRECTION)


def camera_frame(camera: OrbitCamera, width: int, height: int) -> CameraFrame:
    eye = np.asarray(camera_eye(camera), dtype=np.float64)
    target = np.asarray(camera.normalized().target, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array((0.0, 1.0, 0.0)))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return CameraFrame(
        eye=eye,
        right=right,
        up=up,
        forward=forward,
        tan_half_fov=math.tan(math.radians(camera.normalized().fov_deg) * 0.5),
    )


def srgb_to_linear(rgb: Tuple[float, float, float]) -> np.ndarray:
    c = np.asarray(rgb, dtype=np.float64)
    return np.where(
        c <= 0.04045,
        c / 12.92,
        np.power((c + 0.055) / 1.055, 2.4),
    )


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def rotate_hue(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Rotate linear RGB around the neutral axis to create a second hue."""
    axis = np.array((1.0, 1.0, 1.0), dtype=np.float64)
    axis /= math.sqrt(3.0)
    angle = float(amount) * math.pi * 1.55
    c = math.cos(angle)
    s = math.sin(angle)
    cross_matrix = np.array(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        ),
        dtype=np.float64,
    )
    rot = c * np.eye(3) + s * cross_matrix + (1.0 - c) * np.outer(axis, axis)
    return np.clip(rgb @ rot.T, 0.0, 1.0)


def ggx_ndf(n_dot_h: np.ndarray, roughness: float) -> np.ndarray:
    alpha = max(1e-4, float(roughness) * float(roughness))
    a2 = alpha * alpha
    denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0
    return a2 / (math.pi * np.maximum(denom * denom, 1e-12))


def smith_ggx_g1(n_dot_x: np.ndarray, roughness: float) -> np.ndarray:
    a = max(1e-4, float(roughness))
    k = (a + 1.0) * (a + 1.0) / 8.0
    x = np.maximum(n_dot_x, 0.0)
    return x / np.maximum(x * (1.0 - k) + k, 1e-12)


def direct_clearcoat_ggx(
    n: np.ndarray,
    v: np.ndarray,
    sun_radiance: np.ndarray,
    roughness: float = 0.055,
) -> np.ndarray:
    n_dot_l = np.clip(np.sum(n * SUN_DIRECTION, axis=-1), 0.0, 1.0)
    n_dot_v = np.clip(np.sum(n * v, axis=-1), 1e-5, 1.0)
    half_vector = SUN_DIRECTION[None, :] + v
    half_vector = _safe_normalize(half_vector)
    n_dot_h = np.clip(np.sum(n * half_vector, axis=-1), 0.0, 1.0)
    v_dot_h = np.clip(np.sum(v * half_vector, axis=-1), 0.0, 1.0)

    fresnel = 0.04 + 0.96 * np.power(1.0 - v_dot_h, 5.0)
    distribution = ggx_ndf(n_dot_h, roughness)
    geometry = smith_ggx_g1(n_dot_v, roughness) * smith_ggx_g1(n_dot_l, roughness)
    brdf = distribution * geometry * fresnel / np.maximum(4.0 * n_dot_v, 1e-5)
    return brdf[..., None] * sun_radiance[None, :] * n_dot_l[..., None]


def _project_vertices(
    mesh: HoodMesh, frame: CameraFrame, width: int, height: int
):
    relative = mesh.vertices - frame.eye[None, :]
    x_cam = relative @ frame.right
    y_cam = relative @ frame.up
    z_cam = relative @ frame.forward
    aspect = float(width) / float(height)
    safe_z = np.maximum(z_cam, 1e-6)
    ndc_x = x_cam / (safe_z * frame.tan_half_fov * aspect)
    ndc_y = y_cam / (safe_z * frame.tan_half_fov)
    screen_x = (ndc_x * 0.5 + 0.5) * width
    screen_y = (0.5 - ndc_y * 0.5) * height
    return screen_x, screen_y, z_cam


def rasterize_hood(
    mesh: HoodMesh,
    frame: CameraFrame,
    width: int,
    height: int,
    cancel_event,
    progress: Optional[ProgressCallback],
) -> RasterData:
    sx, sy, z = _project_vertices(mesh, frame, width, height)
    depth_buffer = np.full(width * height, np.inf, dtype=np.float64)
    uv_buffer = np.zeros((width * height, 2), dtype=np.float32)
    normal_buffer = np.zeros((width * height, 3), dtype=np.float32)
    tangent_buffer = np.zeros((width * height, 3), dtype=np.float32)
    triangles = mesh.triangles
    tri_count = int(triangles.shape[0])
    next_progress = 0

    for tri_index in range(tri_count):
        if (tri_index & 127) == 0:
            if cancel_event.is_set():
                raise RenderCancelled()
            if progress is not None and tri_index >= next_progress:
                pct = 2 + int(58 * tri_index / max(1, tri_count))
                progress(pct, "光栅化曲面 {}".format(pct))
                next_progress += max(1, tri_count // 50)

        i0, i1, i2 = int(triangles[tri_index, 0]), int(triangles[tri_index, 1]), int(triangles[tri_index, 2])
        z0, z1, z2 = z[i0], z[i1], z[i2]
        if z0 <= 1e-5 or z1 <= 1e-5 or z2 <= 1e-5:
            continue
        x0, x1, x2 = sx[i0], sx[i1], sx[i2]
        y0, y1, y2 = sy[i0], sy[i1], sy[i2]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-10:
            continue
        min_x = max(0, int(math.floor(min(x0, x1, x2))))
        max_x = min(width - 1, int(math.ceil(max(x0, x1, x2)) - 1))
        min_y = max(0, int(math.floor(min(y0, y1, y2))))
        max_y = min(height - 1, int(math.ceil(max(y0, y1, y2)) - 1))
        if max_x < min_x or max_y < min_y:
            continue

        xs = np.arange(min_x, max_x + 1, dtype=np.float64)[None, :] + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float64)[:, None] + 0.5
        w0 = ((x1 - xs) * (y2 - ys) - (x2 - xs) * (y1 - ys)) / area
        w1 = ((x2 - xs) * (y0 - ys) - (x0 - xs) * (y2 - ys)) / area
        w2 = 1.0 - w0 - w1
        mask = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not np.any(mask):
            continue

        inv_z = w0 / z0 + w1 / z1 + w2 / z2
        candidate_depth = 1.0 / np.maximum(inv_z, 1e-12)
        local_y, local_x = np.nonzero(mask)
        pixel_index = (local_y + min_y) * width + (local_x + min_x)
        depths = candidate_depth[mask]
        visible = depths < depth_buffer[pixel_index]
        if not np.any(visible):
            continue

        pixel_index = pixel_index[visible]
        q0 = w0[mask][visible] / z0
        q1 = w1[mask][visible] / z1
        q2 = w2[mask][visible] / z2
        q_sum = np.maximum(q0 + q1 + q2, 1e-15)
        b0 = q0 / q_sum
        b1 = q1 / q_sum
        b2 = q2 / q_sum
        depth_buffer[pixel_index] = depths[visible]
        uv_buffer[pixel_index, 0] = (
            b0 * mesh.uvs[i0, 0] + b1 * mesh.uvs[i1, 0] + b2 * mesh.uvs[i2, 0]
        )
        uv_buffer[pixel_index, 1] = (
            b0 * mesh.uvs[i0, 1] + b1 * mesh.uvs[i1, 1] + b2 * mesh.uvs[i2, 1]
        )
        normal_buffer[pixel_index, :] = (
            b0[:, None] * mesh.normals[i0]
            + b1[:, None] * mesh.normals[i1]
            + b2[:, None] * mesh.normals[i2]
        )
        tangent_buffer[pixel_index, :] = (
            b0[:, None] * mesh.tangents[i0]
            + b1[:, None] * mesh.tangents[i1]
            + b2[:, None] * mesh.tangents[i2]
        )

    covered = int(np.count_nonzero(np.isfinite(depth_buffer)))
    return RasterData(
        depth=depth_buffer,
        uv=uv_buffer,
        normal=normal_buffer,
        tangent=tangent_buffer,
        covered_pixels=covered,
        rasterized_triangles=tri_count,
    )


def _background_radiance(
    env: HdrEnvironment,
    frame: CameraFrame,
    width: int,
    height: int,
    cancel_event,
) -> np.ndarray:
    image = np.empty((height, width, 3), dtype=np.float64)
    aspect = float(width) / float(height)
    x_ndc = ((np.arange(width, dtype=np.float64) + 0.5) / width) * 2.0 - 1.0
    for y0 in range(0, height, 96):
        if cancel_event.is_set():
            raise RenderCancelled()
        y1 = min(height, y0 + 96)
        y_ndc = 1.0 - ((np.arange(y0, y1, dtype=np.float64) + 0.5) / height) * 2.0
        direction = (
            frame.forward[None, None, :]
            + frame.right[None, None, :] * x_ndc[None, :, None] * frame.tan_half_fov * aspect
            + frame.up[None, None, :] * y_ndc[:, None, None] * frame.tan_half_fov
        )
        direction = _safe_normalize(direction)
        image[y0:y1] = env.sample(direction, 0) * 0.34
    return image


def _grain_perturbed_normal(
    grain: GrainTexture,
    uv: np.ndarray,
    normal: np.ndarray,
    tangent: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    tex_normal, active = grain.sample(uv)
    tx, ty, tz = tex_normal[..., 0], tex_normal[..., 1], tex_normal[..., 2]
    bitangent = np.cross(normal, tangent)
    perturbed = normal + tangent * tx[..., None] + bitangent * ty[..., None]
    blended = normal + (perturbed - normal) * active[..., None]
    return _safe_normalize(blended), active


def _shade_pixels(
    indices: np.ndarray,
    raster: RasterData,
    mesh: HoodMesh,
    frame: CameraFrame,
    env: HdrEnvironment,
    grain: Optional[GrainTexture],
    params: PaintParams,
    width: int,
    height: int,
) -> np.ndarray:
    pixel_x = indices % width
    pixel_y = indices // width
    x_ndc = ((pixel_x.astype(np.float64) + 0.5) / width) * 2.0 - 1.0
    y_ndc = 1.0 - ((pixel_y.astype(np.float64) + 0.5) / height) * 2.0
    aspect = float(width) / float(height)
    ray = (
        frame.forward[None, :]
        + frame.right[None, :] * x_ndc[:, None] * frame.tan_half_fov * aspect
        + frame.up[None, :] * y_ndc[:, None] * frame.tan_half_fov
    )
    ray = _safe_normalize(ray)
    view = -ray
    depth = raster.depth[indices][:, None]
    world_position = frame.eye[None, :] + ray * depth
    view = _safe_normalize(frame.eye[None, :] - world_position)

    normal = _safe_normalize(raster.normal[indices].astype(np.float64))
    tangent = raster.tangent[indices].astype(np.float64)
    tangent = tangent - normal * np.sum(tangent * normal, axis=-1, keepdims=True)
    tangent = _safe_normalize(tangent)
    n_dot_v_geo = np.clip(np.sum(normal * view, axis=-1), 1e-5, 1.0)

    base_color = srgb_to_linear(params.base_color)
    shaded_normal = normal
    active_grain = np.zeros(indices.shape[0], dtype=bool)
    if grain is not None and params.grain_density > 0.0:
        shaded_normal, active_grain = _grain_perturbed_normal(
            grain, raster.uv[indices].astype(np.float64), normal, tangent
        )

    color = np.broadcast_to(base_color, (indices.shape[0], 3)).copy()
    if params.paint_type == "pearl" and abs(params.pearl_shift) > 1e-5:
        flake_dot_view = np.abs(np.sum(shaded_normal * view, axis=-1))
        angular_mix = np.clip(0.28 + 0.72 * flake_dot_view, 0.0, 1.0)
        hue_mix = np.clip(active_grain.astype(np.float64) * angular_mix * abs(params.pearl_shift), 0.0, 1.0)
        second_hue = rotate_hue(base_color, params.pearl_shift)
        color = color * (1.0 - hue_mix[:, None]) + second_hue[None, :] * hue_mix[:, None]

    n_dot_v = np.clip(np.sum(shaded_normal * view, axis=-1), 1e-5, 1.0)
    reflection = 2.0 * n_dot_v[:, None] * shaded_normal - view
    reflection = _safe_normalize(reflection)
    metallic = float(params.metallic)
    f0 = 0.04 * (1.0 - metallic) + color * metallic
    fresnel = f0 + (1.0 - f0) * np.power(1.0 - n_dot_v, 5.0)[:, None]
    specular_env = env.sample_rough(reflection, params.roughness)
    base_specular = specular_env * fresnel

    diffuse_env = env.sample_rough(normal, 0.94)
    base_diffuse = (
        (1.0 - metallic) * color * diffuse_env * (1.0 - fresnel) * 0.82
    )
    base_layer = base_specular + base_diffuse

    coat_roughness = 0.055
    coat_fresnel = 0.04 + 0.96 * np.power(1.0 - n_dot_v_geo, 5.0)
    coat_reflection = 2.0 * n_dot_v_geo[:, None] * normal - view
    coat_reflection = _safe_normalize(coat_reflection)
    coat_ibl = env.sample(coat_reflection, 0) * coat_fresnel[:, None] * 0.72
    coat_direct = direct_clearcoat_ggx(
        normal,
        view,
        np.array((5.8, 4.7, 3.3), dtype=np.float64),
        coat_roughness,
    )
    transmission = math.exp(-0.135 * float(params.clearcoat_thickness))
    clearcoat_layer = coat_ibl + coat_direct
    return clearcoat_layer + (1.0 - coat_fresnel[:, None]) * transmission * base_layer


class PaintRenderer:
    def __init__(
        self,
        subdivisions_u: int = 128,
        subdivisions_v: int = 128,
        environment_width: int = 512,
        environment_height: int = 256,
    ) -> None:
        self.mesh = hood_mesh(subdivisions_u, subdivisions_v)
        self.environment = HdrEnvironment.build(environment_width, environment_height)
        self.grain_cache = GrainCache()
        self.closed = False
        self._last_render_stats: Dict[str, object] = {}

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("Renderer resources have already been released")

    def render(
        self,
        params: PaintParams,
        camera: OrbitCamera,
        width: int,
        height: int,
        cancel_event,
        progress: Optional[ProgressCallback] = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        self._ensure_open()
        width = int(width)
        height = int(height)
        if width < 64 or height < 64:
            raise ValueError("Render dimensions must be at least 64x64")
        params = params.normalized()
        camera = camera.normalized()
        started = time.perf_counter()
        if progress is not None:
            progress(0, "准备相机与预过滤环境")
        frame = camera_frame(camera, width, height)
        if cancel_event.is_set():
            raise RenderCancelled()

        background = _background_radiance(
            self.environment, frame, width, height, cancel_event
        )
        if progress is not None:
            progress(60, "光栅化引擎盖曲面")
        raster_started = time.perf_counter()
        raster = rasterize_hood(
            self.mesh, frame, width, height, cancel_event, progress
        )
        raster_ms = (time.perf_counter() - raster_started) * 1000.0

        grain = None
        if params.paint_type in ("metallic", "pearl") and params.grain_density > 0.0:
            grain = self.grain_cache.get(
                params.grain_density, params.grain_seed, params.paint_type
            )

        if progress is not None:
            progress(66, "计算分层 BRDF")
        image_linear = background
        indices = np.flatnonzero(np.isfinite(raster.depth))
        chunk_size = 196608
        chunk_total = int(math.ceil(indices.size / float(chunk_size)))
        for chunk_index, start in enumerate(range(0, indices.size, chunk_size)):
            if cancel_event.is_set():
                raise RenderCancelled()
            stop = min(indices.size, start + chunk_size)
            shaded = _shade_pixels(
                indices[start:stop],
                raster,
                self.mesh,
                frame,
                self.environment,
                grain,
                params,
                width,
                height,
            )
            flat_indices = indices[start:stop]
            image_linear[flat_indices // width, flat_indices % width, :] = shaded
            if progress is not None:
                pct = 66 + int(30 * (chunk_index + 1) / max(1, chunk_total))
                progress(pct, "计算分层 BRDF {}".format(pct))

        if cancel_event.is_set():
            raise RenderCancelled()
        if progress is not None:
            progress(97, "色调映射与伽马输出")
        mapped = linear_to_srgb(aces_tonemap(image_linear, params.exposure))
        rgb8 = np.clip(np.rint(mapped * 255.0), 0, 255).astype(np.uint8)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        stats: Dict[str, object] = {
            "width": width,
            "height": height,
            "render_ms": round(elapsed_ms, 3),
            "raster_ms": round(raster_ms, 3),
            "covered_pixels": raster.covered_pixels,
            "triangle_count": int(self.mesh.triangles.shape[0]),
            "subdivisions": [self.mesh.subdivisions_u, self.mesh.subdivisions_v],
            "env_prefilter_ms": round(self.environment.prefilter_ms, 3),
            "env_prefilter_generation_count": 1,
            "env_levels": self.environment.level_count,
            "env_dimensions": self.environment.level_dimensions,
            "env_digest": self.environment.digest[:16],
            "grain_generated": grain is not None,
            "grain_dimensions": [grain.width, grain.height] if grain is not None else None,
            "grain_digest": grain.digest[:16] if grain is not None else None,
            "grain_generation_ms": round(grain.generation_ms, 3) if grain is not None else 0.0,
            "grain_cache_generation_count": self.grain_cache.generation_count,
            "material_hash": params.material_hash(),
            "camera_hash": camera.camera_hash(),
            "render_hash": render_hash(params, camera, width, height),
        }
        self._last_render_stats = stats
        if progress is not None:
            progress(100, "渲染完成")
        return rgb8, stats

    @property
    def last_render_stats(self) -> Dict[str, object]:
        return dict(self._last_render_stats)

    def close(self) -> None:
        if self.closed:
            return
        self.grain_cache.clear()
        self.environment.clear()
        self.closed = True

