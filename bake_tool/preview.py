from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import Mesh, normalized_rows
from .tbn import TBNResult


@dataclass
class PreviewResult:
    image: np.ndarray
    rendered_pixels: int
    rasterized_triangles: int
    culled_triangles: int
    mean_luminance: float


def _edge_2d(a: np.ndarray, b: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    return (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])


def _sample_texture(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    ix = np.clip(np.floor(u * width).astype(np.int64), 0, width - 1)
    iy = np.clip(np.floor((1.0 - v) * height).astype(np.int64), 0, height - 1)
    return image[iy, ix]


def _decode_normal_map(
    normal_image: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    convention: str,
) -> np.ndarray:
    encoded = _sample_texture(normal_image, u, v).astype(np.float64) / 255.0
    normal = encoded * 2.0 - 1.0
    if convention == "directx":
        normal[:, 1] *= -1.0
    lengths = np.linalg.norm(normal, axis=1, keepdims=True)
    np.divide(normal, lengths, out=normal, where=lengths > 1.0e-12)
    return normal


def render_preview(
    mesh: Mesh,
    normal_map: np.ndarray,
    albedo_map: np.ndarray,
    roughness_map: np.ndarray,
    metallic_map: np.ndarray,
    coverage_mask: np.ndarray,
    tbn: TBNResult,
    resolution: int,
    normal_convention: str,
) -> PreviewResult:
    width = height = int(resolution)
    positions = np.asarray(mesh.positions)
    triangles = np.asarray(mesh.triangles)
    normals = normalized_rows(mesh.normals)

    center = (positions.min(axis=0) + positions.max(axis=0)) * 0.5
    forward = normalized_rows(
        np.asarray([[0.42, -0.58, -0.70]], dtype=np.float64)
    )[0]
    world_up = np.asarray([0.0, 1.0, 0.0])
    right = normalized_rows(np.cross(forward, world_up)[None, :])[0]
    if np.linalg.norm(right) < 1.0e-9:
        right = np.asarray([1.0, 0.0, 0.0])
    up = normalized_rows(np.cross(right, forward)[None, :])[0]

    centered = positions - center
    ortho_x = centered @ right
    ortho_y = centered @ up
    depth_coordinate = centered @ forward
    extent = max(
        float(np.max(np.abs(ortho_x))),
        float(np.max(np.abs(ortho_y))),
        1.0e-6,
    ) * 1.16
    screen_x = (ortho_x / extent * 0.5 + 0.5) * width
    screen_y = (0.5 - ortho_y / extent * 0.5) * height

    triangle_positions = positions[triangles]
    face_normals = normalized_rows(
        np.cross(
            triangle_positions[:, 1] - triangle_positions[:, 0],
            triangle_positions[:, 2] - triangle_positions[:, 0],
        )
    )
    visibility = face_normals @ (-forward) > 0.0

    z_buffer = np.full((height, width), np.inf, dtype=np.float64)
    owner = np.full((height, width), -1, dtype=np.int32)
    barycentric = np.zeros((height, width, 3), dtype=np.float64)
    rasterized = 0
    culled = 0

    for triangle_id in np.nonzero(visibility)[0]:
        a, b, c = triangles[triangle_id]
        sx = screen_x[[a, b, c]]
        sy = screen_y[[a, b, c]]
        p0 = np.asarray([sx[0], sy[0]])
        p1 = np.asarray([sx[1], sy[1]])
        p2 = np.asarray([sx[2], sy[2]])
        area = _edge_2d(p0, p1, p2[0], p2[1])
        if abs(float(area)) < 1.0e-12:
            continue

        x_start = max(0, int(np.floor(sx.min())))
        x_stop = min(width - 1, int(np.ceil(sx.max())))
        y_start = max(0, int(np.floor(sy.min())))
        y_stop = min(height - 1, int(np.ceil(sy.max())))
        if x_start > x_stop or y_start > y_stop:
            continue

        ys, xs = np.mgrid[y_start : y_stop + 1, x_start : x_stop + 1]
        px = xs + 0.5
        py = ys + 0.5
        w0 = _edge_2d(p1, p2, px, py) / area
        w1 = _edge_2d(p2, p0, px, py) / area
        w2 = _edge_2d(p0, p1, px, py) / area
        inside = (w0 >= -1.0e-9) & (w1 >= -1.0e-9) & (w2 >= -1.0e-9)
        if not np.any(inside):
            continue

        depth = w0 * depth_coordinate[a] + w1 * depth_coordinate[b] + w2 * depth_coordinate[c]
        existing = z_buffer[y_start : y_stop + 1, x_start : x_stop + 1]
        update = inside & (
            (depth < existing - 1.0e-12)
            | (
                (np.abs(depth - existing) <= 1.0e-12)
                & (
                    triangle_id
                    <= owner[y_start : y_stop + 1, x_start : x_stop + 1]
                )
            )
        )
        if not np.any(update):
            continue
        py_i = ys[update]
        px_i = xs[update]
        z_buffer[py_i, px_i] = depth[update]
        owner[py_i, px_i] = triangle_id
        barycentric[py_i, px_i, 0] = w0[update]
        barycentric[py_i, px_i, 1] = w1[update]
        barycentric[py_i, px_i, 2] = w2[update]
        rasterized += 1
    culled = int(np.count_nonzero(~visibility))

    rendered = owner >= 0
    background = np.zeros((height, width, 3), dtype=np.float64)
    gy = np.linspace(0.035, 0.075, height, dtype=np.float64)[:, None]
    gx = np.linspace(0.045, 0.025, width, dtype=np.float64)[None, :]
    background[..., 0] = gy + gx * 0.35
    background[..., 1] = gy + gx * 0.45
    background[..., 2] = gy + gx * 0.65
    image = background

    if not np.any(rendered):
        return PreviewResult(
            image=np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8),
            rendered_pixels=0,
            rasterized_triangles=rasterized,
            culled_triangles=culled,
            mean_luminance=0.0,
        )

    py, px = np.nonzero(rendered)
    tri_ids = owner[py, px]
    weights = barycentric[py, px]
    tri_vertices = triangles[tri_ids]
    uv = np.einsum("ni,nij->nj", weights, mesh.uvs[tri_vertices])
    interpolated_normal = normalized_rows(
        np.einsum("ni,nij->nj", weights, normals[tri_vertices])
    )
    interpolated_tangent = np.einsum("ni,nij->nj", weights, tbn.tangents[tri_vertices])
    interpolated_tangent -= interpolated_normal * np.einsum(
        "ij,ij->i", interpolated_tangent, interpolated_normal
    )[:, None]
    interpolated_tangent = normalized_rows(interpolated_tangent)
    interpolated_bitangent = normalized_rows(
        np.cross(interpolated_normal, interpolated_tangent)
    )

    tangent_normal = _decode_normal_map(normal_map, uv[:, 0], uv[:, 1], normal_convention)
    world_normal = (
        interpolated_tangent * tangent_normal[:, :1]
        + interpolated_bitangent * tangent_normal[:, 1:2]
        + interpolated_normal * tangent_normal[:, 2:3]
    )
    world_normal = normalized_rows(world_normal)
    # Keep the baked normal on the geometric hemisphere in malformed fallback cases.
    hemisphere = np.einsum("ij,ij->i", world_normal, interpolated_normal) < 0.0
    world_normal[hemisphere] *= -1.0

    textured_pixels = coverage_mask[
        np.clip((1.0 - uv[:, 1]) * height, 0, height - 1).astype(np.int64),
        np.clip(uv[:, 0] * width, 0, width - 1).astype(np.int64),
    ]
    albedo = _sample_texture(albedo_map, uv[:, 0], uv[:, 1]).astype(np.float64) / 255.0
    roughness = (
        _sample_texture(roughness_map, uv[:, 0], uv[:, 1]).astype(np.float64) / 255.0
    )
    metallic = (
        _sample_texture(metallic_map, uv[:, 0], uv[:, 1]).astype(np.float64) / 255.0
    )
    low_albedo = np.einsum("ni,nij->nj", weights, mesh.albedo[tri_vertices])
    low_roughness = np.einsum("ni,ni->n", weights, mesh.roughness[tri_vertices])
    low_metallic = np.einsum("ni,ni->n", weights, mesh.metallic[tri_vertices])
    albedo[~textured_pixels] = low_albedo[~textured_pixels]
    roughness[~textured_pixels] = low_roughness[~textured_pixels]
    metallic[~textured_pixels] = low_metallic[~textured_pixels]
    roughness = np.clip(roughness, 0.04, 1.0)
    metallic = np.clip(metallic, 0.0, 1.0)

    view_direction = -forward
    key_direction = normalized_rows(np.asarray([[-0.45, 0.60, 0.66]]))[0]
    fill_direction = normalized_rows(np.asarray([[0.68, -0.28, 0.34]]))[0]
    sky = np.asarray([0.28, 0.40, 0.62])
    ground = np.asarray([0.12, 0.09, 0.065])
    hemisphere_weight = world_normal[:, 2:3] * 0.5 + 0.5
    ambient = sky * hemisphere_weight + ground * (1.0 - hemisphere_weight)

    metallic_column = metallic[:, None]
    diffuse_color = albedo * (1.0 - metallic_column)
    f0 = 0.04 * (1.0 - metallic_column) + albedo * metallic_column
    roughness_column = roughness[:, None]
    alpha = np.maximum(roughness * roughness, 1.0e-3)
    radiance = ambient * albedo * 0.35

    for light_direction, light_color, intensity in (
        (key_direction, np.asarray([1.0, 0.94, 0.84]), 2.1),
        (fill_direction, np.asarray([0.42, 0.58, 0.82]), 0.85),
    ):
        ndotl = np.clip(world_normal @ light_direction, 0.0, 1.0)[:, None]
        half_vector = normalized_rows((light_direction[None, :] + view_direction[None, :]))
        ndoth = np.clip(world_normal @ half_vector[0], 0.0, 1.0)[:, None]
        vdoth = np.clip(float(view_direction @ light_direction), 0.0, 1.0)
        ndotv = np.clip(world_normal @ view_direction, 1.0e-4, 1.0)[:, None]
        distribution = alpha[:, None] ** 2 / (
            np.pi
            * (
                ndoth * ndoth * (alpha[:, None] ** 2 - 1.0) + 1.0
            )
            ** 2
            + 1.0e-8
        )
        geometry = (
            0.5
            / (
                ndotl * (1.0 - alpha[:, None]) + alpha[:, None]
                + ndotv * (1.0 - alpha[:, None]) + alpha[:, None]
                + 1.0e-8
            )
        )
        fresnel = f0 + (1.0 - f0) * (1.0 - vdoth) ** 5
        specular = distribution * geometry * fresnel / (4.0 * ndotl * ndotv + 1.0e-7)
        radiance += (
            diffuse_color * ndotl + specular * ndotl * (0.08 + 0.92 * ndotl)
        ) * light_color * intensity

    reflection = ambient * f0 * (1.0 - roughness_column)
    radiance += reflection * 0.6
    mapped = radiance / (radiance + 1.0)
    mapped = np.power(np.clip(mapped, 0.0, 1.0), 1.0 / 2.2)
    image[py, px] = mapped
    image = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
    luminance = image.astype(np.float64) @ np.asarray([0.2126, 0.7152, 0.0722])
    return PreviewResult(
        image=image,
        rendered_pixels=int(np.count_nonzero(rendered)),
        rasterized_triangles=rasterized,
        culled_triangles=culled,
        mean_luminance=float(luminance[rendered].mean()),
    )
