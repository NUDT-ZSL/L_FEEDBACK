"""Shared vertex lighting math.

The Blinn-Phong style lighting here is the *single source of truth* for the
software backend.  The GLSL shader in :mod:`rfr.backends.gl_backend` mirrors
these exact formulas, built-ins and clamping so both paths produce the same
vertex colors.  Computations intentionally run in ``float32`` to match the
GPU's default shader precision.

Per vertex we compute:
  ambient  = ambient_color * base_color
  for each light: diffuse and Blinn-Phong specular, with point-light distance
  attenuation; metallic modulates the specular tint and roughness the exponent.
The resulting linear RGBA color is interpolated across each triangle.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .scene import Light


def compute_vertex_colors(
    world_positions: np.ndarray,   # (V, 3) float32
    world_normals: np.ndarray,     # (V, 3) float32
    base_color: np.ndarray,        # (4,) float32 linear RGBA
    roughness: float,
    metallic: float,
    lights: List[Light],
    ambient: np.ndarray,           # (3,) float
    camera_position: np.ndarray,   # (3,) float
) -> np.ndarray:
    """Return per-vertex linear RGBA colors, shape (V, 4), float32."""
    pos = np.asarray(world_positions, dtype=np.float32)
    nrm = np.asarray(world_normals, dtype=np.float32)
    norms = np.linalg.norm(nrm, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, np.float32(1.0), norms.astype(np.float32))
    nrm = (nrm / norms).astype(np.float32)

    base = np.asarray(base_color, dtype=np.float32)
    base_rgb = base[:3].astype(np.float32)
    eye = np.asarray(camera_position, dtype=np.float32)
    ambient_f = np.asarray(ambient, dtype=np.float32)

    shininess = np.float32(2.0 / max(float(roughness), 1e-3) ** 2)
    metal = np.float32(np.clip(metallic, 0.0, 1.0))
    spec_tint = (base_rgb * metal + np.float32(1.0 - metal)).astype(np.float32)

    color = np.zeros((pos.shape[0], 3), dtype=np.float32)
    color += (ambient_f * base_rgb).astype(np.float32)

    view_dir = eye[None, :] - pos
    view_len = np.linalg.norm(view_dir, axis=1, keepdims=True)
    view_len = np.where(view_len == 0.0, np.float32(1.0), view_len.astype(np.float32))
    view_dir = (view_dir / view_len).astype(np.float32)

    for light in lights:
        if light.kind == "directional":
            light_dir = (-np.asarray(light.direction, dtype=np.float32))
            light_dir = np.broadcast_to(light_dir, pos.shape).copy()
            attenuation = np.ones((pos.shape[0], 1), dtype=np.float32)
        else:
            light_pos = np.asarray(light.position, dtype=np.float32)
            delta = light_pos[None, :] - pos
            dist = np.linalg.norm(delta, axis=1, keepdims=True)
            safe_dist = np.where(dist == 0.0, np.float32(1.0), dist.astype(np.float32))
            light_dir = (delta / safe_dist).astype(np.float32)
            rng = np.float32(max(float(light.light_range), 1e-4))
            att = np.float32(1.0) / (
                np.float32(1.0)
                + np.float32(0.09) * dist.astype(np.float32)
                + np.float32(0.032) * (dist * dist).astype(np.float32)
            )
            att = np.where(dist > rng, np.float32(0.0), att).astype(np.float32)
            attenuation = att

        n_dot_l = np.sum(nrm * light_dir, axis=1, keepdims=True).astype(np.float32)
        diff_factor = np.clip(n_dot_l, 0.0, 1.0)

        half_dir = light_dir + view_dir
        half_len = np.linalg.norm(half_dir, axis=1, keepdims=True)
        half_len = np.where(half_len == 0.0, np.float32(1.0), half_len.astype(np.float32))
        half_dir = (half_dir / half_len).astype(np.float32)
        n_dot_h = np.sum(nrm * half_dir, axis=1, keepdims=True).astype(np.float32)
        spec_factor = np.clip(n_dot_h, 0.0, 1.0) ** shininess
        spec_factor = np.where(n_dot_l > 0.0, spec_factor, np.float32(0.0))

        light_rgb = (np.asarray(light.color, dtype=np.float32)
                     * np.float32(light.intensity)).astype(np.float32)
        diffuse = (base_rgb * light_rgb * diff_factor * attenuation).astype(np.float32)
        specular = (spec_tint * light_rgb * spec_factor * attenuation).astype(np.float32)
        color += diffuse + specular

    color = np.clip(color, 0.0, 1.0).astype(np.float32)
    alpha = np.full((pos.shape[0], 1), float(base[3]), dtype=np.float32)
    return np.concatenate([color, alpha], axis=1).astype(np.float32)
