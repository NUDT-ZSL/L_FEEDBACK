"""Fragment shading: tangent-space normal mapping + Cook-Torrance GGX."""

from __future__ import annotations

import numpy as np


def _normalize_rows(v, eps=1e-12):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, eps)


class Shader:
    """Callable per-triangle fragment shader used by the rasterizer."""

    def __init__(self, materials, normal_maps, eye, dir_lights, point_lights,
                 dir_shadows, cube_shadow, ambient, tangents, bitangents, tri_mat):
        self.materials = materials
        self.normal_maps = normal_maps  # parallel to materials; entry or None
        self.eye = np.asarray(eye, dtype=np.float64)
        self.dir_lights = dir_lights
        self.point_lights = point_lights
        self.dir_shadows = dir_shadows
        self.cube_shadow = cube_shadow
        self.ambient = np.asarray(ambient, dtype=np.float64)
        self.tangents = tangents      # (T, 3) per assembled triangle
        self.bitangents = bitangents  # (T, 3)
        self.tri_mat = tri_mat        # (T,) material index per triangle

    def _world_normal(self, data, t):
        n = _normalize_rows(data["nrm"])
        mat_idx = self.tri_mat[t]
        nmap = self.normal_maps[mat_idx]
        if nmap is None:
            return n
        ts = nmap.sample(data["uv"])  # tangent space, +Y up
        tangent = self.tangents[t][None, :]
        bitangent = self.bitangents[t][None, :]
        # Orthonormalize the tangent frame against the interpolated normal.
        t1 = tangent - n * np.sum(n * tangent, axis=1, keepdims=True)
        t1 = _normalize_rows(t1)
        b1 = bitangent - n * np.sum(n * bitangent, axis=1, keepdims=True)
        b1 = b1 - t1 * np.sum(t1 * b1, axis=1, keepdims=True)
        b1 = _normalize_rows(b1)
        world = t1 * ts[:, 0:1] + b1 * ts[:, 1:2] + n * ts[:, 2:3]
        return _normalize_rows(world)

    @staticmethod
    def _ggx(n, v, l, albedo, roughness):
        h = _normalize_rows(v + l)
        no_l = np.clip(np.sum(n * l, axis=1), 0.0, 1.0)
        no_v = np.clip(np.sum(n * v, axis=1), 1e-5, 1.0)
        no_h = np.clip(np.sum(n * h, axis=1), 0.0, 1.0)
        vo_h = np.clip(np.sum(v * h, axis=1), 0.0, 1.0)
        a = roughness * roughness
        a2 = a * a
        # GGX / Trowbridge-Reitz normal distribution.
        denom = no_h * no_h * (a2 - 1.0) + 1.0
        d = a2 / (np.pi * denom * denom)
        # Schlick Fresnel, F0 = 0.04 (dielectric).
        f = 0.04 + 0.96 * (1.0 - vo_h) ** 5
        # Smith visibility (Schlick-GGX masking).
        k = (roughness + 1.0) ** 2 / 8.0
        g1_l = no_l / (no_l * (1.0 - k) + k)
        g1_v = no_v / (no_v * (1.0 - k) + k)
        g = g1_l * g1_v
        spec = (d * f * g) / np.maximum(4.0 * no_v * no_l, 1e-8)
        diff = albedo / np.pi * (1.0 - f)[:, None]
        return (diff + spec[:, None]) * no_l[:, None]

    def __call__(self, data, t):
        pos = data["world"]
        n = self._world_normal(data, t)
        v = _normalize_rows(self.eye[None, :] - pos)
        mat = self.materials[self.tri_mat[t]]
        albedo = mat.albedo[None, :]
        roughness = mat.roughness
        color = albedo * self.ambient[None, :]
        for light, shadow_map in zip(self.dir_lights, self.dir_shadows):
            l = np.broadcast_to(-light.direction[None, :], pos.shape)
            shadow = shadow_map.visibility(pos, n)
            brdf = self._ggx(n, v, l, albedo, roughness)
            color = color + brdf * light.radiance[None, :] * shadow[:, None]
        for light in self.point_lights:
            lvec = light.position[None, :] - pos
            dist = np.linalg.norm(lvec, axis=1, keepdims=True)
            dist = np.maximum(dist, light.min_distance)  # min-distance clamp
            l = lvec / dist
            atten = 1.0 / (dist * dist)
            shadow = self.cube_shadow.visibility(pos, n)
            brdf = self._ggx(n, v, l, albedo, roughness)
            color = color + brdf * light.radiance[None, :] * atten * shadow[:, None]
        return color
