from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import BakeError


@dataclass
class Mesh:
    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    triangles: np.ndarray
    albedo: np.ndarray
    roughness: np.ndarray
    metallic: np.ndarray
    name: str = "mesh"

    def validate(self, require_uvs: bool = True) -> None:
        positions = np.asarray(self.positions)
        normals = np.asarray(self.normals)
        triangles = np.asarray(self.triangles)
        albedo = np.asarray(self.albedo)
        roughness = np.asarray(self.roughness)
        metallic = np.asarray(self.metallic)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise BakeError(
                "E_MESH_INVALID",
                f"{self.name} positions must have shape (N, 3).",
                {"actual_shape": list(positions.shape)},
                ["Regenerate the mesh with one 3D position per vertex."],
            )
        vertex_count = positions.shape[0]
        if vertex_count < 3:
            raise BakeError(
                "E_MESH_INVALID",
                f"{self.name} contains fewer than three vertices.",
                {"vertex_count": vertex_count},
                ["Provide a non-empty low-poly mesh."],
            )
        if normals.shape != (vertex_count, 3):
            raise BakeError(
                "E_MESH_INVALID",
                f"{self.name} normals must have shape ({vertex_count}, 3).",
                {"actual_shape": list(normals.shape)},
                ["Provide one normal per vertex."],
            )
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise BakeError(
                "E_MESH_INVALID",
                f"{self.name} triangles must have shape (M, 3).",
                {"actual_shape": list(triangles.shape)},
                ["Triangulate the mesh before baking."],
            )
        if triangles.size and (
            triangles.min() < 0 or triangles.max() >= vertex_count
        ):
            raise BakeError(
                "E_MESH_INVALID",
                f"{self.name} contains an out-of-range triangle index.",
                {
                    "vertex_count": vertex_count,
                    "min_index": int(triangles.min()),
                    "max_index": int(triangles.max()),
                },
                ["Rebuild the index buffer using valid vertex indices."],
            )
        if not np.isfinite(positions).all() or not np.isfinite(normals).all():
            raise BakeError(
                "E_MESH_INVALID",
                f"{self.name} contains non-finite geometry data.",
                {},
                ["Remove NaN or infinity values from positions and normals."],
            )
        if albedo.shape != (vertex_count, 3) or not np.isfinite(albedo).all():
            raise BakeError(
                "E_MESH_INVALID",
                f"{self.name} albedo must be a finite ({vertex_count}, 3) array.",
                {"actual_shape": list(albedo.shape)},
                ["Provide one RGB albedo triplet per vertex."],
            )
        if roughness.shape != (vertex_count,) or not np.isfinite(roughness).all():
            raise BakeError(
                "E_MESH_INVALID",
                f"{self.name} roughness must be a finite ({vertex_count},) array.",
                {"actual_shape": list(roughness.shape)},
                ["Provide one roughness scalar per vertex."],
            )
        if metallic.shape != (vertex_count,) or not np.isfinite(metallic).all():
            raise BakeError(
                "E_MESH_INVALID",
                f"{self.name} metallic must be a finite ({vertex_count},) array.",
                {"actual_shape": list(metallic.shape)},
                ["Provide one metallic scalar per vertex."],
            )
        if require_uvs:
            uvs = np.asarray(self.uvs)
            if uvs.shape != (vertex_count, 2):
                raise BakeError(
                    "E_MESH_INVALID",
                    f"{self.name} uvs must have shape ({vertex_count}, 2).",
                    {"actual_shape": list(uvs.shape)},
                    ["Provide one UV coordinate per vertex."],
                )
            if not np.isfinite(uvs).all():
                raise BakeError(
                    "E_MESH_INVALID",
                    f"{self.name} contains non-finite UV coordinates.",
                    {},
                    ["Remove NaN or infinity values from the UV channel."],
                )
            eps = 1e-9
            if np.any(uvs < -eps) or np.any(uvs > 1.0 + eps):
                bad = np.argwhere((uvs < -eps) | (uvs > 1.0 + eps))
                raise BakeError(
                    "E_UV_OUT_OF_RANGE",
                    "Low-poly UV coordinates must stay inside [0, 1].",
                    {
                        "mesh": self.name,
                        "invalid_vertex_count": int(np.unique(bad[:, 0]).size),
                        "first_invalid_vertex": int(bad[0, 0]),
                        "first_invalid_uv": [float(uvs[bad[0, 0], 0]), float(uvs[bad[0, 0], 1])],
                        "allowed_range": [0.0, 1.0],
                    },
                    [
                        "Pack UV islands inside the zero-to-one texture domain.",
                        "Re-run the UV layout/repack step before baking.",
                    ],
                )


def normalized_rows(values: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    result = np.zeros_like(values)
    np.divide(values, lengths, out=result, where=lengths > 1e-15)
    if fallback is not None:
        invalid = lengths[..., 0] <= 1e-15
        if np.any(invalid):
            result[invalid] = fallback
    return result

