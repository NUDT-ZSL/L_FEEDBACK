"""Bind-pose tangent basis construction with deterministic spherical fallback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math3d import orthonormal_tangent, safe_normalize, sphere_fallback_tangent


@dataclass(frozen=True)
class BindBasis:
    normals: np.ndarray
    tangents: np.ndarray
    bitangents: np.ndarray
    fallback_vertices: tuple[int, ...]
    fallback_reasons: dict[int, str]


def build_bind_basis(normals: np.ndarray, tangents: np.ndarray) -> BindBasis:
    normals = np.asarray(normals, dtype=np.float64)
    tangents = np.asarray(tangents, dtype=np.float64)
    count = normals.shape[0]
    out_normals = np.empty((count, 3), dtype=np.float64)
    out_tangents = np.empty((count, 3), dtype=np.float64)
    out_bitangents = np.empty((count, 3), dtype=np.float64)
    fallback: list[int] = []
    reasons: dict[int, str] = {}

    for vertex in range(count):
        n = normals[vertex]
        t_raw = tangents[vertex, :3]
        handedness = 1.0 if tangents[vertex, 3] >= 0.0 else -1.0
        reason = ""
        if not np.all(np.isfinite(n)) or float(np.linalg.norm(n)) < 1e-12:
            reason = "degenerate_normal"
        elif not np.all(np.isfinite(t_raw)) or float(np.linalg.norm(t_raw)) < 1e-6:
            reason = "degenerate_tangent"
        elif float(np.linalg.norm(t_raw - np.dot(t_raw, n) * n / max(np.dot(n, n), 1e-30))) < 1e-8:
            reason = "tangent_parallel_to_normal"
        elif abs(float(np.dot(n, t_raw))) / (
            max(float(np.linalg.norm(n)), 1e-30) * max(float(np.linalg.norm(t_raw)), 1e-30)
        ) > 1e-2:
            reason = "non_orthogonal_basis_sphere_fallback"

        if reason:
            unit_n = safe_normalize(n)
            if unit_n is None:
                unit_n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                reason = "degenerate_normal_sphere_fallback"
            t_hint = sphere_fallback_tangent(unit_n)
        else:
            unit_n = n / np.linalg.norm(n)
            t_hint = t_raw

        built = orthonormal_tangent(unit_n, t_hint)
        if built is None:
            reason = reason or "non_orthogonal_basis_sphere_fallback"
            unit_n = safe_normalize(unit_n) or np.array([0.0, 0.0, 1.0], dtype=np.float64)
            t_hint = sphere_fallback_tangent(unit_n)
            built = orthonormal_tangent(unit_n, t_hint)
        if built is None:
            raise RuntimeError(f"could not construct tangent basis for vertex {vertex}")
        t, b = built
        b = b * handedness
        basis_dot = abs(float(np.dot(unit_n, t)))
        basis_det = abs(float(np.linalg.det(np.stack([unit_n, t, b], axis=1))))
        if basis_dot > 1e-8 or abs(basis_det - 1.0) > 1e-8:
            reason = reason or "non_orthogonal_basis_sphere_fallback"
            unit_n = safe_normalize(unit_n) or np.array([0.0, 0.0, 1.0], dtype=np.float64)
            t_hint = sphere_fallback_tangent(unit_n)
            built = orthonormal_tangent(unit_n, t_hint)
            if built is None:
                raise RuntimeError(f"could not construct fallback tangent basis for vertex {vertex}")
            t, b = built
            b = b * handedness

        if reason:
            fallback.append(vertex)
            reasons[vertex] = reason
        out_normals[vertex] = unit_n
        out_tangents[vertex] = t
        out_bitangents[vertex] = b

    return BindBasis(
        normals=out_normals,
        tangents=out_tangents,
        bitangents=out_bitangents,
        fallback_vertices=tuple(fallback),
        fallback_reasons=reasons,
    )

