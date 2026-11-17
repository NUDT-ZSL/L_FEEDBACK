"""Wavefront OBJ loader.

Supports ``v`` / ``vt`` / ``vn`` / ``f`` statements. Face indices are
1-based; negative indices are resolved relative to the end of the
corresponding list (``-1`` is the most recently defined element).
Polygon faces are triangulated as a fan.
"""

from __future__ import annotations

import os

import numpy as np

from .errors import AssetError


def _resolve_index(token: str, count: int, path: str, lineno: int) -> int:
    try:
        value = int(token)
    except ValueError:
        raise AssetError(f"{path}:{lineno}: invalid face index {token!r}")
    if value == 0:
        raise AssetError(f"{path}:{lineno}: OBJ indices are 1-based, got 0")
    idx = value - 1 if value > 0 else count + value
    if idx < 0 or idx >= count:
        raise AssetError(
            f"{path}:{lineno}: index {value} out of range (list has {count} entries)"
        )
    return idx


def load_obj(path: str) -> dict:
    """Load an OBJ file into per-corner triangle arrays.

    Returns a dict with keys ``pos`` (F,3,3), ``nrm`` (F,3,3) or None,
    ``uv`` (F,3,2) or None.
    """
    if not os.path.isfile(path):
        raise AssetError(f"OBJ file not found: {path}")

    positions = []
    uvs = []
    normals = []
    tri_v = []
    tri_t = []
    tri_n = []
    has_uv = None
    has_nrm = None

    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "v":
                if len(parts) < 4:
                    raise AssetError(f"{path}:{lineno}: 'v' needs 3 coordinates")
                try:
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    raise AssetError(f"{path}:{lineno}: invalid vertex coordinates")
            elif tag == "vt":
                if len(parts) < 3:
                    raise AssetError(f"{path}:{lineno}: 'vt' needs 2 coordinates")
                try:
                    uvs.append([float(parts[1]), float(parts[2])])
                except ValueError:
                    raise AssetError(f"{path}:{lineno}: invalid texcoord")
            elif tag == "vn":
                if len(parts) < 4:
                    raise AssetError(f"{path}:{lineno}: 'vn' needs 3 components")
                try:
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    raise AssetError(f"{path}:{lineno}: invalid normal")
            elif tag == "f":
                if len(parts) < 4:
                    raise AssetError(f"{path}:{lineno}: face needs at least 3 vertices")
                corner_v, corner_t, corner_n = [], [], []
                for spec in parts[1:]:
                    fields = spec.split("/")
                    if len(fields) > 3:
                        raise AssetError(f"{path}:{lineno}: bad face spec {spec!r}")
                    vi = _resolve_index(fields[0], len(positions), path, lineno)
                    ti = ni = -1
                    if len(fields) >= 2 and fields[1] != "":
                        ti = _resolve_index(fields[1], len(uvs), path, lineno)
                    if len(fields) == 3 and fields[2] != "":
                        ni = _resolve_index(fields[2], len(normals), path, lineno)
                    corner_v.append(vi)
                    corner_t.append(ti)
                    corner_n.append(ni)
                face_has_uv = corner_t[0] >= 0
                face_has_nrm = corner_n[0] >= 0
                if any((t >= 0) != face_has_uv for t in corner_t):
                    raise AssetError(f"{path}:{lineno}: inconsistent vt usage in face")
                if any((n >= 0) != face_has_nrm for n in corner_n):
                    raise AssetError(f"{path}:{lineno}: inconsistent vn usage in face")
                if has_uv is None:
                    has_uv = face_has_uv
                elif has_uv != face_has_uv:
                    raise AssetError(f"{path}:{lineno}: mixed faces with/without vt")
                if has_nrm is None:
                    has_nrm = face_has_nrm
                elif has_nrm != face_has_nrm:
                    raise AssetError(f"{path}:{lineno}: mixed faces with/without vn")
                # Fan triangulation of the (convex) polygon.
                for k in range(1, len(corner_v) - 1):
                    tri_v.append([corner_v[0], corner_v[k], corner_v[k + 1]])
                    tri_t.append([corner_t[0], corner_t[k], corner_t[k + 1]])
                    tri_n.append([corner_n[0], corner_n[k], corner_n[k + 1]])
            # Other statements (o/g/s/usemtl/mtllib/...) are ignored.

    if not tri_v:
        raise AssetError(f"{path}: no faces found")

    pos = np.asarray(positions, dtype=np.float64)
    out = {"pos": pos[np.asarray(tri_v, dtype=np.int64)]}
    if has_uv:
        out["uv"] = np.asarray(uvs, dtype=np.float64)[np.asarray(tri_t, dtype=np.int64)]
    else:
        out["uv"] = None
    if has_nrm:
        out["nrm"] = np.asarray(normals, dtype=np.float64)[np.asarray(tri_n, dtype=np.int64)]
    else:
        out["nrm"] = None
    return out
