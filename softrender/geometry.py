"""Camera construction, frustum culling, clipping and screen assembly."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import math3d

SUBPIXEL_BITS = 8
SUBPIXEL = 1 << SUBPIXEL_BITS  # 1/256 px fixed-point precision (>= 1/16 required)


@dataclass
class CameraRig:
    view: np.ndarray          # 4x4 world -> camera
    proj: np.ndarray          # 4x4 camera -> clip
    planes: list              # camera-space clip planes (a, b, c, d)
    eye: np.ndarray           # world-space camera position


def main_camera(bmin: np.ndarray, bmax: np.ndarray, width: int, height: int) -> CameraRig:
    """Frame the merged world-space bounding box of all models."""
    center = (bmin + bmax) * 0.5
    radius = float(np.linalg.norm(bmax - bmin)) * 0.5
    radius = max(radius, 1e-3)
    fovy = math.radians(45.0)
    aspect = width / height
    view_dir = math3d.normalize(np.array([0.62, 0.42, 1.0]))
    dist = radius / math.tan(fovy * 0.5) * 1.05
    eye = center + view_dir * dist
    near = max(1e-3, dist - 2.5 * radius)
    far = dist + 2.5 * radius
    view = math3d.look_at(eye, center, np.array([0.0, 1.0, 0.0]))
    proj = math3d.perspective(fovy, aspect, near, far)
    planes = math3d.perspective_frustum_planes(fovy, aspect, near, far)
    return CameraRig(view=view, proj=proj, planes=planes, eye=eye)


def directional_light_camera(direction: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> CameraRig:
    """Orthographic camera looking along ``direction`` covering the bbox."""
    center = (bmin + bmax) * 0.5
    radius = float(np.linalg.norm(bmax - bmin)) * 0.5
    radius = max(radius, 1e-3)
    direction = math3d.normalize(direction)
    eye = center - direction * (2.0 * radius + 1.0)
    up = np.array([0.0, 1.0, 0.0])
    if abs(float(direction @ up)) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    view = math3d.look_at(eye, center, up)
    corners = np.array(
        [[x, y, z] for x in (bmin[0], bmax[0])
         for y in (bmin[1], bmax[1]) for z in (bmin[2], bmax[2])]
    )
    cam = math3d.apply_mat4(view, corners)[:, :3]
    pad = 0.02 * radius
    l, r = float(cam[:, 0].min()) - pad, float(cam[:, 0].max()) + pad
    b, t = float(cam[:, 1].min()) - pad, float(cam[:, 1].max()) + pad
    near = max(1e-3, -float(cam[:, 2].max()) - pad)
    far = -float(cam[:, 2].min()) + pad
    proj = math3d.orthographic(l, r, b, t, near, far)
    planes = math3d.ortho_frustum_planes(l, r, b, t, near, far)
    return CameraRig(view=view, proj=proj, planes=planes, eye=eye)


def point_light_face_camera(position: np.ndarray, face: int, near: float, far: float) -> CameraRig:
    """Perspective 90-degree camera for one cube face (see shadow.CUBE_FACES)."""
    from .shadow import CUBE_FACES

    major, _, down = CUBE_FACES[face]
    up = -down
    view = math3d.look_at(position, position + major, up)
    proj = math3d.perspective(math.radians(90.0), 1.0, near, far)
    planes = math3d.perspective_frustum_planes(math.radians(90.0), 1.0, near, far)
    return CameraRig(view=view, proj=proj, planes=planes, eye=np.asarray(position, dtype=np.float64))


def _clip_polygon_plane(verts: list, plane) -> list:
    """Sutherland-Hodgman against one plane. Each vertex is a dict of arrays."""
    a, b, c, d = plane
    out = []
    n = len(verts)
    for i in range(n):
        cur = verts[i]
        prv = verts[(i + n - 1) % n]
        pc = prv["cam"]
        cc = cur["cam"]
        d_prv = a * pc[0] + b * pc[1] + c * pc[2] + d
        d_cur = a * cc[0] + b * cc[1] + c * cc[2] + d
        in_prv = d_prv >= 0.0
        in_cur = d_cur >= 0.0
        if in_cur != in_prv:
            t = d_prv / (d_prv - d_cur)
            out.append({k: prv[k] + t * (cur[k] - prv[k]) for k in cur})
        if in_cur:
            out.append(cur)
    return out


def cull_triangle(cam_pts: np.ndarray, planes) -> bool:
    """True if the triangle is completely outside any frustum plane."""
    for a, b, c, d in planes:
        dist = cam_pts[:, 0] * a + cam_pts[:, 1] * b + cam_pts[:, 2] * c + d
        if np.all(dist < 0.0):
            return True
    return False


def assemble_triangles(world, nrm, uv, mat, rig: CameraRig, width: int, height: int) -> dict:
    """Transform, cull, clip, project and quantize triangles.

    Returns screen-space triangle arrays with fixed-point (1/256 px) coords.
    """
    count = world.shape[0]
    cam = world.reshape(-1, 3) @ rig.view[:3, :3].T + rig.view[:3, 3]
    cam = cam.reshape(count, 3, 3)
    # Frustum culling: drop triangles fully outside any plane.
    keep = np.ones(count, dtype=bool)
    for plane in rig.planes:
        a, b, c, d = plane
        dist = cam[:, :, 0] * a + cam[:, :, 1] * b + cam[:, :, 2] * c + d
        keep &= ~np.all(dist < 0.0, axis=1)
    idxs = np.nonzero(keep)[0]
    out = {"sx": [], "sy": [], "z": [], "w": [], "world": [], "nrm": [], "uv": [], "mat": []}
    for t in idxs:
        poly = [
            {"cam": cam[t, k], "world": world[t, k], "nrm": nrm[t, k], "uv": uv[t, k]}
            for k in range(3)
        ]
        for plane in rig.planes:
            poly = _clip_polygon_plane(poly, plane)
            if not poly:
                break
        if len(poly) < 3:
            continue
        # Fan re-triangulation of the clipped polygon.
        for k in range(1, len(poly) - 1):
            tri = [poly[0], poly[k], poly[k + 1]]
            cam_pts = np.stack([v["cam"] for v in tri])
            clip = math3d.apply_mat4(rig.proj, cam_pts)
            w = clip[:, 3]
            if np.any(w <= 1e-12):
                # Cannot happen after near-plane clipping; guard anyway.
                continue
            ndc = clip[:, :3] / w[:, None]
            sx = np.rint((ndc[:, 0] * 0.5 + 0.5) * width * SUBPIXEL).astype(np.int64)
            sy = np.rint((0.5 - ndc[:, 1] * 0.5) * height * SUBPIXEL).astype(np.int64)
            out["sx"].append(sx)
            out["sy"].append(sy)
            out["z"].append(ndc[:, 2])
            out["w"].append(w)
            out["world"].append(np.stack([v["world"] for v in tri]))
            out["nrm"].append(np.stack([v["nrm"] for v in tri]))
            out["uv"].append(np.stack([v["uv"] for v in tri]))
            out["mat"].append(int(mat[t]))
    return {
        "sx": np.asarray(out["sx"], dtype=np.int64).reshape(-1, 3),
        "sy": np.asarray(out["sy"], dtype=np.int64).reshape(-1, 3),
        "z": np.asarray(out["z"], dtype=np.float64).reshape(-1, 3),
        "w": np.asarray(out["w"], dtype=np.float64).reshape(-1, 3),
        "world": np.asarray(out["world"], dtype=np.float64).reshape(-1, 3, 3),
        "nrm": np.asarray(out["nrm"], dtype=np.float64).reshape(-1, 3, 3),
        "uv": np.asarray(out["uv"], dtype=np.float64).reshape(-1, 3, 2),
        "mat": np.asarray(out["mat"], dtype=np.int32).reshape(-1),
    }


def compute_tangents(stris: dict) -> None:
    """Per-triangle tangent/bitangent from UV derivatives (adds tangent/bitangent)."""
    world = stris["world"]
    uv = stris["uv"]
    dp1 = world[:, 1] - world[:, 0]
    dp2 = world[:, 2] - world[:, 0]
    duv1 = uv[:, 1] - uv[:, 0]
    duv2 = uv[:, 2] - uv[:, 0]
    det = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]
    face_n = np.cross(dp1, dp2)
    fn_len = np.linalg.norm(face_n, axis=1, keepdims=True)
    fn_len = np.maximum(fn_len, 1e-12)
    face_n = face_n / fn_len
    tangent = np.zeros_like(face_n)
    bitangent = np.zeros_like(face_n)
    ok = np.abs(det) > 1e-12
    r = np.zeros_like(det)
    r[ok] = 1.0 / det[ok]
    t = (dp1 * duv2[:, 1:2] - dp2 * duv1[:, 1:2]) * r[:, None]
    b = (dp2 * duv1[:, 0:1] - dp1 * duv2[:, 0:1]) * r[:, None]
    tangent[ok] = t[ok]
    bitangent[ok] = b[ok]
    # Fallback for degenerate UVs: any orthonormal basis around the face normal.
    bad = ~ok
    if np.any(bad):
        n = face_n[bad]
        ref = np.where(np.abs(n[:, 1:2]) < 0.9, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        tb = np.cross(n, ref)
        tb /= np.maximum(np.linalg.norm(tb, axis=1, keepdims=True), 1e-12)
        tangent[bad] = tb
        bitangent[bad] = np.cross(n, tb)
    stris["tangent"] = tangent
    stris["bitangent"] = bitangent
