"""Deterministic procedural sample assets: mech OBJ models, materials,
normal maps and a scene description. Everything is derived from a seed.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from . import normalmap


class MeshBuilder:
    """Accumulates flat-shaded triangles: (pos 3x3, normal 3, uv 3x2)."""

    def __init__(self):
        self.tris = []  # (p0, p1, p2, normal, uv0, uv1, uv2)

    def _quad(self, a, b, c, d, normal, uvs):
        self.tris.append((a, b, c, normal, uvs[0], uvs[1], uvs[2]))
        self.tris.append((a, c, d, normal, uvs[0], uvs[2], uvs[3]))

    def box(self, center, size, uv_scale=0.5):
        cx, cy, cz = center
        sx, sy, sz = (v * 0.5 for v in size)
        # (normal, 4 corners CCW seen from outside, uv axes)
        faces = [
            ((0, 0, 1), [(-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)], (0, 1)),
            ((0, 0, -1), [(sx, -sy, -sz), (-sx, -sy, -sz), (-sx, sy, -sz), (sx, sy, -sz)], (0, 1)),
            ((1, 0, 0), [(sx, -sy, sz), (sx, -sy, -sz), (sx, sy, -sz), (sx, sy, sz)], (2, 1)),
            ((-1, 0, 0), [(-sx, -sy, -sz), (-sx, -sy, sz), (-sx, sy, sz), (-sx, sy, -sz)], (2, 1)),
            ((0, 1, 0), [(-sx, sy, sz), (sx, sy, sz), (sx, sy, -sz), (-sx, sy, -sz)], (0, 2)),
            ((0, -1, 0), [(-sx, -sy, -sz), (sx, -sy, -sz), (sx, -sy, sz), (-sx, -sy, sz)], (0, 2)),
        ]
        for n, corners, (a0, a1) in faces:
            pts = [np.array([p[0] + cx, p[1] + cy, p[2] + cz]) for p in corners]
            uvs = [np.array([p[a0] * uv_scale, p[a1] * uv_scale]) for p in pts]
            self._quad(pts[0], pts[1], pts[2], pts[3], np.array(n, dtype=float), uvs)

    def cylinder_y(self, center, radius, height, segments=10, uv_scale=0.5):
        cx, cy, cz = center
        hy = height * 0.5
        for k in range(segments):
            a0 = 2.0 * math.pi * k / segments
            a1 = 2.0 * math.pi * (k + 1) / segments
            x0, z0 = math.cos(a0), math.sin(a0)
            x1, z1 = math.cos(a1), math.sin(a1)
            p = [
                np.array([cx + radius * x0, cy - hy, cz + radius * z0]),
                np.array([cx + radius * x1, cy - hy, cz + radius * z1]),
                np.array([cx + radius * x1, cy + hy, cz + radius * z1]),
                np.array([cx + radius * x0, cy + hy, cz + radius * z0]),
            ]
            nrm = np.array([math.cos((a0 + a1) * 0.5), 0.0, math.sin((a0 + a1) * 0.5)])
            uvs = [
                np.array([a0 * radius * uv_scale, -hy * uv_scale]),
                np.array([a1 * radius * uv_scale, -hy * uv_scale]),
                np.array([a1 * radius * uv_scale, hy * uv_scale]),
                np.array([a0 * radius * uv_scale, hy * uv_scale]),
            ]
            self._quad(p[0], p[1], p[2], p[3], nrm, uvs)
            for sign, y in ((1.0, hy), (-1.0, -hy)):
                c = np.array([cx, cy + y, cz])
                pa = np.array([cx + radius * x0, cy + y, cz + radius * z0])
                pb = np.array([cx + radius * x1, cy + y, cz + radius * z1])
                n = np.array([0.0, sign, 0.0])
                uvc = np.array([0.0, 0.0])
                uva = np.array([x0 * radius * uv_scale, z0 * radius * uv_scale])
                uvb = np.array([x1 * radius * uv_scale, z1 * radius * uv_scale])
                if sign > 0:
                    self.tris.append((c, pb, pa, n, uvc, uvb, uva))
                else:
                    self.tris.append((c, pa, pb, n, uvc, uva, uvb))


def build_mech(rng: np.random.Generator) -> MeshBuilder:
    """A hard-surface mech assembled from boxes and cylinders."""

    def j(scale=0.1):
        return 1.0 + float(rng.uniform(-scale, scale))

    b = MeshBuilder()
    uv = 1.8
    b.box((0, 2.05 * j(), 0), (1.15 * j(), 0.5 * j(), 0.72 * j()), uv)          # pelvis
    b.box((0, 2.85 * j(), 0), (1.55 * j(), 1.15 * j(), 0.95 * j()), uv)         # torso
    b.box((0, 3.0, 0.46), (1.25 * j(), 0.72 * j(), 0.16), uv)                   # chest plate
    b.box((0, 2.32, 0), (0.92 * j(), 0.3, 0.6), uv)                             # waist
    b.box((0, 2.9, -0.58), (1.05 * j(), 0.95 * j(), 0.36), uv)                  # backpack
    b.box((0, 3.72 * j(), 0.04), (0.56 * j(), 0.5 * j(), 0.6 * j()), uv)        # head
    b.box((0, 3.76, 0.36), (0.42 * j(), 0.12, 0.06), uv)                        # visor
    b.cylinder_y((0, 3.42, 0), 0.16 * j(), 0.3, 10, uv)                         # neck
    b.cylinder_y((0.2, 4.15, -0.05), 0.03, 0.55 * j(), 6, uv)                   # antenna
    for side in (-1.0, 1.0):
        sx = side * j(0.05)
        b.box((sx * 1.08, 3.28 * j(), 0), (0.72 * j(), 0.52 * j(), 0.72 * j()), uv)   # shoulder
        b.box((sx * 1.18, 2.72, 0), (0.36 * j(), 0.7 * j(), 0.4 * j()), uv)           # upper arm
        b.box((sx * 1.18, 2.28, 0), (0.3, 0.24, 0.34), uv)                            # elbow
        b.box((sx * 1.18, 1.82, 0), (0.42 * j(), 0.66 * j(), 0.46 * j()), uv)         # forearm
        b.box((sx * 1.18, 1.38, 0.02), (0.3, 0.3, 0.34), uv)                          # hand
    for side in (-1.0, 1.0):
        sx = side * j(0.05)
        b.box((sx * 0.42, 1.68, 0), (0.42, 0.3, 0.5), uv)                       # hip
        b.box((sx * 0.44, 1.14 * j(), 0), (0.44 * j(), 0.85 * j(), 0.52 * j()), uv)   # thigh
        b.box((sx * 0.44, 0.68, 0.12), (0.4, 0.2, 0.42), uv)                    # knee
        b.box((sx * 0.44, 0.36 * j(), 0), (0.4 * j(), 0.6 * j(), 0.46 * j()), uv)     # shin
        b.box((sx * 0.44, 0.09, 0.12), (0.46 * j(), 0.18, 0.82 * j()), uv)      # foot
    for side in (-1.0, 1.0):
        b.cylinder_y((side * 0.36, 2.62, -0.78), 0.18 * j(), 0.42, 10, uv)      # thruster
    return b


def build_ground() -> MeshBuilder:
    b = MeshBuilder()
    b.box((0.0, -0.26, 0.0), (20.0, 0.5, 20.0), uv_scale=0.35)
    return b


def write_obj(path: str, mesh: MeshBuilder, negative_indices: bool = False) -> None:
    """Write mesh as OBJ with deduplicated (v, vt, vn) corners."""
    verts, uvs, norms = [], [], []
    v_map, t_map, n_map = {}, {}, {}
    faces = []

    def intern(table, lst, key):
        idx = table.get(key)
        if idx is None:
            idx = len(lst)
            lst.append(key)
            table[key] = idx
        return idx

    for (p0, p1, p2, n, t0, t1, t2) in mesh.tris:
        nrm = tuple(np.round(np.asarray(n, dtype=float), 12))
        ni = intern(n_map, norms, nrm)
        corner = []
        for p, t in ((p0, t0), (p1, t1), (p2, t2)):
            pk = tuple(np.round(np.asarray(p, dtype=float), 12))
            tk = tuple(np.round(np.asarray(t, dtype=float), 12))
            vi = intern(v_map, verts, pk)
            ti = intern(t_map, uvs, tk)
            corner.append((vi, ti, ni))
        faces.append(corner)

    nv, nt, nn = len(verts), len(uvs), len(norms)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# procedurally generated, deterministic\n")
        for v in verts:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in uvs:
            fh.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
        for n in norms:
            fh.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for face in faces:
            parts = []
            for vi, ti, ni in face:
                if negative_indices:
                    parts.append(f"{vi - nv}/{ti - nt}/{ni - nn}")
                else:
                    parts.append(f"{vi + 1}/{ti + 1}/{ni + 1}")
            fh.write("f " + " ".join(parts) + "\n")


def generate_assets(directory: str, seed: int) -> str:
    """Generate all sample assets deterministically; returns scene.json path."""
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(seed)

    normalmap.write_normal_map(os.path.join(directory, "nmap_panels.png"), seed + 101, 128, "panels")
    normalmap.write_normal_map(os.path.join(directory, "nmap_noise.png"), seed + 202, 128, "noise")

    mechs = [
        ("mech_a", [-3.4, 0.0, 0.6], 14.0, [0.55, 0.58, 0.64], 0.32),
        ("mech_b", [0.3, 0.0, -1.4], -9.0, [0.5, 0.38, 0.3], 0.5),
        ("mech_c", [3.6, 0.0, 1.0], 27.0, [0.32, 0.44, 0.52], 0.4),
    ]
    models = []
    for i, (name, pos, rot, albedo, rough) in enumerate(mechs):
        mesh = build_mech(np.random.default_rng(seed + 1000 + i * 77))
        write_obj(os.path.join(directory, f"{name}.obj"), mesh, negative_indices=(i == 1))
        material = {
            "albedo": [round(float(a + rng.uniform(-0.02, 0.02)), 4) for a in albedo],
            "roughness": round(float(rough + rng.uniform(-0.03, 0.03)), 4),
            "normal_map": "nmap_panels.png",
        }
        with open(os.path.join(directory, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(material, fh, indent=2)
        models.append({
            "obj": f"{name}.obj",
            "material": f"{name}.json",
            "transform": {"translate": pos, "rotate_y_deg": rot},
        })

    write_obj(os.path.join(directory, "ground.obj"), build_ground())
    with open(os.path.join(directory, "ground.json"), "w", encoding="utf-8") as fh:
        json.dump({"albedo": [0.3, 0.31, 0.33], "roughness": 0.85,
                   "normal_map": "nmap_noise.png"}, fh, indent=2)
    models.append({"obj": "ground.obj", "material": "ground.json"})

    def jitter(a):
        return round(float(a + rng.uniform(-0.05, 0.05)), 4)

    scene = {
        "models": models,
        "lights": {
            "directional": [
                {"direction": [jitter(-0.55), jitter(-1.0), jitter(-0.8)],
                 "color": [1.0, 0.95, 0.87], "intensity": 2.6},
                {"direction": [jitter(0.75), jitter(-0.45), jitter(0.35)],
                 "color": [0.5, 0.62, 0.82], "intensity": 1.1},
            ],
            "point": [
                {"position": [jitter(2.2), jitter(4.6), jitter(4.2)],
                 "color": [1.0, 0.78, 0.58], "intensity": 34.0, "min_distance": 0.25}
            ],
        },
        "ambient": [0.035, 0.038, 0.045],
    }
    scene_path = os.path.join(directory, "scene.json")
    with open(scene_path, "w", encoding="utf-8") as fh:
        json.dump(scene, fh, indent=2)
    return scene_path
