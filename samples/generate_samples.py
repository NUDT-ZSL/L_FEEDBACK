"""Deterministically generate local acceptance scenes."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skin_bake.scene import write_scene  # noqa: E402


def _quat_axis_angle(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    half = math.radians(degrees) * 0.5
    return np.array([*(axis * math.sin(half)), math.cos(half)], dtype=np.float64)


def _bone(name: str, parent: int, translation: list[float], bind_world: np.ndarray) -> dict:
    local = np.eye(4, dtype=np.float64)
    local[:3, 3] = translation
    world = bind_world @ local
    return {
        "name": name,
        "parent": parent,
        "bindLocalMatrix": local.tolist(),
        "inverseBindMatrix": np.linalg.inv(world).tolist(),
        "_bindWorld": world,
    }


def build_scene(loop_mode: str = "loop", *, fps: int = 24, duration: float = 2.0) -> dict:
    nx, ny = 15, 9
    xs = np.linspace(-1.2, 1.2, nx)
    ys = np.linspace(-0.9, 0.9, ny)
    positions = []
    normals = []
    tangents = []
    uvs = []
    weights = []
    indices = []
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            z = 0.08 * math.cos(1.5 * x) * math.sin(1.3 * y)
            dzdx = -0.08 * 1.5 * math.sin(1.5 * x) * math.sin(1.3 * y)
            dzdy = 0.08 * 1.3 * math.cos(1.5 * x) * math.cos(1.3 * y)
            tx = np.array([1.0, 0.0, dzdx])
            ty = np.array([0.0, 1.0, dzdy])
            tx /= np.linalg.norm(tx)
            ty /= np.linalg.norm(ty)
            normal = np.cross(tx, ty)
            normal /= np.linalg.norm(normal)
            positions.append([x, y, z])
            normals.append(normal.tolist())
            tangents.append([*tx.tolist(), 1.0])
            uvs.append([(x + 1.2) / 2.4, (y + 0.9) / 1.8])

            t = (x + 1.2) / 2.4
            yt = (y + 0.9) / 1.8
            raw_weights = np.array(
                [
                    max(0.08, 1.0 - t),
                    max(0.04, 1.0 - abs(t - 0.5) * 2.0),
                    max(0.08, t),
                    0.06 + 0.10 * yt,
                ]
            )
            raw_weights /= raw_weights.sum()
            weights.append(raw_weights.tolist())
            indices.append([0, 1, 2, 3])

    degenerate_vertex = 7
    tangents[degenerate_vertex] = [0.0, 0.0, 0.0, 1.0]

    triangles = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            a = iy * nx + ix
            b = a + 1
            c = a + nx
            d = c + 1
            triangles.extend([[a, b, c], [b, d, c]])

    root_world = np.eye(4)
    b1 = _bone("twist_x", 0, [0.78, 0.0, 0.0], root_world)
    b2 = _bone("twist_x_end", 1, [0.78, 0.0, 0.0], b1["_bindWorld"])
    b3 = _bone("cloth_flap", 0, [0.0, -0.85, 0.0], root_world)
    bones = [b1, b2, b3]
    bones.insert(
        0,
        {
            "name": "root",
            "parent": -1,
            "bindLocalMatrix": root_world.tolist(),
            "inverseBindMatrix": np.eye(4).tolist(),
        },
    )
    for bone in bones:
        bone.pop("_bindWorld", None)

    def track(bone: int, keys: list[tuple[float, tuple, tuple, tuple]]) -> dict:
        return {
            "bone": bone,
            "keyframes": [
                {
                    "time": t,
                    "translation": list(translation),
                    "rotation": list(rotation),
                    "scale": list(scale),
                }
                for t, translation, rotation, scale in keys
            ],
        }

    rot_a = _quat_axis_angle(np.array([0.0, 0.0, 1.0]), 6.0)
    rot_b = _quat_axis_angle(np.array([0.0, 0.0, 1.0]), -5.0)
    rot_c = _quat_axis_angle(np.array([1.0, 0.0, 0.0]), 4.0)
    rot_d = _quat_axis_angle(np.array([0.2, 1.0, 0.0]), -7.0)
    tracks = [
        track(
            1,
            [
                (0.0, (0.78, 0.0, 0.0), tuple(rot_a), (1.0, 1.0, 1.0)),
                (1.0, (0.78, 0.04, 0.0), tuple(-rot_a), (1.02, 0.98, 1.0)),
                (2.0, (0.78, 0.0, 0.0), tuple(rot_a), (1.0, 1.0, 1.0)),
            ],
        ),
        track(
            2,
            [
                (0.0, (0.78, 0.0, 0.0), tuple(rot_b), (1.0, 1.0, 1.0)),
                (0.5, (0.78, 0.0, 0.0), tuple(-rot_c), (1.0, 1.04, 0.98)),
                (1.0, (0.78, -0.03, 0.0), tuple(rot_d), (1.0, 1.0, 1.0)),
                (1.5, (0.78, 0.0, 0.0), tuple(-rot_d), (0.98, 1.0, 1.03)),
                (2.0, (0.78, 0.0, 0.0), tuple(rot_b), (1.0, 1.0, 1.0)),
            ],
        ),
        track(
            3,
            [
                (0.0, (0.0, -0.85, 0.0), tuple(_quat_axis_angle(np.array([1.0, 0.0, 0.0]), 0.0)), (1.0, 1.0, 1.0)),
                (0.7, (0.0, -0.85, 0.04), tuple(_quat_axis_angle(np.array([1.0, 0.0, 0.0]), 9.0)), (1.0, 1.0, 1.0)),
                (1.4, (0.0, -0.85, -0.02), tuple(_quat_axis_angle(np.array([1.0, 0.0, 0.0]), -8.0)), (1.0, 1.0, 1.0)),
                (2.0, (0.0, -0.85, 0.0), tuple(_quat_axis_angle(np.array([1.0, 0.0, 0.0]), 0.0)), (1.0, 1.0, 1.0)),
            ],
        ),
    ]

    curves = []
    key_times = np.linspace(0.0, duration, 17)
    for vertex in range(len(positions)):
        ix = vertex % nx
        iy = vertex // nx
        t = (ix / (nx - 1)) * 2.0 - 1.0
        s = (iy / (ny - 1)) * 2.0 - 1.0
        cycles_u = 1 + (ix + iy) % 3
        cycles_v = 1 + (ix * 2 + iy) % 2
        amp_u = 0.011 * (1.0 + 0.35 * abs(t))
        amp_v = 0.008 * (1.0 + 0.25 * abs(s))
        offsets = []
        for key_time in key_times:
            phase = 2.0 * math.pi * key_time / duration
            du = amp_u * math.sin(cycles_u * phase + 0.35 * t)
            dv = amp_v * math.cos(cycles_v * phase + 0.6 * s)
            dn = 0.0
            offsets.append({"time": float(key_time), "offset": [du, dv, dn]})
        curves.append({"vertex": vertex, "keys": offsets})

    document = {
        "description": "Deterministic cloth grid with four-weight skinning and one degenerate tangent.",
        "mesh": {
            "positions": positions,
            "normals": normals,
            "tangents": tangents,
            "uvs": uvs,
            "boneWeights": weights,
            "boneIndices": indices,
            "triangles": triangles,
            "skeleton": bones,
        },
        "animation": {
            "fps": fps,
            "duration": duration,
            "loopMode": loop_mode,
            "boneTracks": tracks,
        },
        "displacement": {
            "space": "bind_normal_frame",
            "integration": "delta",
            "vertexCurves": curves,
        },
        "bake": {
            "textureSampleRate": 60.0,
            "atlasWidth": 128,
            "atlasHeight": 128,
            "minBlockSize": 1,
            "padding": 1,
            "errorTolerance": 0.03,
        },
    }
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "samples" / "generated")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for mode in ("loop", "clamp"):
        write_scene(args.out / f"scene_{mode}.json", build_scene(mode))
    print(f"wrote {args.out / 'scene_loop.json'}")
    print(f"wrote {args.out / 'scene_clamp.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

