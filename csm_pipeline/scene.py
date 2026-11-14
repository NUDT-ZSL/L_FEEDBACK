"""Fixed-seed procedural scene and camera-path generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .geometry import Mesh, box, merge_meshes, plane, sphere
from .matrices import normalize


@dataclass
class CameraKeyframe:
    t: float
    eye: np.ndarray
    target: np.ndarray
    fovy_deg: float

    def to_dict(self) -> dict:
        return {
            "t": float(self.t),
            "eye": [float(x) for x in np.asarray(self.eye, dtype=np.float64)],
            "target": [float(x) for x in np.asarray(self.target, dtype=np.float64)],
            "fovy_deg": float(self.fovy_deg),
        }

    @staticmethod
    def from_dict(data: dict) -> "CameraKeyframe":
        eye = np.asarray(data["eye"], dtype=np.float64)
        target = np.asarray(data["target"], dtype=np.float64)
        if eye.shape != (3,) or target.shape != (3,):
            raise ValueError("keyframe eye/target must each contain 3 numbers")
        fovy = float(data["fovy_deg"])
        if not (0.0 < fovy < 180.0):
            raise ValueError("keyframe fovy_deg out of range")
        return CameraKeyframe(float(data["t"]), eye, target, fovy)


@dataclass
class Scene:
    seed: int
    meshes: list
    light_direction: np.ndarray
    light_up: np.ndarray
    keyframes: list
    rng_identifier: str = field(default="numpy.PCG64")

    @property
    def merged_mesh(self) -> Mesh:
        return merge_meshes(self.meshes)


def _stable_rng(seed: int) -> np.random.Generator:
    # PCG64 with an explicit integer seed is stable across processes and
    # platforms for a given NumPy major stream-compatibility family.
    return np.random.Generator(np.random.PCG64(int(seed)))


def generate_scene(seed: int = 20261115) -> Scene:
    """Produce the ground, occluders, light direction and camera keyframes.

    All random values are pulled from one PCG64 stream in a fixed order.
    """
    rng = _stable_rng(seed)
    meshes: list = [plane("ground", np.array((0.0, 0.0, 0.0)), 40.0, 40.0, 8)]

    cube_count = 5
    for index in range(cube_count):
        cx = float(rng.uniform(-8.0, 8.0))
        cz = float(rng.uniform(-8.0, 8.0))
        size = float(rng.uniform(1.0, 2.6))
        meshes.append(
            box(
                f"cube_{index}",
                np.array((cx, size * 0.5, cz), dtype=np.float64),
                np.array((size, size, size), dtype=np.float64),
            )
        )

    sphere_count = 4
    for index in range(sphere_count):
        cx = float(rng.uniform(-9.0, 9.0))
        cz = float(rng.uniform(-9.0, 9.0))
        radius = float(rng.uniform(0.7, 1.7))
        meshes.append(
            sphere(
                f"sphere_{index}",
                np.array((cx, radius, cz), dtype=np.float64),
                radius,
                segments=16,
                rings=10,
            )
        )

    azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
    elevation = float(rng.uniform(0.35, 0.95) * (np.pi * 0.5))
    light_direction = normalize(
        np.array(
            (
                np.cos(azimuth) * np.cos(elevation),
                -np.sin(elevation),
                np.sin(azimuth) * np.cos(elevation),
            ),
            dtype=np.float64,
        )
    )
    light_up = np.array((0.0, 1.0, 0.0), dtype=np.float64)

    angle_offset = float(rng.uniform(0.0, 2.0 * np.pi))
    keyframes: list = []
    for index, t in enumerate((0.0, 1.0, 2.0, 3.0)):
        angle = angle_offset + 2.0 * np.pi * index / 4.0
        eye = np.array(
            (9.0 * np.cos(angle), 5.5, 9.0 * np.sin(angle)), dtype=np.float64
        )
        keyframes.append(
            CameraKeyframe(
                t=float(t),
                eye=eye,
                target=np.array((0.0, 1.0, 0.0), dtype=np.float64),
                fovy_deg=55.0,
            )
        )

    return Scene(
        seed=int(seed),
        meshes=meshes,
        light_direction=light_direction,
        light_up=light_up,
        keyframes=keyframes,
    )


def scene_fingerprint(scene: Scene) -> str:
    """Deterministic SHA-256 over all generated geometry and the light vector."""
    hasher = hashlib.sha256()
    for mesh in scene.meshes:
        hasher.update(mesh.name.encode("utf-8"))
        hasher.update(np.asarray(mesh.positions, dtype="<f8").tobytes())
        hasher.update(np.asarray(mesh.normals, dtype="<f8").tobytes())
    hasher.update(np.asarray(scene.light_direction, dtype="<f8").tobytes())
    for keyframe in scene.keyframes:
        hasher.update(np.asarray(keyframe.eye, dtype="<f8").tobytes())
        hasher.update(np.asarray(keyframe.target, dtype="<f8").tobytes())
        hasher.update(np.float64(keyframe.fovy_deg).tobytes())
    return hasher.hexdigest()

