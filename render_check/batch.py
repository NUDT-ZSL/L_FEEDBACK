"""Deterministic material-instance batch construction."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .geometry import MeshData, generate_builtin_meshes
from .materials import Material
from .mathutil import look_at, model_matrix, perspective


CAMERA_EYE = (0.0, 0.0, 6.0)
CAMERA_TARGET = (0.0, 0.0, 0.0)
CAMERA_UP = (0.0, 1.0, 0.0)
CAMERA_FOV_Y = 45.0
CAMERA_NEAR = 0.1
CAMERA_FAR = 100.0
LIGHT_POSITION = (3.0, 4.0, 5.0)
LIGHT_COLOR = (1.0, 0.97, 0.90)


@dataclass(frozen=True)
class InstanceSpec:
    index: int
    material_index: int
    material_name: str
    shape: str
    object_id: int
    model: np.ndarray
    rotation: tuple[float, float, float]
    position: tuple[float, float, float]
    scale: float


@dataclass(frozen=True)
class Scene:
    instances: tuple[InstanceSpec, ...]
    meshes: dict[str, MeshData]
    view: np.ndarray
    projection: np.ndarray
    camera_position: tuple[float, float, float]
    light_position: tuple[float, float, float]
    light_color: tuple[float, float, float]


def _stable_instance_seed(seed: int, index: int) -> int:
    mask = (1 << 64) - 1
    mixed = ((seed & mask) ^ (((index + 1) * 0x9E3779B97F4A7C15) & mask)) & mask
    return mixed


def build_scene(
    materials: list[Material],
    instances: int,
    seed: int,
    width: int,
    height: int,
) -> Scene:
    meshes = generate_builtin_meshes()
    specs: list[InstanceSpec] = []
    for index in range(instances):
        material_index = index % len(materials)
        material = materials[material_index]
        shape = "cube" if (index + seed) % 2 == 0 else "sphere"
        rng = random.Random(_stable_instance_seed(seed, index))
        rotation = (
            rng.uniform(-0.55, 0.55),
            rng.uniform(0.0, 6.283185307179586),
            rng.uniform(-0.28, 0.28),
        )
        position = (
            rng.uniform(-0.12, 0.12),
            rng.uniform(-0.10, 0.10),
            0.0,
        )
        instance_scale = rng.uniform(1.05, 1.24)
        specs.append(
            InstanceSpec(
                index=index,
                material_index=material_index,
                material_name=material.name,
                shape=shape,
                object_id=index + 1,
                model=model_matrix(position, rotation, instance_scale),
                rotation=rotation,
                position=position,
                scale=instance_scale,
            )
        )

    view = look_at(CAMERA_EYE, CAMERA_TARGET, CAMERA_UP)
    projection = perspective(CAMERA_FOV_Y, width / height, CAMERA_NEAR, CAMERA_FAR)
    return Scene(
        instances=tuple(specs),
        meshes=meshes,
        view=view,
        projection=projection,
        camera_position=CAMERA_EYE,
        light_position=LIGHT_POSITION,
        light_color=LIGHT_COLOR,
    )

