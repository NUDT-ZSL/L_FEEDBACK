"""Bone hierarchy evaluation and linear-blend skinning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .basis import build_bind_basis
from .errors import numeric
from .math3d import (
    compose_trs,
    orthonormal_tangent,
    slerp_short,
    sphere_fallback_tangent,
)
from .scene import Keyframe, Scene


@dataclass(frozen=True)
class FrameSkin:
    time: float
    positions: np.ndarray
    normals: np.ndarray
    tangents: np.ndarray
    bitangents: np.ndarray
    bone_matrices: np.ndarray


def _prepared_keys(keys: tuple[Keyframe, ...], duration: float, loop: bool) -> tuple[Keyframe, ...]:
    prepared = list(keys)
    if prepared[0].time > 1e-9:
        first = prepared[0]
        prepared.insert(0, Keyframe(0.0, first.translation, first.rotation, first.scale))
    if prepared[-1].time < duration - 1e-9:
        if loop:
            first = prepared[0]
            prepared.append(
                Keyframe(duration, first.translation, first.rotation, first.scale)
            )
        else:
            last = prepared[-1]
            prepared.append(Keyframe(duration, last.translation, last.rotation, last.scale))
    elif loop and prepared[-1].time >= duration - 1e-9:
        first = prepared[0]
        prepared[-1] = Keyframe(
            duration, first.translation, first.rotation, first.scale
        )
    return tuple(prepared)


class AnimationSampler:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.duration = scene.duration
        self.loop = scene.loop_mode == "loop"
        self.bind_basis = build_bind_basis(scene.normals, scene.tangents)
        self.keys: dict[int, tuple[Keyframe, ...]] = {}
        for bone, track in scene.tracks.items():
            self.keys[bone] = _prepared_keys(track.keys, self.duration, self.loop)

    @staticmethod
    def _sample_keys(keys: tuple[Keyframe, ...], time: float) -> Keyframe:
        if time <= keys[0].time:
            return keys[0]
        if time >= keys[-1].time:
            return keys[-1]
        lo = 0
        hi = len(keys) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if keys[mid].time <= time:
                lo = mid
            else:
                hi = mid
        a = keys[lo]
        b = keys[hi]
        span = b.time - a.time
        alpha = 0.0 if span <= 1e-15 else (time - a.time) / span
        translation = a.translation + (b.translation - a.translation) * alpha
        rotation = slerp_short(a.rotation, b.rotation, alpha)
        scale = a.scale + (b.scale - a.scale) * alpha
        return Keyframe(time, translation, rotation, scale)

    def local_matrix(self, bone: int, time: float) -> np.ndarray:
        if bone in self.keys:
            key = self._sample_keys(self.keys[bone], time)
            return compose_trs(key.translation, key.rotation, key.scale)
        return self.scene.bones[bone].bind_local

    def world_matrices(self, time: float) -> np.ndarray:
        count = len(self.scene.bones)
        worlds = np.empty((count, 4, 4), dtype=np.float64)
        for i, bone in enumerate(self.scene.bones):
            local = self.local_matrix(i, time)
            if bone.parent < 0:
                worlds[i] = local
            else:
                worlds[i] = worlds[bone.parent] @ local
        if not np.all(np.isfinite(worlds)):
            raise numeric(f"non-finite bone matrix at time {time}")
        return worlds

    def skin_matrices(self, time: float) -> np.ndarray:
        worlds = self.world_matrices(time)
        inverse_bind = np.stack([bone.inverse_bind for bone in self.scene.bones], axis=0)
        return worlds @ inverse_bind

    def skin(self, time: float) -> FrameSkin:
        scene = self.scene
        matrices = self.skin_matrices(time)
        blended = np.einsum("vk,vkij->vij", scene.bone_weights, matrices[scene.bone_indices])
        positions = (
            np.einsum("vij,vj->vi", blended[:, :3, :3], scene.positions)
            + blended[:, :3, 3]
        )
        raw_normals = np.einsum(
            "vij,vj->vi", blended[:, :3, :3], self.bind_basis.normals
        )
        raw_tangents = np.einsum(
            "vij,vj->vi", blended[:, :3, :3], self.bind_basis.tangents
        )

        normals = np.empty_like(raw_normals)
        tangents = np.empty_like(raw_tangents)
        bitangents = np.empty_like(raw_tangents)
        for i in range(scene.vertex_count):
            n_len = float(np.linalg.norm(raw_normals[i]))
            n = (
                raw_normals[i] / n_len
                if math.isfinite(n_len) and n_len > 1e-12
                else np.array([0.0, 0.0, 1.0], dtype=np.float64)
            )
            built = orthonormal_tangent(n, raw_tangents[i])
            if built is None:
                t = sphere_fallback_tangent(n)
                b = np.cross(n, t)
                b /= max(float(np.linalg.norm(b)), 1e-15)
            else:
                t, b = built
            handedness = 1.0 if scene.tangents[i, 3] >= 0.0 else -1.0
            normals[i] = n
            tangents[i] = t
            bitangents[i] = b * handedness

        return FrameSkin(time, positions, normals, tangents, bitangents, matrices)


def animation_frame_times(scene: Scene) -> np.ndarray:
    frame_count = int(round(scene.duration * scene.fps))
    if scene.loop_mode == "loop":
        return np.arange(frame_count, dtype=np.float64) / scene.fps
    return np.arange(frame_count + 1, dtype=np.float64) / scene.fps


def interpolate_curve(
    curve_times: np.ndarray,
    curve_values: np.ndarray,
    time: float,
    duration: float,
    loop: bool,
) -> np.ndarray:
    if loop:
        t = time % duration
    else:
        t = min(duration, max(0.0, time))
    if not loop and t <= curve_times[0]:
        return curve_values[0].copy()
    if t >= curve_times[-1] and not loop:
        return curve_values[-1].copy()
    if not loop:
        hi = int(np.searchsorted(curve_times, t, side="right"))
        lo = hi - 1
        span = curve_times[hi] - curve_times[lo]
        alpha = 0.0 if span <= 1e-15 else (t - curve_times[lo]) / span
        return curve_values[lo] + (curve_values[hi] - curve_values[lo]) * alpha

    extension_times: list[float] = []
    extension_values: list[np.ndarray] = []
    if curve_times[0] > 1e-9:
        extension_times.append(float(curve_times[-1] - duration))
        extension_values.append(curve_values[-1])
    extension_times.extend(float(v) for v in curve_times)
    extension_values.extend(curve_values)
    if curve_times[-1] < duration - 1e-9:
        extension_times.append(float(curve_times[0] + duration))
        extension_values.append(curve_values[0])
    ext_t = np.asarray(extension_times, dtype=np.float64)
    ext_v = np.asarray(extension_values, dtype=np.float64)
    if t <= ext_t[0]:
        return ext_v[0].copy()
    if t >= ext_t[-1]:
        return ext_v[-1].copy()
    hi = int(np.searchsorted(ext_t, t, side="right"))
    lo = hi - 1
    span = ext_t[hi] - ext_t[lo]
    alpha = 0.0 if span <= 1e-15 else (t - ext_t[lo]) / span
    return ext_v[lo] + (ext_v[hi] - ext_v[lo]) * alpha

