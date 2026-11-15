"""SkeletalAnimator: deterministic mesh, animation sampling and FK."""

from __future__ import annotations

import numpy as np

from . import mesh as mesh_mod
from .errors import ErrorLog, E_NON_FINITE_MATRIX


def _translation_matrix(offset: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = offset
    return m


def _rotation_z_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4)
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return m


class SkeletalAnimator:
    """Two-bone chain (root + child) driving a skinned capsule mesh.

    The animation is sampled at a fixed frame rate: the root bone translates
    along a deterministic Lissajous-like path and the child bone rotates
    about its local Z axis. Per-seed phases keep runs reproducible.
    """

    def __init__(self, seed: int = 0, fps: float = 24.0) -> None:
        self.fps = float(fps)
        rng = np.random.default_rng(seed)
        # Deterministic per-seed animation phases.
        self._phase_tx, self._phase_ty, self._phase_rot = rng.uniform(
            0.0, 2.0 * np.pi, size=3
        )
        self.vertices, self.triangles = mesh_mod.make_capsule()
        self.weights = mesh_mod.compute_skin_weights(self.vertices)
        # Bind pose: root at identity, child offset along +Y.
        self.bind_world = [
            np.eye(4),
            _translation_matrix(mesh_mod.CHILD_OFFSET),
        ]
        self.inv_bind_world = [np.linalg.inv(m) for m in self.bind_world]

    def root_translation(self, t: float) -> np.ndarray:
        tx = 1.6 * np.sin(2.0 * np.pi * 0.5 * t + self._phase_tx)
        ty = 0.3 * np.sin(2.0 * np.pi * 0.25 * t + self._phase_ty)
        return np.array([tx, ty, 0.0])

    def child_rotation_angle(self, t: float) -> float:
        return 0.9 * np.sin(2.0 * np.pi * 1.0 * t + self._phase_rot)

    def frame_world_matrices(
        self, frame: int, error_log: ErrorLog | None = None
    ) -> list[np.ndarray] | None:
        """Forward kinematics for one frame.

        Returns [root_world, child_world] 4x4 matrices, or None if any
        matrix element is non-finite (reported as E_NON_FINITE_MATRIX).
        """
        t = frame / self.fps
        root_local = _translation_matrix(self.root_translation(t))
        child_local = _translation_matrix(mesh_mod.CHILD_OFFSET) @ \
            _rotation_z_matrix(self.child_rotation_angle(t))
        world = [root_local, root_local @ child_local]
        for bone_idx, m in enumerate(world):
            if not np.all(np.isfinite(m)):
                if error_log is not None:
                    error_log.report(
                        frame,
                        E_NON_FINITE_MATRIX,
                        f"bone {bone_idx} world matrix has non-finite "
                        f"elements; frame skipped",
                    )
                return None
        return world

    def skin_matrices(self, world: list[np.ndarray]) -> list[np.ndarray]:
        """world @ inverse(bind) for each bone."""
        return [w @ ib for w, ib in zip(world, self.inv_bind_world)]
