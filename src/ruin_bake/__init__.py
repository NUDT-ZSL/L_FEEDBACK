"""Deterministic offline irradiance probe baking for procedural ruins."""

from .bake import BakeOutput, BakeParams, bake_full, bake_incremental
from .layout import LayoutParams, Probe, ProbeLayout, build_probe_layout
from .scene import (
    BoxGeometry,
    DirectionalLight,
    PointLight,
    Scene,
    SceneGeneratorConfig,
    apply_scene_patch,
    generate_scene,
)
from .snapshot import (
    ProbeRecord,
    Snapshot,
    SnapshotError,
    load_snapshot,
    write_snapshot_atomic,
)

__all__ = [
    "BakeParams",
    "BoxGeometry",
    "DirectionalLight",
    "LayoutParams",
    "PointLight",
    "Probe",
    "ProbeLayout",
    "ProbeRecord",
    "BakeOutput",
    "Scene",
    "SceneGeneratorConfig",
    "Snapshot",
    "SnapshotError",
    "apply_scene_patch",
    "bake_full",
    "bake_incremental",
    "build_probe_layout",
    "generate_scene",
    "load_snapshot",
    "write_snapshot_atomic",
]

