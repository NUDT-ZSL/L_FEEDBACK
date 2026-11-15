"""Hierarchical scene description, loading and per-frame state updates."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .errors import RenderError
from . import mathutils


@dataclass
class Material:
    name: str
    base_color: np.ndarray            # linear RGBA in [0, 1], float64
    roughness: float
    metallic: float
    blend: str = "opaque"             # opaque | alpha | additive

    @property
    def opacity(self) -> float:
        return float(self.base_color[3])


@dataclass
class Mesh:
    name: str
    positions: np.ndarray             # (V, 3) float64
    normals: np.ndarray               # (V, 3) float64
    indices: np.ndarray               # (T*3,) int32

    @property
    def vertex_count(self) -> int:
        return self.positions.shape[0]

    @property
    def triangle_count(self) -> int:
        return self.indices.shape[0] // 3


@dataclass
class Node:
    name: str
    local: np.ndarray                 # float64 TRS
    translate: List[float]
    rotate: List[float]               # radians, euler XYZ
    scale: List[float]
    mesh: Optional[Mesh] = None
    material: Optional[Material] = None
    spin: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orbit: float = 0.0
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)
    world: np.ndarray = field(default_factory=mathutils.identity)
    depth: int = 0
    init_rotate: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    init_translate: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def recompute_local(self) -> None:
        self.local = mathutils.compose(self.translate, self.rotate, self.scale)


@dataclass
class Light:
    kind: str                         # directional | point
    direction: np.ndarray             # unit vector, direction the light travels
    position: np.ndarray
    color: np.ndarray                 # RGB linear
    intensity: float
    light_range: float = 1e9


@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_y: float
    near: float
    far: float


class Scene:
    def __init__(self, data: Dict[str, Any], base_dir: str = "") -> None:
        self.raw = data
        self.base_dir = base_dir
        self.width = int(data.get("width", 320))
        self.height = int(data.get("height", 240))
        if self.width <= 0 or self.height <= 0:
            raise RenderError("E_SIZE_MISMATCH", "Framebuffer size must be positive.",
                              {"width": self.width, "height": self.height})
        self.background = np.asarray(
            data.get("background", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
        cam = data["camera"]
        self.camera = Camera(
            eye=np.asarray(cam["eye"], dtype=np.float64),
            target=np.asarray(cam["target"], dtype=np.float64),
            up=np.asarray(cam.get("up", [0.0, 1.0, 0.0]), dtype=np.float64),
            fov_y=float(cam.get("fov_y", math.radians(60.0))),
            near=float(cam.get("near", 0.1)),
            far=float(cam.get("far", 100.0)),
        )
        self.materials: Dict[str, Material] = {
            name: self._parse_material(name, spec)
            for name, spec in data.get("materials", {}).items()
        }
        self.meshes: Dict[str, Mesh] = {
            name: self._parse_mesh(name, spec)
            for name, spec in data.get("meshes", {}).items()
        }
        self.lights: List[Light] = [self._parse_light(s) for s in data.get("lights", [])]
        self.ambient = np.asarray(data.get("ambient", [0.05, 0.05, 0.05]), dtype=np.float64)
        self.nodes: List[Node] = []
        self.roots: List[Node] = []
        self._parse_nodes(data.get("nodes", []))
        if not self.roots:
            raise RenderError("E_SCENE_INVALID", "Scene has no root nodes.")
        self.update_world_matrices()

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: str) -> "Scene":
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError as exc:
            raise RenderError("E_SCENE_INVALID", f"Scene file not found: {path}",
                              {"path": path}) from exc
        except json.JSONDecodeError as exc:
            raise RenderError("E_SCENE_INVALID", "Scene JSON is malformed.",
                              {"path": path, "line": exc.lineno, "col": exc.colno}) from exc
        except OSError as exc:
            raise RenderError("E_SCENE_INVALID", "Scene file is unreadable.",
                              {"path": path}) from exc
        if not isinstance(data, dict):
            raise RenderError("E_SCENE_INVALID", "Scene root must be a JSON object.")
        return cls(data, base_dir=os.path.dirname(os.path.abspath(path)))

    def _parse_material(self, name: str, spec: Dict[str, Any]) -> Material:
        color = spec.get("base_color", [0.8, 0.8, 0.8, 1.0])
        if len(color) != 4:
            raise RenderError("E_SCENE_INVALID", "Material base_color must have 4 channels.",
                              {"material": name})
        blend = spec.get("blend", "opaque")
        if blend not in ("opaque", "alpha", "additive"):
            raise RenderError("E_SCENE_INVALID", "Unknown blend mode.",
                              {"material": name, "blend": blend})
        return Material(
            name=name,
            base_color=np.asarray(color, dtype=np.float64),
            roughness=float(spec.get("roughness", 0.5)),
            metallic=float(spec.get("metallic", 0.0)),
            blend=blend,
        )

    def _parse_mesh(self, name: str, spec: Dict[str, Any]) -> Mesh:
        try:
            positions = np.asarray(spec["positions"], dtype=np.float64)
            normals = np.asarray(spec["normals"], dtype=np.float64)
            indices = np.asarray(spec["indices"], dtype=np.int32)
        except KeyError as exc:
            raise RenderError("E_MESH_INVALID", "Mesh is missing a required array.",
                              {"mesh": name, "missing": str(exc)}) from exc
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise RenderError("E_MESH_INVALID", "Mesh positions must be (V, 3).",
                              {"mesh": name, "shape": list(positions.shape)})
        if normals.shape != positions.shape:
            raise RenderError("E_MESH_INVALID", "Normal/position size mismatch.",
                              {"mesh": name,
                               "positions": list(positions.shape),
                               "normals": list(normals.shape)})
        if indices.ndim != 1 or indices.size % 3 != 0 or indices.size == 0:
            raise RenderError("E_MESH_INVALID", "Mesh indices must be triplets.",
                              {"mesh": name, "size": int(indices.size)})
        if int(indices.min()) < 0 or int(indices.max()) >= positions.shape[0]:
            raise RenderError("E_MESH_INVALID", "Index out of range.", {"mesh": name})
        return Mesh(name, positions, normals, indices)

    def _parse_light(self, spec: Dict[str, Any]) -> Light:
        kind = spec.get("type", "directional")
        if kind not in ("directional", "point"):
            raise RenderError("E_SCENE_INVALID", "Unsupported light type.", {"type": kind})
        direction = np.asarray(spec.get("direction", [0.0, -1.0, 0.0]), dtype=np.float64)
        if np.linalg.norm(direction) == 0:
            raise RenderError("E_SCENE_INVALID", "Light direction must be non-zero.")
        direction = direction / np.linalg.norm(direction)
        return Light(
            kind=kind,
            direction=direction,
            position=np.asarray(spec.get("position", [0.0, 0.0, 0.0]), dtype=np.float64),
            color=np.asarray(spec.get("color", [1.0, 1.0, 1.0]), dtype=np.float64),
            intensity=float(spec.get("intensity", 1.0)),
            light_range=float(spec.get("range", 1e9)),
        )

    def _parse_nodes(self, raw_nodes: Sequence[Dict[str, Any]],
                     parent: Optional[Node] = None) -> None:
        for spec in raw_nodes:
            name = spec.get("name", f"node_{len(self.nodes)}")
            translate = list(spec.get("translate", [0.0, 0.0, 0.0]))
            rotate = list(spec.get("rotate", [0.0, 0.0, 0.0]))
            scale = list(spec.get("scale", [1.0, 1.0, 1.0]))
            node = Node(
                name=name,
                local=mathutils.compose(translate, rotate, scale),
                translate=translate,
                rotate=rotate,
                scale=scale,
                spin=list(spec.get("spin", [0.0, 0.0, 0.0])),
                orbit=float(spec.get("orbit", 0.0)),
                parent=parent,
                depth=0 if parent is None else parent.depth + 1,
                init_rotate=list(rotate),
                init_translate=list(translate),
            )
            if "mesh" in spec:
                mesh_name = spec["mesh"]
                if mesh_name not in self.meshes:
                    raise RenderError("E_SCENE_MESH", "Node references undefined mesh.",
                                      {"node": name, "mesh": mesh_name})
                node.mesh = self.meshes[mesh_name]
                mat_name = spec.get("material")
                if mat_name is not None:
                    if mat_name not in self.materials:
                        raise RenderError("E_SCENE_INVALID",
                                          "Node references undefined material.",
                                          {"node": name, "material": mat_name})
                    node.material = self.materials[mat_name]
            self.nodes.append(node)
            if parent is None:
                self.roots.append(node)
            else:
                parent.children.append(node)
            self._parse_nodes(spec.get("children", []), node)

    # ------------------------------------------------------------- animation
    def update(self, frame: int, fps: float = 30.0) -> None:
        """Apply deterministic per-frame state, then refresh world matrices."""
        t = frame / fps
        for node in self.nodes:
            node.rotate = [node.init_rotate[i] + node.spin[i] * t for i in range(3)]
            if node.orbit:
                base = node.init_translate
                radius = math.hypot(base[0], base[2]) or 1.0
                angle = node.orbit * t + math.atan2(base[2], base[0])
                node.translate = [
                    radius * math.cos(angle),
                    base[1],
                    radius * math.sin(angle),
                ]
            node.recompute_local()
        self.update_world_matrices()

    def update_world_matrices(self) -> None:
        for root in self.roots:
            self._update_subtree(root, mathutils.identity())

    def _update_subtree(self, node: Node, parent_world: np.ndarray) -> None:
        node.world = parent_world @ node.local
        for child in node.children:
            self._update_subtree(child, node.world)

    def depth_first_nodes(self) -> List[Node]:
        ordered: List[Node] = []
        for root in self.roots:
            self._collect(root, ordered)
        return ordered

    def _collect(self, node: Node, out: List[Node]) -> None:
        out.append(node)
        for child in node.children:
            self._collect(child, out)
