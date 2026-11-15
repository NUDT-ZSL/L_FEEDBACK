"""High level renderer: builds the shared draw list and drives backends."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .errors import RenderError
from . import mathutils
from .lighting import compute_vertex_colors
from .pipeline import DrawCall, StateCache
from .scene import Scene, Node


def build_draw_list(scene: Scene) -> List[DrawCall]:
    """Traverse the graph depth-first and emit one draw call per mesh node."""
    camera_position = scene.camera.eye
    draws: List[DrawCall] = []
    for node in scene.depth_first_nodes():
        if node.mesh is None:
            continue
        if node.material is None:
            raise RenderError("E_SCENE_INVALID", "Mesh node has no material.",
                              {"node": node.name})
        mesh = node.mesh
        world_f64 = node.world
        normal_m = mathutils.normal_matrix(world_f64)

        positions_h = np.concatenate(
            [mesh.positions, np.ones((mesh.vertex_count, 1), dtype=np.float64)], axis=1)
        world_positions = (world_f64 @ positions_h.T).T[:, :3]
        world_normals = (normal_m @ mesh.normals.T).T

        colors = compute_vertex_colors(
            world_positions=world_positions.astype(np.float32),
            world_normals=world_normals.astype(np.float32),
            base_color=node.material.base_color.astype(np.float32),
            roughness=node.material.roughness,
            metallic=node.material.metallic,
            lights=scene.lights,
            ambient=scene.ambient,
            camera_position=camera_position,
        )
        draws.append(
            DrawCall(
                node_name=node.name,
                positions=world_positions.astype(np.float32),
                normals=world_normals.astype(np.float32),
                colors=colors,
                indices=mesh.indices.astype(np.int32),
                blend=node.material.blend,
                depth_test=True,
                cull="back",
            )
        )
    return draws


class Renderer:
    """Owns a backend and accumulates frame statistics."""

    def __init__(self, backend, scene: Scene) -> None:
        self.backend = backend
        self.scene = scene
        self.state = StateCache()
        self.draw_calls = 0
        self.triangles = 0
        self.warnings: List[str] = []

    def render_frame(self, draws: Optional[List[DrawCall]] = None) -> np.ndarray:
        if draws is None:
            draws = build_draw_list(self.scene)
        framebuffer = self.backend.begin_frame(self.scene)
        for draw in draws:
            changes = self.state.apply(draw)
            self.backend.apply_state(draw, changes)
            self.backend.draw(draw)
            self.draw_calls += 1
            self.triangles += draw.triangle_count
        self.backend.end_frame()
        return framebuffer

    def statistics(self, elapsed_seconds: float) -> dict:
        stats = {
            "backend": self.backend.name,
            "width": self.scene.width,
            "height": self.scene.height,
            "draw_calls": self.draw_calls,
            "triangles": self.triangles,
            "state_switches": self.state.total_switches,
            "state_switch_detail": dict(self.state.switch_counts),
            "elapsed_seconds": round(float(elapsed_seconds), 6),
            "warnings": list(self.warnings),
        }
        stats.update(self.backend.backend_statistics())
        return stats
