"""Transparent panel ordering by view-space depth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .interpolation import CameraState, rotate_vector
from .scene import GlassPanel, Scene
from .util import quantize


SORT_QUANTIZATION_DECIMALS = 6


@dataclass(frozen=True)
class PanelOrder:
    order_index: int
    panel_id: int
    view_depth: float
    depth_key: float
    front_facing: bool
    behind_camera: bool
    rendered: bool

    def stats(self) -> dict[str, Any]:
        return {
            "order_index": self.order_index,
            "panel_id": self.panel_id,
            "view_depth": quantize(self.view_depth, SORT_QUANTIZATION_DECIMALS),
            "depth_key": quantize(self.depth_key, SORT_QUANTIZATION_DECIMALS),
            "front_facing": self.front_facing,
            "behind_camera": self.behind_camera,
            "rendered": self.rendered,
        }


def panel_view_depth(panel: GlassPanel, camera: CameraState) -> float:
    delta = (
        panel.center[0] - camera.position[0],
        panel.center[1] - camera.position[1],
        panel.center[2] - camera.position[2],
    )
    return float(
        delta[0] * camera.forward[0]
        + delta[1] * camera.forward[1]
        + delta[2] * camera.forward[2]
    )


def panel_is_front_facing(panel: GlassPanel, camera: CameraState) -> bool:
    to_camera = (
        camera.position[0] - panel.center[0],
        camera.position[1] - panel.center[1],
        camera.position[2] - panel.center[2],
    )
    dot = (
        panel.normal[0] * to_camera[0]
        + panel.normal[1] * to_camera[1]
        + panel.normal[2] * to_camera[2]
    )
    return dot > 1e-9


def order_panels(scene: Scene, camera: CameraState) -> tuple[PanelOrder, ...]:
    raw: list[tuple[float, float, GlassPanel, bool, bool]] = []
    for panel in scene.glass_panels:
        depth = panel_view_depth(panel, camera)
        depth_key = quantize(depth, SORT_QUANTIZATION_DECIMALS)
        front = panel_is_front_facing(panel, camera)
        behind = depth <= 0.0
        raw.append((depth, depth_key, panel, front, behind))

    # Far to near.  Quantized depth is the public sorting key; panel ID is
    # the deterministic secondary key for equal depths.
    raw.sort(key=lambda item: (-item[1], item[2].id))
    result: list[PanelOrder] = []
    for order_index, (depth, depth_key, panel, front, behind) in enumerate(raw):
        result.append(
            PanelOrder(
                order_index=order_index,
                panel_id=panel.id,
                view_depth=depth,
                depth_key=depth_key,
                front_facing=front,
                behind_camera=behind,
                rendered=front and not behind,
            )
        )
    return tuple(result)


def ordering_stats(
    orders: tuple[PanelOrder, ...],
    panel_lookup: dict[int, GlassPanel],
) -> dict[str, Any]:
    drawn = [order for order in orders if order.rendered]
    return {
        "rule": "quantized_view_depth_desc_then_panel_id_asc",
        "depth_quantization": f"1e-{SORT_QUANTIZATION_DECIMALS}",
        "view_depth_definition": "dot(panel_center - camera_position, camera_forward)",
        "panels": [order.stats() for order in orders],
        "drawn_panel_ids": [order.panel_id for order in drawn],
        "drawn_order_indices": [order.order_index for order in drawn],
        "drawn_panel_colors": [
            [
                quantize(component)
                for component in (
                    *panel_lookup[order.panel_id].color,
                    panel_lookup[order.panel_id].transmittance,
                )
            ]
            for order in drawn
        ],
    }

