from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    max_nodes: int = 512
    max_graph_depth: int = 64
    max_width: int = 1024
    max_height: int = 1024
    max_pixels: int = 1024 * 1024
    max_texture_dimension: int = 2048
    max_texture_bytes: int = 16 * 1024 * 1024
    max_json_bytes: int = 8 * 1024 * 1024
    max_objects: int = 64
    max_lights: int = 8

    def as_dict(self) -> dict:
        return {
            "max_nodes": self.max_nodes,
            "max_graph_depth": self.max_graph_depth,
            "max_width": self.max_width,
            "max_height": self.max_height,
            "max_pixels": self.max_pixels,
            "max_texture_dimension": self.max_texture_dimension,
            "max_texture_bytes": self.max_texture_bytes,
            "max_json_bytes": self.max_json_bytes,
            "max_objects": self.max_objects,
            "max_lights": self.max_lights,
        }


DEFAULT_LIMITS = Limits()
