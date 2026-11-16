"""Single source of truth for the G-buffer attachment contract.

The renderer, PNG exporter, statistics builder, and verifier all import this
module so attachment indices and channel meanings cannot drift apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GBufferAttachment:
    name: str
    location: int
    internal_format: str
    format: str
    pixel_type: str
    channels: tuple[str, ...]
    semantics: tuple[str, ...]
    encoding: str


GBUFFER_ATTACHMENTS = (
    GBufferAttachment(
        name="position",
        location=0,
        internal_format="RGBA32F",
        format="RGBA",
        pixel_type="FLOAT",
        channels=("x", "y", "z", "linear_depth"),
        semantics=("world_position_x", "world_position_y", "world_position_z", "camera_distance"),
        encoding="scene_bounds_normalized_rgb",
    ),
    GBufferAttachment(
        name="normal",
        location=1,
        internal_format="RGBA16F",
        format="RGBA",
        pixel_type="FLOAT",
        channels=("x", "y", "z", "coverage"),
        semantics=("world_normal_x", "world_normal_y", "world_normal_z", "geometry_mask"),
        encoding="signed_unit_to_unorm",
    ),
    GBufferAttachment(
        name="albedo",
        location=2,
        internal_format="RGBA8",
        format="RGBA",
        pixel_type="UNSIGNED_BYTE",
        channels=("r", "g", "b", "a"),
        semantics=("linear_albedo_r", "linear_albedo_g", "linear_albedo_b", "material_mask"),
        encoding="linear_to_srgb",
    ),
    GBufferAttachment(
        name="material",
        location=3,
        internal_format="RGBA16F",
        format="RGBA",
        pixel_type="FLOAT",
        channels=("r", "g", "b", "a"),
        semantics=("metallic", "roughness", "reserved", "material_mask"),
        encoding="linear_scalar",
    ),
)

GBUFFER_LOCATIONS = tuple(attachment.location for attachment in GBUFFER_ATTACHMENTS)
GBUFFER_NAMES = tuple(attachment.name for attachment in GBUFFER_ATTACHMENTS)


def attachment_by_name(name: str) -> GBufferAttachment:
    for attachment in GBUFFER_ATTACHMENTS:
        if attachment.name == name:
            return attachment
    raise KeyError(name)


def layout_manifest() -> dict:
    return {
        "version": 1,
        "draw_buffer_order": list(GBUFFER_NAMES),
        "attachments": [asdict(attachment) for attachment in GBUFFER_ATTACHMENTS],
        "derived_channels": {
            "metallic": {"source": "material", "component": 0},
            "roughness": {"source": "material", "component": 1},
            "depth": {"source": "position", "component": 3},
            "coverage": {"source": "normal", "component": 3},
        },
    }
