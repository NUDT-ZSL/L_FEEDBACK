"""Pure Python deterministic tiled deferred renderer."""

from .errors import (
    E_SCENE_INVALID,
    E_SNAPSHOT_CORRUPT,
    E_SNAPSHOT_IO,
    E_TILE_FAILED,
    PipelineError,
    SnapshotCorruptError,
)
from .gbuffer import GBufferTile, material_shading, rasterize_tile
from .lighting import shade_tile
from .pipeline import (
    PipelineOptions,
    ReferenceResult,
    RenderPipeline,
    RenderResult,
    apply_options,
    render_reference,
    render_scene,
    write_ppm,
    write_report,
)
from .scene import Scene, TileSpec, load_scene
from .shadows import build_shadow_maps, estimate_tile_shadow_bytes, shadow_visibility
from .snapshot import (
    SNAPSHOT_FORMAT,
    SNAPSHOT_VERSION,
    Snapshot,
    SnapshotStore,
    load_snapshot,
    snapshot_filename,
    snapshot_path,
    write_snapshot,
)

render_frame = render_scene

__all__ = [
    "E_SCENE_INVALID",
    "E_SNAPSHOT_CORRUPT",
    "E_SNAPSHOT_IO",
    "E_TILE_FAILED",
    "GBufferTile",
    "PipelineError",
    "PipelineOptions",
    "ReferenceResult",
    "RenderPipeline",
    "RenderResult",
    "SNAPSHOT_FORMAT",
    "SNAPSHOT_VERSION",
    "Scene",
    "Snapshot",
    "SnapshotCorruptError",
    "SnapshotStore",
    "TileSpec",
    "apply_options",
    "build_shadow_maps",
    "estimate_tile_shadow_bytes",
    "load_scene",
    "load_snapshot",
    "material_shading",
    "rasterize_tile",
    "render_reference",
    "render_frame",
    "render_scene",
    "shadow_visibility",
    "shade_tile",
    "snapshot_filename",
    "snapshot_path",
    "write_ppm",
    "write_report",
    "write_snapshot",
]
