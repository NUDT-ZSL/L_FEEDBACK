# Deterministic Tiled Deferred Renderer

This workspace contains a dependency-free Python implementation of an offline tiled deferred renderer. It rasterizes scene geometry into per-tile diffuse-albedo, world-normal, and linear camera-depth G-buffers, builds directional and point-light shadow maps, shades and composites the frame, and persists validated tile snapshots for incremental recovery.

## Requirements

- Python 3.10 or newer
- No third-party graphics, imaging, or numerical libraries

## Quick Start

```bash
python -m render_pipeline --scene sample_scene.json --output out.ppm
```

Defaults:

- Diagnostic report: `out.report.json`
- Snapshot directory: `out.snapshots/`
- Incremental restore: enabled when valid snapshots exist
- Thread pool: 4 workers

The sample scene can be regenerated deterministically with:

```bash
python samples/generate_sample_scene.py
```

## CLI

```text
python -m render_pipeline --scene SCENE --output OUT.ppm
    [--report REPORT.json]
    [--snapshot-dir DIRECTORY]
    [--incremental] [--no-snapshot] [--no-incremental]
    [--quality high|fast]
    [--tile-width N] [--tile-height N]
    [--workers N]
    [--pcf-kernel N]
    [--shadow-resolution N]
    [--directional-resolution N] [--point-resolution N]
    [--memory-limit-bytes N] [--memory-limit-mb N]
```

The process exits with status `0` when a frame, including a partial frame, was produced. Tile failures are represented in the report with status `partial`. Invalid scene input exits with status `2` and prints an `E_SCENE_INVALID` JSON error.

## Scene Format

The parser accepts either lists or objects for materials and lights. Inputs are normalized once; objects are traversed in sorted-id order so declaration ordering cannot affect floating-point accumulation.

```json
{
  "scene_id": "example",
  "version": 1,
  "frame": {"width": 320, "height": 240},
  "tile": {"width": 32, "height": 24},
  "camera": {
    "position": [3.0, 2.0, 5.0],
    "target": [0.0, 0.5, 0.0],
    "up": [0.0, 1.0, 0.0],
    "fov_y_degrees": 55.0,
    "near": 0.05,
    "far": 50.0
  },
  "background": [0.01, 0.01, 0.02],
  "ambient": 0.05,
  "materials": [
    {"id": "floor", "albedo": [0.7, 0.7, 0.7], "roughness": 0.8}
  ],
  "triangles": [
    {
      "positions": [[-2, 0, 2], [2, 0, 2], [2, 0, -2]],
      "normals": [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
      "material": "floor"
    }
  ],
  "lights": [
    {
      "id": "sun",
      "type": "directional",
      "direction": [-0.4, -1.0, -0.3],
      "color": [1.0, 0.9, 0.75],
      "intensity": 1.1,
      "resolution": 256
    },
    {
      "id": "lamp",
      "type": "point",
      "position": [1.0, 2.0, 0.0],
      "color": [0.4, 0.7, 1.0],
      "intensity": 2.0,
      "range": 6.0,
      "resolution": 128
    }
  ],
  "quality": "high",
  "shadow": {
    "pcf_kernel": 5,
    "bias": 0.002,
    "normal_bias": 0.01
  },
  "memory_limit_bytes": 67108864
}
```

Mesh input may also use `vertices` plus optional `indices` and per-vertex `normals`. A top-level `tile_size`, `resolution`, `frame_size`, `shadow_resolution`, and the documented aliases are accepted.
The memory limit may also be specified as `memory_limit_mb`.

Conventions:

- `direction` is the direction in which directional light rays travel. Diffuse lighting uses the vector toward the light, `-direction`.
- Depth in the G-buffer is linear camera-space depth along the camera forward axis.
- PPM output stores clamped linear radiance without gamma conversion.
- Camera near/far clipping happens before projection; per-tile rasterization always uses global pixel centers.

## Shadow Quality

| Mode | Directional/point map | PCF kernel |
| --- | --- | --- |
| `high` | configured full resolution | configured odd kernel, default 5x5 |
| `fast` | exactly half resolution | single sample |

Directional lights use an orthographic map fitted to the scene bounds. Point lights use six deterministic cube faces. Surface normal offset, world-space depth bias, depth epsilon, and slope-scaled PCF footprint bias are applied before comparison.

## Snapshots and Incremental Rendering

Every successful, freshly rendered tile writes an atomic JSON snapshot named `tile-XXXX-YYYY.snapshot.json`. A failed tile writes nothing and removes any stale snapshot at its path. Snapshot files contain:

- format marker, version, tile id and grid bounds;
- scene id, scene version, full scene summary, structural/base signature, and mutable material/light signatures;
- encoded G-buffer albedo, normal, depth, material id, and primitive id channels;
- encoded directional depth map or six point-light faces for every light;
- light state and per-tile material/light dependencies;
- SHA-256 checksum over canonical JSON content.

`load_snapshot` rejects a wrong version, format marker, malformed content, or checksum mismatch with `E_SNAPSHOT_CORRUPT`. The pipeline records such a load failure in `snapshot_errors`, refuses that snapshot, and rerenders the tile.

Incremental rules:

- Geometry, camera, tile layout, scene version, quality, shadow configuration, or other structural changes invalidate all snapshots.
- A material parameter change rerenders only tiles whose stored material dependency contains that material.
- A directional/point light color or intensity change rerenders only tiles whose stored light dependency contains that light. Their shadow maps remain reusable.
- A light transform or shadow parameter change invalidates shadow caches and rerenders tiles with coverage.
- An untouched tile reuses both its G-buffer and shadow maps and is only re-composited. A dirty material tile may reuse its shadow maps.
- A quality change forces a full rerender and never reuses old-resolution shadow maps.

## Business Errors

- `E_SCENE_INVALID`: malformed scene or camera configuration.
- `E_TILE_FAILED`: tile-local material validation, allocation/memory-limit failure, or another captured tile exception.
- `E_SNAPSHOT_CORRUPT`: unsupported snapshot version, format mismatch, malformed content, or checksum mismatch.
- `E_SNAPSHOT_IO`: snapshot read/write failure.

## Tests

```bash
python -m unittest discover -v tests
```

The suite covers exact tiled/reference G-buffer equality across tile boundaries, thread-count-invariant output hashes, directional and point shadow quality presets, PCF variation, snapshot round trips and corruption/version rejection, material/light/version incremental behavior, quality-change cache invalidation, invalid-material partial frames, memory-limit failures, failed-tile snapshot removal, and the CLI entry point.

See `TECHNICAL_DESIGN.md` for implementation details and `artifacts/verification.json` for the recorded local verification run.
