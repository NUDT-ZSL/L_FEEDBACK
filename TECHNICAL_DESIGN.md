# Technical Design

## Package Layout

- `render_pipeline/scene.py`: scene parsing, tile grid, normalized material/light state, deterministic signatures.
- `render_pipeline/math3d.py`: vector math, camera basis, near/far clipping support.
- `render_pipeline/gbuffer.py`: tile-local G-buffer, software triangle rasterization, material-channel fill.
- `render_pipeline/shadows.py`: orthographic directional maps, point cube maps, PCF, bias handling.
- `render_pipeline/lighting.py`: deferred light accumulation and frame composition.
- `render_pipeline/snapshot.py`: atomic snapshot serialization/deserialization and checksum validation.
- `render_pipeline/pipeline.py`: thread-pool scheduling, reuse planning, diagnostics, PPM output.
- `render_pipeline/__main__.py`: CLI.

## Per-Tile Pipeline

Each tile is one independent job and goes through these ordered stages:

1. Geometry rasterization into a tile-local G-buffer.
2. Material shading fills the diffuse-albedo channel and validates every covered material.
3. Shadow projection builds or restores all directional/point maps owned by that tile.
4. Deferred lighting and composition produce tile-local color.
5. On success only, a snapshot is atomically persisted.

The scene object and its tuples are immutable and shared read-only. Each job owns its G-buffer lists and its shadow-map dictionaries. A restored snapshot is dedicated to one tile, so its decoded arrays are not shared with another job. Results are assembled by tile index, never by thread completion order.

## G-Buffer and Rasterization

G-buffer channels per tile:

- diffuse albedo: three float64 components per pixel;
- world-space normal: three float64 components per pixel;
- linear camera-space depth: one float64 per pixel;
- auxiliary material and primitive ids: 32-bit signed integers.

Triangle vertices are transformed into camera space, clipped against near/far planes, and projected only after clipping. Each tile rasterizes global pixel centers `(frame_x + 0.5, frame_y + 0.5)` and computes barycentrics from global screen coordinates. Depth, normals, and eventual world-position reconstruction therefore do not depend on a tile's local origin.

Perspective-correct attribute interpolation is used for perspective cameras:

```text
inverse_depth = b0/z0 + b1/z1 + b2/z2
attribute = (b0*a0/z0 + b1*a1/z1 + b2*a2/z2) / inverse_depth
```

Orthographic camera attributes use direct barycentric interpolation. Depth testing uses the interpolated linear camera depth. The whole-frame reference renderer uses the identical rasterizer with one full-frame tile, which makes cross-boundary equality testable without a separate code path.

## Shadows

### Directional

The fitted light basis uses the configured ray direction. The scene AABB is transformed into that basis and padded before normalization. Every caster is clipped to the light depth range and rasterized into one orthographic depth buffer at the effective resolution.

### Point

Six fixed cube-face bases are used in a fixed order. Geometry is clipped against the point-light near/far interval and rasterized into six perspective depth buffers. A fragment selects the face with the greatest dot product against its normalized light direction and projects into that face's stored basis.

### Filtering and Bias

PCF taps always iterate `offset_y` outer and `offset_x` inner, from negative to positive. The visibility fraction is `lit_count / kernel_area`; there is no unordered accumulation. The fast preset forces a 1x1 kernel and half resolution.

Before sampling, the receiving point is offset along its world normal. Comparison bias includes:

```text
normalized depth bias
+ configured world bias / map depth extent
+ depth epsilon
+ slope-scaled PCF footprint bias
```

The slope term accounts for adjacent PCF taps crossing a sloped receiver and prevents self-shadow banding while retaining actual cast shadows.

## Snapshot Format

Snapshot version is `1`; files are canonical JSON. Arrays are little-endian base64 `float64-le` or `int32-le` blocks with explicit counts. Conceptually:

```json
{
  "format": "render_pipeline.snapshot",
  "version": 1,
  "tile": {"index": 0, "id": "tile-0000-0000", "x": 0, "y": 0, "width": 32, "height": 24},
  "scene": {
    "scene_id": "example",
    "version": 1,
    "summary": "sha256:...",
    "base_signature": "sha256:...",
    "materials": {"floor": "sha256:..."},
    "light_colors": {"sun": "sha256:..."},
    "light_shadows": {"sun": "sha256:..."}
  },
  "render": {"quality": "high", "shadow": {"pcf_kernel": 5}},
  "gbuffer": {
    "albedo": {"dtype": "float64-le", "count": 2304, "data": "..."},
    "normal": {"dtype": "float64-le", "count": 2304, "data": "..."},
    "depth": {"dtype": "float64-le", "count": 768, "data": "..."},
    "material_ids": {"dtype": "int32-le", "count": 768, "data": "..."}
  },
  "shadows": {
    "sun": {"kind": "directional", "resolution": 256, "depths": {"dtype": "float64-le", "data": "..."}}
  },
  "lights": {"sun": {"color": [1, 0.9, 0.75], "direction": [-0.4, -1, -0.3]}},
  "dependencies": {"materials": ["floor"], "lights": ["sun"], "has_coverage": true},
  "checksum": "sha256:..."
}
```

The checksum is computed by removing `checksum`, canonicalizing with sorted keys and compact separators, then hashing the UTF-8 JSON. Loading checks the format marker, frozen version, and checksum before decoding arrays or allowing restore. Writes use a unique temporary file, `fsync`, and `os.replace` for atomic publication.

## Incremental Decision Model

The structural/base signature covers:

- scene id/version, frame size, tile size;
- camera and clip settings;
- triangle positions, normals, material assignments, material ids;
- background, ambient, light ids/types/order;
- quality and all shadow configuration.

It deliberately excludes material parameter values and light color/intensity so those can be changed incrementally. Per-material and per-light signatures are stored separately. Each tile stores the material ids and potentially contributing light ids visible in its G-buffer.

Reuse decision:

- Any base mismatch: rerender every tile, shadow cache disabled.
- No mismatch, material signature intersects tile dependency: rerender G-buffer, reuse shadow maps.
- No mismatch, color signature intersects tile light dependency: rerender lighting/G-buffer job, reuse shadow maps.
- Any light shadow signature changed: rerender coverage tiles and discard shadow caches.
- Otherwise: restore G-buffer and shadow maps, then shade/composite them deterministically.

A snapshot for a different scene version is therefore never reused, regardless of unchanged tile offsets.

## Failure and Resource Isolation

Material definitions are retained with a validation problem instead of failing scene parsing. The tile fails only when a covered pixel references that material, yielding `E_TILE_FAILED`. Memory checks happen before rasterization or shadow allocation:

```text
tile bytes = width * height * 64
           + sum(face_count * resolution^2 * 8)
```

`MemoryError` and all other per-tile exceptions are captured in that tile's result. Successful sibling tiles continue and form a partial frame over the configured background. Failed tiles never persist state; an existing snapshot at their path is removed.

## Determinism

Fixed ordering exists at every accumulating step:

- tile grid row-major, results assembled by tile index;
- lights in normalized scene order;
- pixels row-major;
- triangles in scene order;
- PCF y offsets outer, x offsets inner;
- canonical JSON with sorted keys for signatures and snapshots;
- deterministic base64 little-endian array encoding.

No random sampling, timing, wall clock, unordered set accumulation, or completion-order composition affects pixel values. The test suite verifies identical PPM hashes with 1, 4, and 8 workers and across fresh versus reused snapshot rendering.
