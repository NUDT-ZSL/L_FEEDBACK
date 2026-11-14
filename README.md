# Deterministic PBR Material Baker

`bake_material.py` is an offline command-line tool for deterministic PBR
material baking, normal-space consistency checks, seam diagnostics and camera
previews. It targets Windows 10+ with Python 3.11 and uses only NumPy and
OpenCV for image processing.

## Setup

```bat
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\activate
```

The pinned versions are:

```text
numpy==1.26.4
opencv-python==4.10.0.84
```

For an air-gapped machine, populate a wheelhouse on a connected machine with
the same Python/platform, then copy it across:

```bat
py -3.11 -m pip download -r requirements.txt -d wheelhouse
.venv\Scripts\python -m pip install --no-index --find-links wheelhouse -r requirements.txt
```

No Docker, container runtime, network call at bake time or external texture
attachment is required.

## Quick Start

Generate the deterministic local fixtures:

```bash
python tools/generate_samples.py --out samples
```

This writes `samples/scene.json`, PNG base/roughness/metallic maps and a TGA
normal map. The default fixture contains a U-mirrored UV island so the
handedness correction and mirror-seam path are exercised.

Run the required entry point:

```bash
python bake_material.py --scene samples/scene.json --out out
```

Optional output controls:

```bash
python bake_material.py \
  --scene samples/scene.json \
  --out out \
  --resolution 640x480 \
  --bit-depth 16 \
  --name-pattern "asset_{kind}.png" \
  --chunk-rows 48 \
  --seed 42
```

`--resolution` controls the three camera previews. Baked texture resolution
follows the input maps. `--bit-depth`, `--name-pattern`, `--chunk-rows` and
`--seed` override the corresponding scene fields. Existing compatible
checkpoints are resumed automatically; use `--force` to discard them.

## Outputs

Default PNG names use `{kind}.png`:

| Kind | Contents |
|---|---|
| `baked_color` | Relit linear result after ACES tone mapping and sRGB encoding |
| `baked_normal` | Object-space normal encoded as `0.5 * n + 0.5` |
| `material_scalar` | RGB = roughness, metallic, 1 |
| `lighting_diffuse` | Direct diffuse component visualization |
| `lighting_specular` | Direct Cook-Torrance specular component visualization |
| `lighting_indirect` | Roughness-aware environment contribution visualization |
| `preview_main` | Camera-projected relit material |
| `preview_normal` | Camera-projected object-space normal |
| `preview_single_light` | Camera-projected direct result for the first canonical light |
| `validation_mask` | Pixel/tile anomaly localization mask; blue lines are UV boundaries |

Fixed artifacts:

| File | Contents |
|---|---|
| `asset_manifest.json` | Source hashes, dimensions, channel handling and linearization method per map |
| `validation_report.json` | Business errors or normal/seam diagnostics |
| `manifest.json` | Run fingerprint, tangent method, determinism settings and output hashes |
| `lighting_components.npy` | `H x W x 9` float32 cache: diffuse RGB, specular RGB, indirect RGB |
| `baked_linear.npy` | `H x W x 3` float32 relit result |
| `single_light_linear.npy` | `H x W x 3` float32 single-light direct result |
| `.work/state.json` | Checkpoint manifest used for interruption recovery |

## Scene JSON

Paths are resolved relative to the scene file. A complete fixture is generated
at `samples/scene.json`. The important fields are:

```json
{
  "schema_version": 1,
  "assets": {
    "base_color": {"path": "textures/basecolor.png", "color_space": "srgb"},
    "normal": {"path": "textures/normal.tga", "space": "tangent", "green_sign": 1},
    "roughness": {"path": "textures/roughness.png", "channel": 0, "min": 0.04, "max": 1.0},
    "metallic": {"path": "textures/metallic.png", "channel": 0},
    "defaults": {"roughness": 0.48, "metallic": 0.05}
  },
  "surface": {
    "geometry_normal": [0, 0, 1],
    "uv_axis_u": [1, 0, 0],
    "uv_axis_v": [0, -1, 0],
    "uv_origin": [0, 0, 0],
    "material_size": [2, 2],
    "seam_smoothing_radius": 1,
    "islands": [
      {"id": 0, "bounds": [0, 0, 128, 192], "flip_u": false, "flip_v": false},
      {"id": 1, "bounds": [128, 0, 128, 192], "flip_u": true, "flip_v": false}
    ]
  },
  "lights": [
    {"direction": [-0.38, -0.28, 0.88], "color": [1, 0.91, 0.78], "intensity": 3.6}
  ],
  "environment": {"intensity": 0.3, "color": [0.68, 0.76, 0.94]},
  "camera": {
    "projection": "perspective",
    "position": [0, 0, 3.15],
    "target": [0, 0, 0],
    "up": [0, 1, 0],
    "fov_y_degrees": 38,
    "ortho_scale": 2.25,
    "resolution": [384, 288]
  },
  "baking": {"chunk_rows": 24, "view_direction": [0, 0, 1], "exposure": 1.05},
  "validation": {
    "seed": 20261115,
    "stratified": true,
    "normal_length_tolerance": 0.035,
    "angle_threshold_deg": 34,
    "seam_angle_threshold_deg": 10,
    "reference_gain": 0.85,
    "reference_min_gradient": 0.008,
    "reference_tilt_min": 0.04,
    "seam_width": 2,
    "tile_size": 64,
    "max_entries": 512
  },
  "output": {
    "bit_depth": 8,
    "preview_resolution": [384, 288],
    "filename_pattern": "{kind}.png",
    "background": [0.018, 0.023, 0.032]
  }
}
```

`lights[].direction` points from the surface toward the light. When roughness
or metallic is absent, the matching value under `assets.defaults` is
broadcast to a single-channel map. Scalar maps may be single-channel or select
an explicit packed channel.

## Processing Notes

### Loading and linearization

- PNG is decoded with OpenCV `IMREAD_UNCHANGED`; TGA uses a built-in decoder for
  uncompressed/RLE 8/16/24/32-bit variants because not every OpenCV wheel
  ships a TGA codec. TGA origin bits are honored.
- Base color is converted with the IEC 61966-2-1 sRGB piecewise curve unless
  `"color_space": "linear"` is declared. Normals and scalar masks are decoded
  as linear data.
- Base color accepts RGB/RGBA and ignores alpha. Normal maps require exactly
  RGB. Scalar maps remain one channel.
- A decoded normal whose length differs from one by more than
  `normal_length_tolerance` is a hard `E005` error with the worst pixel.
- Every map's source hash, bit depth, channel handling, clamp range and
  linearization method is stored in `asset_manifest.json`.

### Tangent frame and object-space normals

The tangent and bitangent are derived from the scene's UV grid derivatives.
For every UV island:

1. Project the U/V axes onto the geometric tangent plane.
2. Apply U/V flip signs from the island declaration.
3. Compute handedness from
   `sign(dot(cross(N, T), dB/dv))`.
4. Rebuild the bitangent as `cross(N, T) * handedness`.

The local tangent direction is preserved while adjusting the bitangent sign,
so U-flips, V-flips and mixed mirrors are corrected without a manual
bitangent-sign switch. In the seam band and degenerately projected pole
areas, the bitangent is box-smoothed, projected back to the tangent plane,
and the tangent is recomputed without losing its local orientation.

### Shading

Direct lighting uses GGX/Trowbridge-Reitz distribution, Smith-Schlick
geometry, Schlick Fresnel, metallic F0 and Lambertian diffuse. Each scene light
contributes diffuse RGB and specular RGB. Environment injection uses a
roughness-dependent diffuse/specular split approximation.

Lights are normalized into a content-derived order before accumulation. The
finite output is therefore independent of JSON light traversal order.

### Normal consistency and seams

The validator constructs a diagnostic reference normal from a low-frequency
base-color luminance gradient plus the geometric/tangent frame. It checks:

- object normal angle against the reference;
- tangent-XY direction agreement with the albedo-gradient direction;
- X-flip, Y-flip and direct object-space interpretation hypotheses;
- UV island boundaries by paired object normals.

`validation_report.json` contains full-image ratios, severity counts,
fixed-seed stratified anomaly samples and per-sample image coordinates. Seam
entries include tile coordinates, both boundary coordinates, island IDs,
handedness change and `mirror_seam`, `hard_edge_seam`, `flipped_seam` or
`consistent` classification. The seam band has its own anomaly ratio.

The albedo-derived reference is a diagnostic low-frequency consistency cue,
not a physically derived height field. For production assets with unrelated
albedo detail, tune `reference_gain`, `reference_min_gradient` and
`reference_tilt_min`; the seam checks remain independent of that heuristic.

### Determinism and recovery

- Lights are content-sorted; chunk boundaries are pixel-local and use no
  cross-pixel accumulation.
- Validation sampling uses PCG64 with the configured fixed seed and an
  optional deterministic stratified schedule.
- PNG files are encoded by this repository with filter 0, zlib level 9 and no
  timestamps/text chunks. NPY streams use little-endian float32.
- Reports and manifests contain no wall-clock time or random runtime data.
- Row chunks are atomically checkpointed under `out/.work/chunks/`. Resume
  validates the content fingerprint and every completed chunk hash, recomputes
  nothing that is valid, and streams the same final bytes as an uninterrupted
  run.

### Error codes

| Code | Exit | Meaning |
|---|---:|---|
| `E001` | 11 | Scene or texture input missing |
| `E002` | 12 | Unsupported/un-decodable image |
| `E003` | 13 | Channel dimension mismatch |
| `E004` | 14 | Invalid scene JSON field |
| `E005` | 15 | Non-unit normal vector beyond tolerance |
| `E006` | 16 | Texture dimensions do not match |
| `E007` | 17 | Invalid/overlapping/incomplete UV island layout |
| `E008` | 18 | Unexpected I/O failure |
| `E009` | 19 | Invalid output naming/path |
| `E010` | 20 | Checkpoint belongs to a different run |
| `E011` | 21 | Invalid command-line override |

Business failures still write `out/validation_report.json` with the code,
message, asset name and pixel location when available.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite covers deterministic byte equality, 8/16-bit naming controls,
chunk-boundary invariance, interruption/resume equality, TGA RLE/origin
decoding, scalar defaults, source-light reordering, normal/mirror/hard-edge
diagnostics and error exit codes. See `docs/VERIFICATION.md` for the results
recorded in this workspace.
