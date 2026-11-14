# Terrain Heightfield Baker

Offline tool that turns a deterministic procedural heightfield into:

- a tangent-space normal map (sRGB-marked RGB8 PNG),
- multi-level LOD vertex/index buffers (explicit little-endian binary + JSON),
- offscreen diffuse/normal preview renders,
- a loadable, checksummed `metadata.json` manifest.

All processing is tile/band streamed against a memory-mapped heightfield so
that large terrains (up to 4097x4097) never materialize the full grid in RAM.

## Requirements

- Python 3.10+ (developed and verified on Python 3.12).
- `numpy`, `Pillow`, `PyOpenGL` (`pip install -r requirements.txt`).
- A software GL backend for previews: OSMesa (`osmesa32.dll`) or Mesa
  EGL/llvmpipe. If neither can create a context the core artifacts are still
  produced and the process returns the dedicated exit code `6`.

### Windows notes

- All paths use `os.path` and forward-slash relative keys inside JSON; the
  staging directory lives next to the target so `os.replace` stays on one
  volume and is atomic.
- `sys.exit` only ever receives small non-negative integers, matching Windows
  32-bit process exit semantics.
- Files are little-endian explicitly (`<f4`, `<u4`) so buffers load the same
  on Windows and elsewhere.
- Prefer an OSMesa/llvmpipe Mesa build; set `LIBGL_ALWAYS_SOFTWARE=1` when
  using a Mesa driver package.

## Usage

```bash
python -m terrain_baker \
  --seed 2026 --resolution 1025 --height-scale 8 --frequency 2 \
  --levels 0 1 2 --tile 512 \
  --memory-limit-mb 64 \
  --output out/terrain
```

`--skip-preview` disables GL rendering. Run `python -m terrain_baker -h` for
all options.

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 2 | Invalid parameter |
| 3 | Baking failure (incl. memory budget) |
| 4 | Filesystem / atomic-commit failure |
| 6 | Offscreen backend unavailable (core artifacts still complete) |

## Output layout

```
out/terrain/
  data/heightfield.dat        # raw <f4 memmap, resolution*resolution
  textures/normal.png         # sRGB RGB8 tangent-space normals
  meshes/lod0.bin ...         # vertex block then index block per LOD
  preview/diffuse.png         # optional
  preview/normal.png          # optional
  metadata.json
```

### Binary mesh format

Each `lodN.bin` is one little-endian file:

- vertex block at `vertex_offset_bytes` (0), `vertex_count` vertices of
  stride 32 bytes: `position[3]<f4`, `normal[3]<f4`, `uv[2]<f4`;
- index block at `index_offset_bytes`, `index_count` `uint32` triangle-list
  indices, counter-clockwise winding when viewed from +Z.

All counts, strides and offsets are repeated in `metadata.json`, which also
carries per-file and overall SHA-256 hashes, bounding boxes and tile timings.

## How requirements are met

1. **Deterministic heightfield** — self-contained splitmix64-hashed value
   noise and fBm with no global PRNG state (`terrain_baker/noise.py`);
   identical parameters produce byte-identical float32 memmap output. Odd
   resolutions from 257 to 4097 are validated.
2. **Normal map** — central differences with a one-texel clamped halo; edge
   pixels are flagged and boundary samples are clamped. RGB8 encoded, with an
   explicit PNG `sRGB`/`gAMA` marker. After decode+normalize, vector length
   error is ~1e-7 (well under 1e-3).
3. **LOD meshes** — each level applies a separable tent low-pass filter
   before 2x decimation to suppress aliasing, keeps shared corners exact, and
   streams vertices/indices in row bands with bounding boxes and UVs.
4. **Geometric consistency** — the normal map and mesh normals use the same
   operator and the LOD0 world spacing; per-LOD mesh normals are recomputed at
   that level's spacing and compared against bilinear, edge-clamped normal-map
   samples at matching UVs. Max tangent-space angle stays under 5 degrees.
5. **Tiling and memory** — above 1025 the bake runs fixed 512x512 normal tiles
   and row-band mesh passes; a `MemoryBudget` guard enforces the 64 MiB ceiling
   on live working arrays (memmap page cache is not counted as resident work
   data). Tiles carry a clamped halo so results equal the monolithic result
   exactly and no seams appear; per-tile timings are recorded.
6. **Atomic output / cleanup** — everything is staged in a same-volume
   `*.staging` directory and moved with `os.replace`; a prior manifest drives
   removal of stale files. Any failure aborts and removes staging, leaving no
   partial set in the target.
7. **Preview** — ctypes-backed OSMesa then EGL/llvmpipe contexts render a
   shaded and a normal-visualization PNG in an offscreen framebuffer; failure
   raises a typed error, writes diagnostics into metadata and returns code 6.
8. **Metadata** — `metadata.json` records inputs, per-LOD stats, relative
   paths, tile timings, memory usage, quality metrics and SHA-256 hashes.
   `terrain_baker/verifier.py` independently re-parses buffers and hashes.

## Tests and verification

```bash
python -m unittest discover -s tests        # 38 tests
LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe python acceptance.py
```

`acceptance.py` writes `artifacts/acceptance_report.txt`. Recorded local
results (Mesa llvmpipe, Python 3.12):

- 257 and 1025 bakes: all LOD angle checks pass (worst ~2.9 deg),
  normalized normal length error 1.19e-7.
- Tiled vs monolithic normal map: byte-identical, no border seam.
- 4097 full bake: peak live memory 32 MiB (< 64 MiB), completed in ~14 s,
  LOD0 = 16,785,409 vertices / 33,554,432 triangles.
- Determinism: two independent processes produce identical heightfield,
  PNG and binary SHA-256 hashes.
- Exit codes verified: bad params -> 2, unavailable GL backend -> 6 with
  intact core artifacts, write failure -> clean target and removed staging.

## Package layout

```
terrain_baker/
  noise.py        deterministic value noise / fBm
  heightfield.py  validated spec + memmap generation
  normal.py       tiled central-difference normal map, RGB8 codec
  lod.py          low-pass decimation + streaming binary meshes
  consistency.py  map-vs-mesh angular checks
  memory.py       live-array memory budget guard
  atomicio.py     same-volume staging and atomic commit
  pngutil.py      deterministic sRGB PNG writer
  preview.py      OSMesa / EGL software offscreen renderer
  metadata.py     manifest assembly
  verifier.py     independent buffer/hash verification
  baker.py        end-to-end orchestration
  cli.py          command line and exit codes
tests/            unit and integration tests
acceptance.py     end-to-end acceptance driver
```
