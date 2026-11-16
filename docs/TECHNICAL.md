# Technical design

## Shared evaluation path

`AssetReader` produces an immutable-in-practice `Asset`. For a requested
frame, `evaluateFrame`:

1. samples every node track by clamped linear interpolation; quaternion
   samples use normalized spherical interpolation;
2. composes each local matrix according to the asset's declared T/R/S order;
3. evaluates parent-to-child world matrices in topological order;
4. computes `skin = world * inverseBind`;
5. normalizes each vertex's active weights when their sum differs from one by
   more than `1e-6`;
6. blends positions with the skin matrix and normals with the inverse
   transpose of each skin matrix's upper-left 3x3 block.

The result is a `BakedFrame` with positions, normals, colors, indices, skin
matrices and per-frame statistics. The OpenGL path converts those exact arrays
to an interleaved VBO. The offline path rasterizes those exact arrays. This is
the single source of truth for real-time/offline consistency.

`composeLocalTransform` normalizes every quaternion and rejects lengths below
`1e-8`. It does not mutate the output matrix on failure. Numeric tests compare
all six composition orders against an independently assembled matrix chain
with a `1e-5` tolerance.

## Weight policy

The binding topology is never modified. For every vertex with up to four
active weights:

- all active weights must be finite and non-negative;
- a non-positive sum rejects the asset;
- if `abs(sum - 1) > 1e-6`, all active weights are divided by the sum;
- inactive slots must contain zero;
- each normalized vertex increments the frame and report counter.

No weight is clamped, discarded or fabricated.

## Degenerate transforms and NaN reporting

Positions and normals are counted independently. A singular skin matrix can
leave positions finite while its inverse transpose produces non-finite
normals. The offline path still writes the requested PFM/PPM and JSON report,
then exits with code `70` if any frame has a non-zero position or normal NaN
count. This makes NaN behavior visible to a pipeline rather than silently
producing corrupt frames.

## Offline rendering

The CPU rasterizer uses:

- the same `lookAtRH` view and `perspectiveRH_ZO` projection as OpenGL;
- near-plane clipping in homogeneous space;
- camera-facing independent triangle rasterization (no back-face culling);
- pixel-center sampling and a depth buffer in `[0,1]`;
- perspective-correct interpolation for shading attributes;
- a fixed directional light and fixed background.

PFM output uses `PF\n<width> <height>\n-1.0\n`, little-endian floats and
bottom-to-top rows. PPM output is binary `P6` with maximum value 255. Frame
numbers use at least four digits, for example `depth_0003.pfm`.

Image files contain no timestamps or paths. With the same executable, input
asset, frame and camera parameters, repeated runs are byte-identical on the
same CPU/platform.

## JSON reports

The report schema is `skelpreview.statistics.v1`. It includes:

- total rendered frames, vertex count and triangle count;
- the number of vertices whose weights were normalized;
- per-frame position NaN and normal NaN counts;
- output file names, camera parameters and image dimensions;
- total and per-frame elapsed milliseconds.

All timing values are isolated under `timing` and per-frame `elapsed_ms`.
Deterministic comparisons strip only those fields. The generated JSON is
parsed before and after writing; parse failure maps to `INVALID_JSON` (29).

## Preview and fallback

The preview requests an OpenGL 4.1 core context, checks the actual context
version, loads only the required extension entry points through GLFW, and
shares `evaluateFrame` with offline mode. Controls:

- left drag: orbit;
- mouse wheel: zoom;
- `W`: filled/wireframe;
- `Space`: pause/resume animation;
- `R`: reset camera;
- `Esc`: exit.

Any failure to initialize GLFW, create a 4.1 window, load entry points or
compile shaders prints a diagnostic and invokes the deterministic offline
renderer. Offline mode itself has no GPU dependency.

