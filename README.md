# Material Graph Renderer

Windows-local deterministic material graph compiler and OpenGL 3.3 offscreen
renderer. It reads JSON material graphs, validates structure and types, generates
GLSL vertex/fragment shaders, renders the five semantic channels through an MRT
framebuffer, writes PNG files, and emits a summary JSON with pixel-derived
statistics, consistency checks, resource counts and structured errors.

No external textures, images, containers or online services are used. Built-in
textures, sample graphs and tests are generated deterministically in code.

## Requirements

- Windows 10/11 with an OpenGL 3.3-capable driver (integrated graphics is enough)
- Python 3.11 or newer
- No Docker or other container runtime

PowerShell setup:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[test]"
```

The equivalent package list is in `requirements.txt`:
`numpy`, `PyOpenGL`, `PyOpenGL-accelerate`, `glfw`, `Pillow`.

## Run

Use all built-in deterministic samples:

```powershell
.\.venv\Scripts\python -m material_tool --output-dir .\render_output
```

Render an explicit material directory and channel subset:

```powershell
.\.venv\Scripts\python -m material_tool .\materials `
  --output-dir .\render_output `
  --channels base_color,roughness,metallic,normal,emissive
```

Other options:

```text
--size 512                 square framebuffer size, default 512
--tolerance 1e-4           roughness/metallic/normal cross-pass tolerance
--no-reload-check          disable the per-graph hot-reload probe
```

No positional material directory means the five built-in samples are used.
Structural material errors are written to stderr as one JSON object per line and
do not change the process exit code from 0. A valid graph after an invalid graph
is still processed. Command-line usage errors use the normal argparse exit code 2.

## Output Layout

```text
render_output/
  summary.json
  demo_material/
    material.json
    base_color.png
    roughness.png
    metallic.png
    normal.png
    emissive.png
    shaders/
      material.vert
      material.frag
  inconsistent_material/
    ...
```

`summary.json` records:

- per-channel image path, dimensions and statistics: mean, variance, min, max,
  per-RGB-channel statistics, pixel count and SHA-256
- consistency conclusion, tolerance, compared channels, maximum absolute
  difference and node paths for every mismatch
- hot-reload resource counts before/after reload and before/after re-rendering
- resource counts around each material session and leak checks
- structured parser, shader and renderer errors
- process-level baseline and post-close resource counts

Image statistics are not taken from shader internals. A channel is rendered,
read back with `glReadPixels`, written to PNG, re-opened with Pillow, decoded to
pixels, and only then summarized.

## JSON Schema

Top-level graph:

```json
{
  "schema_version": 1,
  "id": "demo_material",
  "camera": {
    "position": [0.0, 0.15, 3.15],
    "target": [0.0, 0.0, 0.0],
    "up": [0.0, 1.0, 0.0],
    "fov_degrees": 42.0
  },
  "light": {
    "direction": [-0.45, -0.65, -0.60],
    "color": [1.0, 0.96, 0.90],
    "intensity": 1.0
  },
  "parameters": {
    "rough_base": {"type": "float", "value": 0.38},
    "base_tint": {"type": "vec3", "value": [0.82, 0.28, 0.13]}
  },
  "nodes": [],
  "edges": [],
  "outputs": {
    "base_color": "base_color.out",
    "roughness": "roughness.out",
    "metallic": "metal_value.out",
    "normal": "normal_mapped.out",
    "emissive": "emissive.out"
  }
}
```

Edges use `from` and `to` references of the form `node.port`. The parser also
accepts `{"node": "id", "port": "out"}` reference objects. Output bindings are
strictly typed:

| Channel | Type |
|---|---|
| `base_color` | `vec3` |
| `roughness` | `float` |
| `metallic` | `float` |
| `normal` | `vec3` |
| `emissive` | `vec3` |

### Parameters

Parameters may use the shorthand `"name": 0.4` for a float or
`"name": [0.1, 0.2, 0.3]` for a vector. Explicit form
`{"type": "float|vec2|vec3|vec4", "value": ...}` is preferred.

All material parameters are emitted into the explicit `MaterialParameters`
std140 uniform block. The CPU-side offsets and sizes follow std140 alignment.
Samplers cannot live in a uniform block, so built-in textures use separate
`sampler2D` uniforms. Camera and light data use the `SceneParameters` uniform
block.

### Nodes

Supported node types:

- Constants: `constant_float`, `constant_vec2`, `constant_vec3`,
  `constant_vec4`, `constant_color`
- Inputs: `parameter`, `uv`, `world_position`, `world_normal`
- Texture: `texture` with built-in `checker`, `noise`, `normal_tile`, `uv_grid`
- Math: `add`, `sub`, `mul`, `div`, `pow`, `min`, `max`, `dot`, `cross`,
  `normalize`, `length`, `abs`, `sin`, `cos`, `tan`, `fract`, `floor`, `ceil`,
  `round`, `sqrt`, `exp`, `log`, `negate`, `clamp`, `mix`, `smoothstep`, `step`
- Vector construction/selection: `combine`, `swizzle` (`x/y/z/w` components)
- Mapping: `normal_map`, `quantize`
- Diagnostic per-pass value: `channel_switch`

`channel_switch` brings a node value in from a specific render channel. It is
useful for testing consistency detection because a graph that should share one
physical parameter across passes can deliberately diverge. The built-in
`inconsistent_material` sample uses it for roughness and normal. A normal
production graph should use one shared parameter/node for all passes.

## Rendering and Validation

The renderer creates a hidden GLFW window, requests an OpenGL 3.3 core context,
and performs all drawing into a 512x512 (or `--size`) framebuffer. Geometry is a
ray-traced sphere drawn by a full-screen triangle, so results do not depend on a
mesh asset. Camera and light direction are read from each graph.

The framebuffer has four color attachments:

- attachment 0: selected semantic channel, `RGBA8`, saved as PNG
- attachment 1: roughness, `RGBA32F`
- attachment 2: metallic, `RGBA32F`
- attachment 3: material normal, `RGBA32F`

Every requested channel pass is rendered separately. The float attachments are
compared pixel by pixel across passes. If roughness, metallic or normal differs
by more than the configured tolerance, the summary contains a difference entry
with the parameter, channel pair, maximum absolute difference, first differing
pixel and the output node path. The tolerance itself is always written to the
summary.

## GPU Resources and Hot Reload

`ResourceLedger` records each actual OpenGL object ID at allocation and removes
it at deletion. Counts are derived only from this ledger, not from estimates.

Per renderer:

- built-in textures and the full-screen VAO persist for the renderer lifetime
- a per-render temporary framebuffer and texture are allocated for channel
  readback and always deleted in `finally`

Per material session:

- persistent MRT framebuffer and its attachments
- material/scene UBOs
- shader program

A hot reload reparses changed JSON, generates new GLSL, compiles and links a new
program while the old program remains valid. Only after a successful link is the
old program deleted and the new one installed. The framebuffer, its textures and
UBOs keep their original GL IDs. Because OpenGL program objects are immutable,
the program binary must be replaced, but the count of live program objects is
stable. The JSON probe changes graph metadata, recompiles, re-renders all
requested channels, compares image hashes, and verifies equal resource counts.

If compilation, linking or rendering fails, temporary objects and the failing
session are released, the failure is recorded, and the next material graph is
processed. The process does not abort.

## Structured Error Codes

Representative codes:

- `GRAPH_JSON_INVALID`, `GRAPH_ROOT_NOT_OBJECT`, `GRAPH_ID_MISSING`
- `NODE_ID_DUPLICATE`, `NODE_TYPE_UNKNOWN`, `NODE_PARAM_INVALID`
- `EDGE_NODE_MISSING`, `EDGE_PORT_UNKNOWN`, `EDGE_TYPE_MISMATCH`
- `GRAPH_CYCLE`, `NODE_INPUT_MISSING`, `OUTPUT_MISSING`,
  `OUTPUT_TYPE_MISMATCH`
- `PARAMETER_MISSING`, `PARAMETER_INVALID_TYPE`, `TEXTURE_UNKNOWN`
- `GL_SHADER_COMPILE_ERROR`, `GL_SHADER_LINK_ERROR`, `GL_RENDER_ERROR`
- `CONSISTENCY_MISMATCH`, `HOT_RELOAD_PARSE_ERROR`, `RESOURCE_LEAK`

Shader compile/link errors include the saved source paths, raw compiler log,
source line numbers, offending source lines and parsed diagnostics. A
`shaders/compile.log` is also written when a raw log exists.

## Tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

The suite covers:

- valid graph parsing and GLSL generation for all five channels
- cycle detection with path
- edge type mismatch (`vec3` normal into a scalar input)
- missing node references and independent multi-error accumulation
- deterministic PNG/statistics consistency after PNG re-open
- valid consistency, and deliberate roughness/normal cross-pass mismatch with
  node paths
- hot reload with identical resource counts and persistent FBO/texture/UBO IDs
- shader compile failure rollback followed by successful processing of the next
  material graph
- CLI built-in mode, channel subset, structured stderr and exit code 0

The OpenGL tests use a hidden window and offscreen framebuffer. They do not need
a visible display or external assets. If no OpenGL 3.3 context can be created,
those tests are skipped; parser tests still run. Running with `DISPLAY` and
`WAYLAND_DISPLAY` removed was also verified (`11 passed, 6 skipped`).

## Verified Locally

Environment used for verification:

- Linux/WSL2 development shell (the implementation itself is Windows-compatible)
- Python 3.11.18
- Mesa 26.0.8 llvmpipe, OpenGL 4.5 core context (the requested minimum is 3.3)
- `pytest`: 17 passed

The default 512x512 built-in run produces five PNGs for `demo_material`, five
for `inconsistent_material`, structured errors for three invalid samples, a
passing consistency result for the valid graph, and failing roughness/normal
consistency entries for the deliberately inconsistent graph. `summary.json`
records stable hot-reload counts and empty resource counts after renderer close.
Detailed measured values and captured artifacts are recorded in `VERIFICATION.md`.
