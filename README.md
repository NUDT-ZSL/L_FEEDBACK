# rfr — Deterministic Offline Renderer

A small, fully local renderer that loads a hierarchical JSON scene, evaluates
parent/child transforms, runs vertex lighting, generates and executes draw
calls, and writes an RGBA PNG plus JSON statistics. Two interchangeable
backends produce matching pixels:

- **OpenGL backend** built on ModernGL (headless EGL/OSMesa, no window needed).
- **Software backend** built only on NumPy (vectorized triangle rasterization).

## Features

- Scene graph with TRS local transforms; all hierarchy/camera math is float64,
  world matrices are cast to float32 only at render time (`rfr/scene.py`,
  `rfr/mathutils.py`).
- Deterministic per-frame animation (spin and orbit) with repeatable results.
- Shared Blinn-Phong vertex lighting for directional and point lights, with
  base color, roughness and metallic materials (`rfr/lighting.py`). The GLSL
  shader mirrors the NumPy formula exactly.
- Cached render-state machine tracking program, blend mode, depth test and
  culling; a switch is counted only when a value changes (`rfr/pipeline.py`).
- Opaque, alpha and additive blending executed in depth-first node order, with
  the software compositor matching the GL blend equations and depth-test/blend
  ordering.
- Pixel consistency check with configurable per-channel tolerance, reporting
  mismatch count and coordinate bounding box (`rfr/compare.py`).
- Structured business errors (`E_SCENE_INVALID`, `E_SCENE_MESH`,
  `E_BACKEND_UNAVAILABLE`, `E_SHADER_COMPILE`, `E_SIZE_MISMATCH`, ...) on
  stderr with non-zero exit codes and preserved partial statistics.

## Layout

```
rfr/
  mathutils.py      float64 matrices, look-at, perspective, normal matrix
  scene.py          JSON loading, hierarchy, animation
  lighting.py       shared vertex lighting (single source of truth)
  pipeline.py       DrawCall and cached StateCache
  renderer.py       depth-first draw list, frame driver, statistics
  compare.py        framebuffer pixel comparison
  factory.py        backend selection with software fallback
  sample_scene.py   deterministic procedural scene (no external assets)
  cli.py            command line entry point
  backends/
    gl_backend.py   ModernGL shaders, VAO/VBO, blend/depth/cull state
    software.py     NumPy rasterizer with equivalent state machine
tests/              pytest suite (20 tests)
scenes/sample.json  generated deterministic sample scene
```

## Usage

```bash
pip install -r requirements.txt
python -m rfr.cli --scene scenes/sample.json \
    --output artifacts/out.png --frames 4 --backend opengl \
    --check-consistency --stats artifacts/out.json
```

Key flags: `--backend {opengl,software,auto}`, `--frames N`,
`--check-consistency`, `--tolerance 2`, `--stats path.json`. On a headless
host export the Mesa library path first, e.g.
`export LD_LIBRARY_PATH=/path/to/mesa/lib:$LD_LIBRARY_PATH`. See
`VALIDATION.md` for the recorded environment and verification results.
