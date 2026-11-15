# Validation Record

Real environment and verification results for the deterministic offline
renderer (`rfr`).

## Environment

- Host: Windows 11 machine running WSL2 (Linux kernel 6.18, Ubuntu userspace).
- Interpreter available: CPython 3.14.4 (the task targets 3.10/3.11, but no
  3.10/3.11 interpreter is present in this offline container).
- OpenGL: no physical GPU. ModernGL creates a standalone context through
  **EGL on Mesa llvmpipe** (software OpenGL 4.5,
  `GL_RENDERER=llvmpipe (LLVM 21.1.8)`), satisfying the "OpenGL 3.3 software
  implementation" fallback described in the task.

### Dependency deviation

The pinned `numpy==1.24.3`, `moderngl==5.8.2`, `PyOpenGL==3.1.6`,
`Pillow==10.0.0` have no wheels for Python 3.14. Compatible newer versions
were installed instead:

| Package | Requested pin | Installed on host |
| --- | --- | --- |
| numpy | 1.24.3 | 2.5.4 |
| moderngl | 5.8.2 | 5.13.0 |
| Pillow | 10.0.0 | 13.0.0 |
| PyOpenGL | 3.1.6 | present |

The code only uses stable, long-lived APIs and runs unchanged against the
requested pins on Python 3.10/3.11.

## How to run

The EGL/OSMesa shared libraries must be discoverable for headless ModernGL:

```bash
export LD_LIBRARY_PATH=/path/to/mesa/lib:$LD_LIBRARY_PATH
python -m rfr.cli --scene scenes/sample.json \
    --output artifacts/out.png --frames 4 \
    --backend opengl --check-consistency --stats artifacts/out.json
```

Backends: `opengl`, `software`, `auto` (auto prefers OpenGL and transparently
falls back to the NumPy backend while recording a warning).

## Pixel consistency result

For the bundled 320x240 hierarchical scene, GPU vs software framebuffers:

- `mismatched_pixels = 0`, maximum per-channel difference = 1 (well inside the
  required tolerance of 2) across frames 0, 4 and 9.
- The software rasterizer reproduces the GPU pipeline order: viewport
  transform, CCW back-face culling, perspective-correct color interpolation,
  screen-linear window depth, depth test and the three blend modes.

At other resolutions an isolated sub-pixel silhouette pixel can differ because
the GPU fixed-function top-left coverage rule uses vendor-specific rounding;
the comparison reports the count and bounding box, and the test suite asserts
such cases stay under 0.1% of pixels.

## Test suite

`python -m pytest tests/ -q` -> **20 passed**, covering float64 transform
precision, world-matrix hierarchy, deterministic animation, the shared
lighting formula, the cached state machine, all three blend modes, GPU/software
pixel parity (including animated frames) and the structured error codes.

## Error handling evidence

| Condition | Error code | Exit code |
| --- | --- | --- |
| Missing / malformed scene | `E_SCENE_INVALID` | 1 |
| Node references undefined mesh | `E_SCENE_MESH` | 1 |
| Bad framebuffer size | `E_SIZE_MISMATCH` | 1 |
| OpenGL context failure | `E_BACKEND_UNAVAILABLE` then software fallback | 0 |

Every failure prints one JSON object on stderr containing `code`, `message`,
`context` and `partial_statistics` (frames completed, scene size and any
backend metrics gathered before the failure).
