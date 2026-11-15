# Verification Record

This file records the actual verification performed for the delivered code. The
development environment is WSL2, not a physical Windows desktop. The code uses
GLFW/OpenGL/WGL-compatible APIs and a Windows batch launcher, but the final
execution below was on Mesa llvmpipe under WSL2. A Windows run still requires the
target driver to expose an OpenGL 3.3 core context.

## Environment

```text
Python                 3.11.18
OpenGL request         3.3 core, hidden GLFW window
Actual context         4.5 (Core Profile) Mesa 26.0.8
Renderer               llvmpipe (LLVM 21.1.8, 256 bits)
Vendor                 Mesa
Packages               numpy 2.4.6, PyOpenGL 3.1.10,
                       PyOpenGL-accelerate 3.1.10, glfw 2.11.0,
                       Pillow 13.0.0, pytest 9.2.0
```

Editable package installation and console entry point were verified:

```text
uv pip install --python .venv/bin/python -e .
material-graph-renderer --help
```

## Commands

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m material_tool \
  --output-dir artifacts/verification/render_output
```

Results:

```text
pytest: 17 passed in 1.04s
material-tool exit code: 0
graphs: 2 rendered, 3 invalid, 0 failed
consistency: 1 pass, 1 intentional failure
```

Headless/no-display degradation was checked separately:

```text
env -u DISPLAY -u WAYLAND_DISPLAY .venv/bin/python -m pytest -q
11 passed, 6 skipped
```

Parser and invalid-input CLI tests still run without an OpenGL display; tests
that require actual GPU execution are skipped.

Captured output:

- `artifacts/verification/stdout.txt`
- `artifacts/verification/stderr.jsonl`
- `artifacts/verification/exit_code.txt`
- `artifacts/verification/render_output/summary.json`
- `artifacts/verification/five_channels.png`

## 512x512 Channel Results

Statistics below are recomputed by reopening the written PNG files, not from
shader-side values.

| Graph/channel | Mean | Variance | Min | Max |
|---|---:|---:|---:|---:|
| demo/base_color | 0.083508086 | 0.012186512 | 0.003921569 | 0.847058824 |
| demo/roughness | 0.273476634 | 0.050688045 | 0.000000000 | 0.458823529 |
| demo/metallic | 0.107520728 | 0.007835189 | 0.000000000 | 0.180392157 |
| demo/normal | 0.648467055 | 0.061292427 | 0.003921569 | 1.000000000 |
| demo/emissive | 0.035224336 | 0.001170574 | 0.003921569 | 0.098039216 |

The five demo PNG SHA-256 values are all distinct. Exact values and per-RGB
statistics are in `summary.json`.

The intentionally inconsistent graph reports:

```text
roughness: max abs diff 0.03999999165534973
node path: graph.outputs.roughness -> rough_by_pass

normal: max abs diff 0.3499999940395355
node path: graph.outputs.normal -> normal_by_pass

tolerance: 1e-4
```

## Errors and Exit Behavior

`stderr.jsonl` contains one JSON object per error. The default built-in run has
14 independent error records, including:

```text
GRAPH_CYCLE              invalid_cycle
EDGE_TYPE_MISMATCH       invalid_type (vec3 -> float)
EDGE_NODE_MISSING        invalid_missing_reference
CONSISTENCY_MISMATCH     inconsistent_material (roughness and normal)
```

The process exit code is 0 despite invalid graphs. Both valid graphs following
invalid graphs are still rendered.

## Resource Lifecycle

Default run resource counts:

```text
before sessions:      {"texture": 4, "vertex_array": 1}
before renderer close {"texture": 4, "vertex_array": 1}
after renderer close  {}
leak_free_after_close true
```

For both rendered graphs:

```text
hot_reload.resource_counts_equal              true
hot_reload.image_hashes_equal                 true
hot_reload.resource_counts_stable_after_render true
resources.session_leak_free                   true
```

The hot-reload test additionally verifies that the framebuffer, MRT attachment
texture IDs and UBO IDs remain unchanged, while the immutable GL program object
is replaced with a stable live-object count.

## Automated Coverage

The 17 tests include actual GL execution for:

- five-channel render + PNG reopen + exact statistics comparison
- valid multi-channel consistency
- deliberate roughness and normal mismatch with node paths and tolerance
- hot reload with equal counts and reused FBO/texture/UBO IDs
- shader compile failure with source-line diagnostics, rollback and continuation
- shader link failure rollback and continuation
- temporary framebuffer cleanup after every render
- zero tracked resources after renderer close
- CLI built-in mode and channel-subset rendering
- CLI structured stderr and exit code 0 for invalid material input

Parser-only coverage includes valid graphs, cycles, missing node references,
edge type mismatch, unknown node type, independent multi-error accumulation,
invalid JSON, GLSL generation and std140 uniform layout.

