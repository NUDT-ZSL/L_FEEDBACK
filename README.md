# Deterministic character-appearance software baker

This project renders layered animated quads entirely on the CPU and writes a
deterministic 32-bit RGBA result. It has no GPU, image-library, or asset
dependency: `scene_gen` creates both the scene binary and the procedural RGBA
noise texture from a fixed seed.

## Build

Linux or another CMake/POSIX host:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Windows 10/11, Visual Studio 2022:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --parallel
```

`clang-cl` can be selected with `-T ClangCL`. The code is C++20 and uses only
the standard library plus Win32/POSIX file APIs.

## Commands

```text
scene_gen --out SCENE [--width N] [--height N] [--layers N]
          [--tex-size N] [--seed N]

baker --input SCENE --output OUT.rgba --snapshot SNAP
      [--time T] [--sync-io]
```

`--sync-io` is the single-threaded reference path. Normal operation puts each
snapshot and final output job on a bounded queue serviced by one writer thread.

Useful environment variables:

- `BAKER_QUEUE_CAPACITY=N`: bounded job queue size. Without an override the
  capacity is `max(8, min(layer_count + 1, 64))`, so normal scenes can queue
  all layer snapshots plus the final output without waiting on the writer.
- `BAKER_IO_DELAY_MS=N`: deliberately slow every atomic file write.
- `BAKER_SYNC_IO=1`: same behavior as `--sync-io`.

## Verification scripts

```bash
./scripts/run_twice.sh
./scripts/kill_during_write.sh
```

PowerShell helpers for the two-run and forced-kill checks are
`scripts/run_twice.ps1` and `scripts/kill_during_write.ps1`.

The CTest suite includes focused unit tests, full integration checks, and
forced-kill recovery tests:

```bash
ctest --test-dir build --output-on-failure
```

## Rendering contract

- Scene data is read strictly little-endian and validated before any renderer
  object is created. Failures print `ERR_MAGIC`, `ERR_VERSION`, `ERR_LENGTH`,
  or `ERR_VALUE` to stderr and exit with codes `2`, `3`, `4`, or `5`.
- Playback mode `clamp` clamps time to the first/last key. Mode `loop` wraps
  over the interval `[first,last)` with `fmod`, so time `last` evaluates at
  the first key. A single key is constant in both modes.
- Projection has a 90-degree vertical field of view. Pixel centers are fixed
  at `(x + 0.5, y + 0.5)`. UVs are perspective-correct and repeat outside
  `[0,1]`; bilinear sampling uses texel centers.
- Degenerate quads are skipped and counted once. A fixed screen-area tolerance
  of `1e-7` and barycentric tolerance of `1e-9` are used.
- Lighting is ambient `0.08`, diffuse coefficient `0.92`, normalized light
  direction `(0.44721359, 0.71554178, -0.53665632)`, and Blinn-Phong
  shininess `clamp(2 / roughness^2 - 2, 2, 2048)`. Specular is additive and
  each scanline triangle uses its own face normal, including non-planar quads.
- Non-premultiplied samples are converted by multiplying lit RGB by the sample
  alpha before composition. Premultiplied layer RGB is used directly, as
  required by the input contract. Source-over composition is performed in
  premultiplied float RGBA and clamped after every layer.
- Final conversion is deterministic truncation after `clamp(value, 0, 1) *
  255`, in R, G, B, A order. The output header is `QRBA`, version `1`, width,
  height, followed by premultiplied 8-bit RGBA rows.

See `docs/FORMAT.md` for exact byte layouts and `docs/VALIDATION.md` for the
recorded Linux validation run.

## Recovery and atomicity

Every completed layer atomically replaces the snapshot with a versioned record
containing the scene hash, resolution, layer count, completion index, time,
diagnostics, and premultiplied float state. A rerun accepts the snapshot only
when all render-affecting parameters match; otherwise the snapshot is discarded
and rendering begins at layer zero.

Final output and snapshot files are written to unique same-directory temporary
files, flushed, then atomically renamed. A failed or forcibly killed write can
leave only an unreferenced temporary file; the previous target remains valid.
Graceful `SIGINT`/`SIGTERM` handling removes the active temporary file.
