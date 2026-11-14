# Deterministic CPU Volumetric Cloud Prototype

A self-contained C++20/CMake prototype for a game weather system. It generates a
128^3 R8 Perlin-Worley cloud density field, ray-marches it on the CPU, renders a
1280x720 RGBA8 frame, accumulates history through camera reprojection, and
stores/restores the complete renderer state in CRC-protected versioned snapshots.

No GPU API, image library, noise library, JSON library, Docker, network access, or
external asset is used. The executable consumes a JSON file and writes raw RGBA8,
PPM, depth, count, snapshot, statistics, and JSON-line log files.

## Build

### Windows 10/11, Visual Studio 2022 / MSVC

Open "x64 Native Tools Command Prompt for VS 2022":

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
build\Release\cloud_render.exe --config configs\full_hd_quality.json
```

The Windows build uses `/fp:strict`, C++20, and no graphics dependency. The
working directory must be writable because snapshots and logs are written there.

### Linux / GCC or Clang

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
python3 scripts/validate.py --build-dir build --workspace "$PWD"
```

The `validate` CMake target runs the same end-to-end harness when Python 3 is
available:

```bash
cmake --build build --target validate
```

## Quick Start

```bash
./build/cloud_render --config configs/full_hd_quality.json
```

This renders one 1280x720 frame at half resolution, depth-aware upsamples it,
measures PSNR against a direct full-resolution pass, writes the final snapshot,
and writes:

- `frame_0000.rgba` and `frame.rgba`: RGBA8, 1280x720.
- `frame_0000.ppm`: dependency-free preview image.
- `frame_0000.depth.f32`, `frame.depth.f32`: ray depth.
- `frame_0000.counts.u32`, `frame.counts.u32`: per-pixel accumulation counts.
- `snapshot.cld`: v2 renderer snapshot with CRC32.
- `run_stats.json`: final counters and PSNR values.
- `render.log`: one JSON object per line with events and business codes.

`configs/temporal_demo.json` renders five orbiting-camera frames with temporal
accumulation. `configs/stratus_low_memory.json` demonstrates stratus density and
the 32^3 low-memory path.

## Configuration

All runtime parameters are injected through JSON. Important fields:

```json
{
  "outputDir": "artifacts/full_hd_quality",
  "width": 1280,
  "height": 720,
  "startFrame": 0,
  "endFrame": 1,
  "seed": 1337,
  "cloudType": "cumulus",
  "renderScale": 2,
  "maxSteps": 64,
  "lightSteps": 5,
  "maxHistoryLength": 8,
  "async": true,
  "threads": 4,
  "memory": { "availableMB": 512 },
  "snapshot": {
    "version": 2,
    "loadPath": "",
    "savePath": "artifacts/full_hd_quality/snapshot.cld",
    "saveEveryFrames": 1,
    "stopAfterSave": false
  },
  "quality": { "compareAgainstFullResolution": true }
}
```

`endFrame` is exclusive. If `snapshot.loadPath` is set, its frame number is
authoritative and rendering resumes at `snapshot.frameIndex + 1`.

The camera can be static, orbit deterministically with the frame index, or be
overridden with explicit row-major `viewMatrix` and `projectionMatrix` entries in
`cameraFrames`. The sun can also rotate deterministically per frame. No wall
clock, process ID, uninitialized memory, or unordered floating-point reduction
participates in rendered output.

## Memory Levels

The density footprint presets are:

| Available memory | Density |
|---|---|
| >= 128 MiB | 128^3 |
| >= 64 MiB | 64^3 |
| >= 32 MiB | 32^3 |
| < 32 MiB | `E_OUT_OF_MEMORY`, exit 13 |

The generator always constructs the deterministic 128^3 field, then performs
successive 2x2x2 integer-averaging downsamples to 64^3 or 32^3. Logs contain
`W_DENSITY_DOWNSAMPLED`, the selected level, the available memory, and the reason.
Snapshots retain the selected density level so continuation uses the identical
field.

## Snapshot and Error Codes

The snapshot stores seed, cloud shape, density level, actual frame number,
previous camera matrices, accumulation color, depth, per-pixel count, reprojection
history length, and next-frame lighting. Files use little-endian fields and a
standard IEEE CRC32 over the payload. v2 adds sun direction/color, ambient light,
and history length over v1. Loading v1 fills the in-memory v2 fields with
documented defaults and emits `W_SNAPSHOT_UPGRADED`.

See [docs/SNAPSHOT_FORMAT.md](docs/SNAPSHOT_FORMAT.md) for the byte-level field
tables. Parsing selects the complete table for the detected version; v2 reads
never reuse v1 data offsets.

| Code | Exit | Meaning |
|---|---:|---|
| `E_SNAPSHOT_MISSING` | 10 | Load path does not exist or cannot be opened |
| `E_SNAPSHOT_CRC` | 11 | Payload CRC32 mismatch |
| `E_VERSION_UNSUPPORTED` | 12 | Snapshot version is not 1 or 2 |
| `E_OUT_OF_MEMORY` | 13 | Memory cannot fit even 32^3, or snapshot level no longer fits |
| `E_SNAPSHOT_FORMAT` | 14 | Invalid magic, dimensions, layout, or field data |

The process logs each error as one JSON object, destroys the worker pool, joins
all threads, releases buffers, and returns the matching nonzero exit code.

## Determinism and Validation

- Density uses integer hashes and fixed-order float evaluation. Slice tasks write
  disjoint voxels, so async and sync generation are byte-identical.
- Rendering and temporal accumulation are single-threaded and deterministic. The
  main thread waits on `std::condition_variable`; it never spins.
- A full snapshot serializes exact IEEE-754 float bit patterns for accumulation
  and depth, so a restored run can be byte-identical to uninterrupted rendering.
- Per-pixel counts always increase by one per serial frame. Hits plus misses
  equals the ray count each frame, and the hit rate is logged.

Actual results from this workspace (see `artifacts/validation/VALIDATION.md` and
`artifacts/validation/report.json`):

- 1280x720 target, half-resolution depth-aware upsample: **70.38 dB** PSNR.
- 640x360 acceptance run: **60.98 dB** PSNR.
- v2 snapshot interruption/resume: **0.0 maximum relative error**, byte-identical.
- v1 upgrade warning and default-field continuation: passed.
- 64^3 and 32^3 low-memory continuation: byte-identical at each level.
- Async worker density versus sync density: byte-identical output.
- Error exit codes 10/11/12/13: observed and validated.
- Worker cleanup event after each successful run: observed.

The checked-in summary is reproduced by `scripts/validate.py`; generated artifacts
are under `artifacts/`.

## Source Layout

- `src/density.*`: deterministic Perlin-Worley field and staged downsampling.
- `src/renderer.*`: raymarch scattering, half-resolution upsample, temporal
  reprojection, PSNR.
- `src/snapshot.*`: v1/v2 layouts, CRC32, serialization and upgrade defaults.
- `src/thread_pool.*`: worker pool and condition-variable event wait.
- `src/config.*`: JSON configuration and deterministic camera/sun sequences.
- `tests/cloud_tests.cpp`: density, snapshot, PSNR, and continuation tests.
- `scripts/validate.py`: CLI-level acceptance and business-error harness.

