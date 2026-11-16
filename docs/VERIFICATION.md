# Actual Verification Record

Date: 2026-11-17

## Host

- WSL2 / Ubuntu 26.04 on x86-64
- CMake 3.30.5
- GCC 15.2.0
- Mesa 26.0.8 EGL/OpenGL software path (`LIBGL_ALWAYS_SOFTWARE=1`)
- No visible window or display-backed rendering surface used by the verifier

The Windows/GLFW branch could not be executed here because this workspace has no Windows
runtime. It was, however, cross-compiled and linked with LLVM-MinGW 18.1.7 against GLFW
3.4 source. The executed runtime verification below covers the shared renderer plus the
Linux EGL headless backend.

## Build

```sh
cmake -S . -B build-make -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
cmake --build build-make -j 4
```

Result: success, no compiler diagnostics in the final build.

### Windows Cross-Build Check

```sh
ORV_MINGW_ROOT=/tmp/orv-toolchain/llvm \
cmake -S . -B build-win2 \
  -DCMAKE_TOOLCHAIN_FILE=cmake/llvm-mingw-toolchain.cmake \
  -DOFFLINE_RENDER_GLFW_SOURCE_DIR=/tmp/orv-toolchain/glfw/glfw-3.4 \
  -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_DOCS=OFF \
  -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-win2 -j 4
```

Result: `offline_render_verify.exe` and `scene_generator.exe` were linked successfully.

```text
PE32+ executable for MS Windows 6.00 (console), x86-64
Subsystem: Windows CUI
```

The cross-build validated the hidden-GLFW/Windows conditionals and Win32 atomic file
paths. The generated `.exe` was not run because Wine/Windows is unavailable. LLVM-MinGW
upstream GLFW sources emitted their own warning set during cross-compilation; project
sources built cleanly.

## Integration Suite

Command:

```sh
cmake --build build-make --target test
```

Result:

```text
100% tests passed, 0 tests failed out of 1
Total Test time (real) = 0.90 sec
```

The single CTest entry runs nine integration groups:

```text
PASS successful_passes
PASS invalid_scenes
PASS invalid_cli
PASS framebuffer_failure
PASS zero_lights
PASS stats_write_failure
PASS recovery
PASS forced_termination_recovery
PASS signal_handling
PASS all 9 integration groups
```

Covered behavior:

- Geometry, lighting, and full PPMs have valid `P6` headers, 8-bit channels, expected
  dimensions, non-uniform content, and pairwise-different payloads.
- Anisotropy strength 0 versus 0.9 produces a different final highlight bitmap.
- Missing field, huge/out-of-range index, invalid enum, invalid tangent, and more than
  four point lights return exit 2 with the expected business code and no outputs.
- Invalid pass returns 5; invalid shadow sizes 500 and 8192 return 4 without rendering.
- Forced G-Buffer incompleteness returns 3 and leaves no PPM, stats, marker, or temp.
- Zero lights produces `light_count=0`, `shadow_method="none"`, and an all-zero PPM.
- Stats target collision returns 6 (`IO_ERR_STATS_WRITE`) and removes the marker.
- A synthetic stale marker plus two temp files is cleaned with `RECOVERY_CLEANED`.
- A real `SIGKILL` during a marker pause leaves the marker; the next run cleans it and
  continues successfully.
- A handled signal exits 130 and emits one interrupted final-state JSON object.

## Baseline Full Render

Command:

```sh
LIBGL_ALWAYS_SOFTWARE=1 ./build-make/offline_render_verify \
  --scene samples/scene.json \
  --output artifacts/final/verify \
  --pass full --width 640 --height 360 \
  --shadow-size 1024 --shadow-bias 0.002
```

Exit code: `0`

Statistics:

```json
{
  "resolution": {"width": 640, "height": 360},
  "render_time_ms": 31.120818,
  "visible_pixels": 138050,
  "average_luminance": 0.300786893,
  "maximum_luminance": 1,
  "average_luminance_clamped": false,
  "maximum_luminance_clamped": true,
  "shadow_method": "directional_2d_pcf3x3+point_cubemap_pcf3x3",
  "light_count": 4,
  "pass": "full"
}
```

The three 640x360 output files each contain 691,200 PPM payload bytes:

```text
verify_geometry.ppm  93547a899f9e1935a7f786cfe82c8ee7c0ec03ccb85f530c69685dd2543d3188
verify_lighting.ppm  d98c242523b7af63572b26a5baf81097fda8a95c7c6ec12113f46e1a2f593784
verify_full.ppm      5709b1ffddc0ad8ce1c136e2c82110acf1a8893593a2a7b5b1d08c137df49c95
```

Visual inspection confirmed the normal channel, cast shadow-factor channel, and a
tone-mapped final image with colored point-light highlights and anisotropic torus
reflections.

## Zero-Light Render

Command used the generated `samples/scene_no_lights.json` at 128x72 with shadow size 512.
The run exited 0 with:

```json
{
  "visible_pixels": 5521,
  "average_luminance": 0,
  "maximum_luminance": 0,
  "shadow_method": "none",
  "light_count": 0
}
```

Every byte in `no_lights_full.ppm` after the header was zero (`max=0`, nonzero count `0`).

## Deterministic Sample Hashes

```text
329cb16c15086d781e9f31ce73c6b68d673f16751bfa386bb684230212977fc9  samples/scene.json
fb9a1cf33bb4c880a6c329ac3cbf2832494a76868df1cf61ea6c17d26bcd875e  samples/scene_no_lights.json
```
