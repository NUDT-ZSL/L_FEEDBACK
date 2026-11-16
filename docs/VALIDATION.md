# Validation Record

Date: 2026-11-17  
Host: WSL2 Ubuntu 26.04 on x86_64  
Compiler: GCC 15.2  
CMake: 3.30.5  
OpenGL: 4.5 Core, Mesa llvmpipe (`LLVM 21.1.8, 256 bits`)  
Backend: EGL surfaceless

The conditional GLFW backend was also compile-tested with GLFW 3.4 using
`-DSKINRENDER_USE_GLFW=ON`; the actual pixel validation used the headless EGL
backend. Windows/Visual Studio execution was not available in this WSL
environment, but the Windows path uses the same GL loader, shaders, renderer,
and FBO code.

## Build and Automated Test

Command:

```bash
scripts/build_and_test.sh
```

Observed:

```text
PASS deterministic PPM output and resource reuse
PASS versioned baseline comparison statistics
PASS collision-safe PPM naming
PASS invalid-group isolation and recovery
PASS initialization failure exit code 2
PASS default 512x512 dimensions
validation: ALL PASSED
```

The same suite also passed against a separate UBSan build:

```text
cmake ... -fsanitize=undefined ...
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 python3 scripts/validate.py
validation: ALL PASSED
```

## Valid Batch and Determinism

Command:

```bash
LIBGL_ALWAYS_SOFTWARE=1 ./build/skin_renderer \
  --config configs/demo.json --output-dir output
```

Observed stdout:

```text
success=2 failed=0 exit=0
```

Two independent runs produced byte-identical files:

```text
b67478f48d0bf6b063c0ec04426271e1f915bc0818ca602b6a5dd088511cd252  skin_satin.ppm
fed5ca2e95ec97058c2a3272537eb30e9eaf42b14416c11d048f24d6a859d411  skin_rough.ppm
```

Both PPM files are `P6`, `512x512`, max value `255`, with exactly `512*512*3` data bytes.

Self-baseline comparison:

```text
[STAT] group="skin_satin" schema=1 ... pixels=262144 mse=0 mean_abs=0 max_abs=0
[STAT] group="skin_rough" schema=1 ... pixels=262144 mse=0 mean_abs=0 max_abs=0
```

Resource summary:

```text
framebuffers   created=2 released=2 balanced=yes
textures       created=2 released=2 balanced=yes
renderbuffers  created=1 released=1 balanced=yes
programs       created=2 released=2 balanced=yes
shader_objects created=4 released=4 balanced=yes
vertex_arrays  created=2 released=2 balanced=yes
buffers        created=2 released=2 balanced=yes
```

## Failure Isolation

Command:

```bash
LIBGL_ALWAYS_SOFTWARE=1 ./build/skin_renderer \
  --config configs/failure_demo.json --output-dir out
```

Observed stdout and exit:

```text
success=2 failed=2 exit=1
```

Relevant log:

```text
[ERROR] group="bad_roughness" code=111 message=roughness must be greater than 0 and at most 1
[WARN]  group="valid_duplicate" code=0 message=baseline file does not exist; comparison skipped: ...
[ERROR] group="bad_anisotropy" code=112 message=anisotropy must be within [0, 1]
[INFO]  group="-" run_complete success=2 failed=2 exit=1
```

The two valid groups after failures produced:

```text
valid_duplicate.ppm      147471 bytes
valid_duplicate_001.ppm  147471 bytes
```

Invalid groups produced no files. The collision suffix rule prevented overwrite.

## Initialization Failure

Command:

```bash
LIBGL_ALWAYS_SOFTWARE=1 ./build/skin_renderer \
  --config configs/init_failure.json --output-dir out
```

Observed:

```text
success=0 failed=0 exit=2
[ERROR] group="-" code=100 message=config initialization failed: field 'width' must be within [1, 8192]
```

No PPM was produced.

## Files Retained

- `baselines/skin_satin.ppm`, `baselines/skin_rough.ppm`: versioned sample baselines.
- `output/skin_satin.ppm`, `output/skin_rough.ppm`: final rendered samples.
- `output/render.log`, `output/diff_stats.json`: final sample logs and schema-v1 statistics.
- `configs/failure_demo.json`, `configs/init_failure.json`: reproducible negative-case configs.

