# Validation record

Environment used for the recorded run:

- Ubuntu 26.04.1 LTS (WSL)
- GCC 15.2.0, C++20, Release/RelWithDebInfo
- CMake 3.30.5
- Python 3.14 for test-side byte comparisons
- No GPU or third-party graphics/image library

Build:

```text
TMPDIR=$PWD/build-tmp cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
TMPDIR=$PWD/build-tmp cmake --build build -j 4
```

The workspace `/tmp` was full, so compiler temporaries were redirected to
`build-tmp`; this does not affect the produced files.

## Automated result

```text
1/3 Test #1: baker_unit ....................... Passed
2/3 Test #2: baker_integration ................ Passed
3/3 Test #3: baker_kill_recovery .............. Passed
100% tests passed, 0 tests failed out of 3
```

The integration test verifies:

- two generated scenes from the same seed are byte-identical;
- two async bakes are byte-identical and report identical diagnostics;
- the async result is byte-identical to `--sync-io`;
- a capacity-1 queue with injected I/O delay still produces the same bytes;
- completed snapshots resume at layer 6;
- hash, resolution, layer-count, snapshot-version, render-time, and corruption
  cases are discarded with distinct reasons;
- `ERR_MAGIC=2`, `ERR_VERSION=3`, `ERR_LENGTH=4`, and `ERR_VALUE=5` are returned
  before rendering and an existing output sentinel is untouched.

The forced-kill test verifies:

- SIGKILL during final output write leaves the previous target byte-identical;
- rerun from the persisted snapshot reproduces the sequential baseline;
- SIGKILL during a partial snapshot leaves the previous output intact, and the
  rerun reports `snapshot=hit resume_from=<persisted layer>`;
- SIGTERM during output write returns nonzero, keeps the previous output, and
  removes the temporary file.

An additional Debug build with AddressSanitizer and UndefinedBehaviorSanitizer
passed all three tests (`3/3`, `0` failures), covering the malformed-input and
forced-kill paths under runtime memory/UB checks.

## Sample deterministic run

Generated with seed `22`, `160x120`, six layers, and a `64x64` RGBA noise map:

```text
[baker] hash=0fb414e3d72e6493 res=160x120 layers=6
[baker] snapshot=discard reason=missing
[baker] degenerate_layers=1
[baker] layer_pixels=19200,1317,1714,5077,1420,0
[baker] output=work/manual1/out1.rgba

[baker] hash=0fb414e3d72e6493 res=160x120 layers=6
[baker] snapshot=hit resume_from=6
[baker] degenerate_layers=1
[baker] layer_pixels=19200,1317,1714,5077,1420,0
[baker] output=work/manual1/out2.rgba
```

`cmp` reported no differences between `out1.rgba`, `out2.rgba`, and the
`--sync-io` reference. The generated file is `76816` bytes: a 16-byte `QRBA`
header plus `160 * 120 * 4` RGBA bytes.

## Format interoperability check

An existing `QSNP v1` fixture was copied to a scratch directory and loaded by
this implementation. It resumed at layer 6, reported the historical layer
statistics, and its path-produced `QRBA` bytes matched the corresponding
reference file byte-for-byte. The supplied `QSNP v2` round-trip is covered by
the unit test.

As a non-required compatibility probe, a fresh render of the bundled legacy
scene differed from its old reference image in only 10 of 76,800 bytes at
`t=0` and 12 bytes at `t=0.35`; those residual bytes are 8-bit truncation
boundaries caused by different floating-point contraction between compilers.
Determinism tests do not depend on matching that external implementation.
