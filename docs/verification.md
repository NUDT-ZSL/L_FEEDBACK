# Verification Record

Machine-readable aggregate: `artifacts/validation/summary.json`.

## Environment

- WSL2 Ubuntu 26.04, x86_64, 24 logical cores.
- Portable CMake 4.0.1, Ninja, Zig 0.13.0 C++ frontend (Clang 18.1.6).
  The final Linux test executable was a static `x86_64-linux-musl` build.
- Python 3.14.4 for process-level integration tests.
- No Docker, graphics library, or network service was used.
- The Windows source path was cross-compiled to
  `PE32+ executable for MS Windows 6.00 (console), x86-64, 11 sections`.
  The PE was not executed on a Windows host in this environment; MSVC 2019+
  uses the same C++17 sources and portable filesystem APIs.

## Reproduced commands

The local machine did not ship a system C++ compiler or CMake. Validation used
the existing portable tools:

```bash
export TMPDIR=$PWD/.work-tmp
export ZIG_GLOBAL_CACHE_DIR=$PWD/.work-zig-cache/global
export ZIG_LOCAL_CACHE_DIR=$PWD/.work-zig-cache/local
cmake -S . -B build-final -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$HOME/.cache/skinbake-tools/zig-cxx \
  -DCMAKE_MAKE_PROGRAM=/tmp/lod-sysroot/root/usr/bin/ninja \
  -DSKINBAKE_WARNINGS_AS_ERRORS=ON
cmake --build build-final
ctest --test-dir build-final --output-on-failure
```

In a normal Windows VS environment only the commands in `README.md` are needed.
The local wrapper executable invokes `zig c++ -target x86_64-linux-musl`, so the
result is independent of the host glibc development package.

## Actual test output

```text
Start 1: skinbake_unit
1/2 Test #1: skinbake_unit ....................   Passed    0.18 sec
Start 2: skinbake_integration
2/2 Test #2: skinbake_integration .............   Passed    2.45 sec

100% tests passed, 0 tests failed out of 2  (2.63 sec total)
```

The unit executable reports:

```text
PASS all skinbake unit checks
```

The process-level suite reports each phase:

```text
[1/8] serial/parallel byte identity and from-zero verification
[2/8] stable business errors and no partial bake output
[3/8] snapshot version and checksum rejection
[4/8] forced termination, atomic progress and resume
[5/8] version/fingerprint invalidation discards old progress
[6/8] transition blend and state-machine timeline
[7/8] exact time rounding
[8/8] integration checks passed
```

A separate debug build with `-fsanitize=undefined -fno-omit-frame-pointer` and
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1` ran both the unit executable
and the process-level suite successfully with no sanitizer diagnostics.

## Determinism and precision

512-frame serial and 40-worker (bounded to 24 logical cores) bakes produced
identical files:

```text
af0d587f6473a6bf00bb98b6f7a6e07d40790f94bd4bf01e3bc258a8a4823de5
```

The same hash applied to `serial.snapshot` and `parallel.snapshot`. The from-zero
verifier reported:

```json
{"ok":true,"mode":"verify","passed":true,"frames":512,
 "tolerance":1.0000000000000001e-05,"maxAbsError":0,
 "maxErrorFrame":0,"byteIdentical":true,"mismatchFrames":[]}
```

The full report is in
`artifacts/validation/parallel/parallel-evidence.json`: requested workers 40,
actual workers 24, logical cores 24, frames 512, `byte_identical=true`.

The 10000-frame timeline check reported:

```json
{"ok":true,"mode":"timeline-check","passed":true,"frames":10000,
 "fps":60,"maxDeviationMs":0,"finalPlaybackTimeMs":650,
 "finalPropagationTimeMs":166650}
```

## Snapshot rejection examples

Version mismatch (`stored=9`, requested=3):

```json
{"ok":false,"code":"SNAPSHOT_VERSION_MISMATCH","codeNumber":4003,
 "message":"snapshot version mismatch: actual=9 expected=3",
 "actualVersion":"9","expectedVersion":"3"}
```

Checksum mismatch:

```json
{"ok":false,"code":"SNAPSHOT_CHECKSUM_MISMATCH","codeNumber":4004,
 "message":"snapshot checksum mismatch: stored=0x6315317d computed=0xe8ab8997",
 "storedChecksum":"1662333309","computedChecksum":"3903555991"}
```

Both commands exited with code 4. A subsequent load of the untouched valid file
succeeded, demonstrating that rejected reads did not partially apply or damage
state.

## Stable input error codes

Each scenario was executed twice; stderr bytes were identical:

```text
CURVE_TIME_NOT_INCREASING    1001
BEZIER_DENOMINATOR_ZERO      1002
WEIGHT_SUM_ZERO              2005
WEIGHT_SUM_NOT_ONE           2006
SKELETON_CYCLE               2002
CURVE_EMPTY                  1003
```

For the Bezier sample, the deterministic details are:

```text
dx control deltas: 0.000000, 200.000000, 100.000000
```

An invalid bake was also run against a fresh output path. No final snapshot and
no `.progress` file were created, and the process exited with code 2.

## Forced termination and resume

The integration suite uses the documented `_Exit(137)` crash hook after an atomically
committed frame. In this run, frame 21 was the last commit:

```text
progress inspect: frames=22, complete=false
resume report: resumed=true, resumeFrame=22, newFramesCommitted=158
event log: commit 0..179 exactly once, resume frame 22
```

The resumed final snapshot was byte-identical to an uninterrupted 180-frame
reference at the point of comparison. A separate evidence run is recorded in
`artifacts/validation/resume-evidence.json`.

Version change (3 → 4) reset progress with reason `version`; changing one curve
value with `--variant` and keeping version 4 reset progress with reason
`fingerprint`. Both re-baked from frame 0 and passed from-zero verification.

## Blend continuity

The transition scenario switches at frame 12 with a 200 ms window. Unit checks
confirmed:

- switch frame blend weight is exactly 0 and equals the frozen old pose;
- frame 13 weight is `17/200` using rounded 60 fps times;
- a vertex at frame 20 equals the direct linear interpolation between the old
  snapshot and new snapshot pose;
- the mutable state machine and pure parallel timeline agree within `1e-12`.

The 48-frame transition snapshot passed from-zero verification with
`maxAbsError=0`, `byteIdentical=true`.

