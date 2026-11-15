# Verification Results

## Environment

- WSL2 Ubuntu, GCC 15.2.0, CMake 3.30.5
- libsndfile 1.2.2 development headers
- FFTW 3.3.10 single-precision development files
- Python 3.14.4
- A native Visual Studio 2022 build was not available in this execution
  environment. The source, CMake discovery, DLL copy rules, and library calls
  are Windows/MSVC compatible; all numerical and behavioral checks below were
  executed on the same C++17 code with the Linux builds of libsndfile/FFTW.

## Commands run

```bash
cmake -S . -B build-make -DCMAKE_BUILD_TYPE=Release
cmake --build build-make -j2
./build-make/automix_tests
python3 scripts/generate_samples.py --out-dir build-make/acceptance/samples
python3 scripts/acceptance_test.py \
  --exe build-make/automix \
  --work-dir build-make/acceptance
```

## DSP and unit checks

`./build-make/automix_tests` result: `RESULT 6/6 tests passed`.

- Resampler worst passband deviation from 1 kHz through 20 kHz for a
  44.1 kHz to 48 kHz conversion: **0.000226979 dB** (limit 0.1 dB).
- Resampler alias suppression for a 30 kHz tone converted from 96 kHz to
  48 kHz: **110.76 dB** (minimum 60 dB).
- Resampler output is sample-identical across repeated conversions.
- DRC hard-knee threshold reduction is 0 dB; the 8 dB soft-knee case produces
  a reproducible steady-state reduction of 0.75 dB at the threshold.
- For 5 ms and 200 ms attack settings, the step-response gain-reduction curve
  preserves the expected sample-wise attack-time monotonicity.
- Limiter: 10 ms at 48 kHz reports **480** lookahead samples, rejects
  over-threshold samples, and reports positive attenuation.
- Log rotation with a 512-byte threshold and three retained files produced
  exactly three files; an unwritable log destination degraded to stderr
  without throwing.
- BS.1770-4 synthetic stereo 997 Hz, -20 dBFS reference:
  **-20.0000 LUFS**, **-19.9911 dBTP**.

## Generated reference

`scripts/generate_samples.py` uses only the Python standard library and writes:

- 44.1 kHz/16-bit mono WAV
- 48 kHz/24-bit stereo WAV
- 48 kHz/32-bit IEEE-float mono WAV
- 44.1 kHz/16-bit stereo WAV
- 48 kHz/24-bit stereo loudness reference
- unsupported 8-bit WAV and deliberately truncated WAV

The independent Python BS.1770-4 reference for the reference WAV is
**-19.999978 LUFS** and **-19.991147 dBTP**.

## End-to-end acceptance

`scripts/acceptance_test.py` passed:

- Full four-track render: 48 kHz, stereo, 5.0 seconds.
- Output sample peak: **-3.000000249 dBFS**, not above the -3 dBFS ceiling.
- The separately measured true peak was **-2.9886 dBTP**. The limiter guarantee
  is sample-domain (`abs(sample) <= threshold`); the 0.0114 dB difference is
  the expected inter-sample reconstruction overshoot.
- Limiter: 418 triggered input samples, 1.219 dB maximum attenuation,
  480-sample lookahead.
- Tool versus independent measurement on the final mix:
  - integrated loudness error: **1.63e-9 LU**
  - true-peak error: **3.80e-10 dB**
- Reference render versus generated reference:
  - integrated loudness error: **4.45e-10 LU**
  - true-peak error: **3.39e-9 dB**
- Repeated full renders produced byte-identical WAV files; SHA-256:
  `8827d89886120f7feb7f890164c8a63312f4787978d50bcef546ac679a6bc253`.
- Mono input was distributed identically to left and right.
- Error mappings verified:
  - `E_CFG_SYNTAX` -> exit 2
  - `E_CFG_MISSING` -> exit 2
  - `E_FILE_MISSING` -> exit 3
  - `E_FILE_FORMAT` -> exit 3
  - `E_FILE_CORRUPT` -> exit 3
- Every failing-case check confirmed that neither a WAV nor report was
  committed, while logs remained available.
- A configuration whose log path was a regular file still rendered both WAV
  and report successfully and emitted the required stderr warning.

The generated machine-readable summary is
`build-make/acceptance/acceptance_summary.json` after running the acceptance
command above.
