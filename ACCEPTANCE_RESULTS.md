# Acceptance Results

Run date: 2026-11-13

## Environment

- Host used for execution: Linux container supplied by the coding harness.
- Python: 3.14.4.
- NumPy: 2.5.4 (the Python 3.14 fallback selected by `requirements.txt`).
- Windows target: Python 3.11+, NumPy 1.26.4; only `pathlib`, `os`, and
  path-relative operations differ, and both the module CLI and `run_tests.cmd`
  are included. No container or nonstandard service was used.

The acceptance harness passed all ten groups:

```text
test_01_header_validation ... ok
test_02_resampling_and_fft ... ok
test_03_sync_alignment ... ok
test_04_gain_automation_and_validation ... ok
test_05_smoothing ... ok
test_06_peak_limiter ... ok
test_07_loudness ... ok
test_08_output_formats_and_downmix ... ok
test_09_determinism_and_manifest_order ... ok
test_10_partial_and_total_failure ... ok

Ran 10 tests in 9.270s
OK
```

Command:

```text
python3 -m unittest discover -s tests -v
```

## Measured Results

### 1. Header-only loading

- A manifest declaring 48000 Hz against an actual 44100 Hz file: exit `1`,
  stderr contained `declared_48k_actual_44k1` and `SAMPLE_RATE_MISMATCH`.
- Replacing the source with a 48000 Hz WAV: exit `0`; stdout reported
  `track_count=1`, matching the manifest.
- A manifest declaring stereo against a mono file: exit `1`, stderr contained
  `declared_stereo_actual_mono` and `CHANNEL_COUNT_MISMATCH`.
- Validation does not read sample payloads; truncated data is detected later by
  the render read path.

### 2. Sample-rate unification

- 44100-to-48000 track: `input_sr=44100`, `output_sr=48000`,
  `resampled=true`, `resampled_samples=48000`, exact rational ratio `160/147`.
- 48000 track: `resampled=false`.
- Output header: 48000 Hz, 48000 frames.
- 1024-point analysis of the converted 1 kHz sine: largest 1 kHz-band component;
  residual image band measured `-113.401 dB` relative to the fundamental.

### 3. Sync alignment

- Shifted source offset: `offset_samples=37`,
  `offset_samples_output=37`.
- Mixed output sync position: sample `480`, expected `480`, normalized
  correlation `1.000000`.
- Cross-rate case (48000 reference plus 44100 shifted source): original offset
  `37`, output offset `40`, mixed marker still at output sample `480`.
- 200-sample deviation: exit `0`, stderr included `ALIGNMENT_WARNING`.
- Missing pulse: exit `0`, stderr included `SYNC_MARKER_NOT_FOUND`, offset `0`.

### 4. Automation

- Reversed node time: exit `1`, stderr:
  `AUTOMATION_TIME_ORDER ... node_index=2`.
- Gain +20 dB node: exit `1`, stderr:
  `GAIN_OUT_OF_RANGE ... node_index=1`.
- The two ramp curves were sampled directly at 0.1 s and 0.9 s:
  `-10.8 dB / -1.2 dB` and `-1.2 dB / -10.8 dB`, matching the linear
  interpolation endpoints. All node conversions use half-even rational rounding.

### 5. Smoothing

- Equal-time -20 to 0 dB edge: exit `0`, `smooth_steps=1`; adjacent output
  samples around 0.5 s stayed below the 0.05-local-difference assertion.
- All-gradient curve: `smooth_steps=0`.
- Requested 200 ms on a 100 ms curve: exit `0`, stderr included
  `SMOOTH_WINDOW_CLAMPED`; effective report window was `100.000000 ms`.

### 6. Peak limiting

- +3 dBFS hot sine: output peak `-1.000000 dBFS` (ceiling -1 dBFS),
  `limited_samples=25116`, `clipping_events=25116`,
  `max_attenuation_db=4.000000`.
- The sync pulse remained at sample `480` after limiting, within one output
  sample of the original position.
- -10 dB mix: limiter bypassed, `limited_samples=0`; direct float-array
  comparison confirmed the output was byte-identical to the unlimited input.

### 7. Loudness

- BS.1770 calibration check: a 1 kHz, 0.5-peak sine measured
  `-9.024 LUFS`.
- Deterministic -20 dBFS-RMS pink-noise project, target -16 LUFS:
  measured input `-18.365003 LUFS`, normalization gain `+2.365003 dB`,
  final integrated loudness exactly `-16.000000 LUFS`, no warning.
  The gain follows the actual K-weighted source loudness; the final target
  tolerance is within the required 0.5 LU. (The task's illustrative "+4 dB"
  depends on the source's spectrum; this standard BS.1770 measurement of the
  generated pink source starts at -18.365 LUFS rather than -20 LUFS.)
- High target that crosses the ceiling: `NORMALIZATION_PEAK_CONFLICT` warning,
  re-limited peak `-1.000000 dBFS`, final loudness recomputed as
  `-6.736259 LUFS`.
- Target equal to the measured source: normalization gain `0.000000 dB`,
  no warnings.

### 8. Output formats

- 16-bit output: 48000 frames at 48000 Hz; `wave` integer values equal
  `round-half-to-even(float * 32767)` exactly (zero differing samples).
- 24-bit output: 48000 frames, file size `144044 = 44 + 48000 * 3`; packed
  little-endian samples decode within one 24-bit LSB of the float32 reference.
- Float output: 32-bit IEEE float, 48000 frames at 48000 Hz.
- Stereo-to-mono project: `downmix_applied=true`, report channels `1`.

### 9. Determinism

Two independent CLI processes produced:

```text
WAV SHA-256:
03c0eca2d014edbc56814ec417d3d5599a0984e99f30895254ca766ed98dd7c2

Report SHA-256:
a4317b6a024b5fcf3ffbce38783244c968ea0b7a84c9a9d97666c72e1b285860
```

Both hashes were identical across runs. Adding an unused config comment left
both hashes unchanged. Reversing manifest track order produced the identical
WAV hash because mixing is normalized to ascending track name.

### 10. Failure recovery

- One of three sources truncated by 64 bytes: exit `2`, output WAV exists,
  one `SOURCE_READ_ERROR` for `mid_track`, and the other two tracks rendered.
- All three truncated: exit `1`, no output WAV, report contained only the
  `errors` key.
- Two failures: errors sorted as `alpha_track`, `zeta_track` regardless of
  manifest order.

## Artifacts

The unit harness uses isolated temporary directories. The complete latest test
console transcript is in `artifacts/acceptance.log`; synthetic source and render
outputs are regenerated with `python postaudio.py generate`.
