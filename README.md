# Postaudio

`postaudio` is a deterministic, dependency-light multi-track post-production CLI.
It reads a synthetic project manifest, validates WAV headers, aligns tracks on
task-generated 2 kHz sync pulses, resamples with a rational polyphase FIR,
applies interpolated gain automation, mixes, limits, optionally normalizes
loudness, and writes PCM 16/24 or IEEE-float WAV plus a canonical JSON report.

The only third-party runtime dependency is NumPy. RIFF/WAVE parsing and writing,
resampling, marker correlation, K-weighting, loudness gating, all limiting
statistics, and report serialization are implemented in this repository.

## Windows setup

Target: Windows 10/11 x64 and Python 3.11+.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python postaudio.py --help
```

`numpy==1.26.4` is selected for Python 3.11/3.12. The validation environment
used Python 3.14, for which the requirements file selects NumPy 2.x; no audio
library or API outside NumPy's array operations is used.

## Commands

Create a deterministic synthetic project:

```powershell
python postaudio.py generate --scenario resample --out C:\work\resample
```

Validate declarations against WAV headers only:

```powershell
python postaudio.py validate C:\work\resample
```

Render and write `out\output.wav` and `out\report.json`:

```powershell
python postaudio.py render C:\work\resample
python postaudio.py render C:\work\resample --config other.json `
  --output C:\work\result24.wav --report C:\work\result24.json
```

`load` is an alias for `validate`; `export` is an alias for `render`.

Exit codes:

- `0`: success (warnings are printed to stderr but do not fail).
- `1`: fatal project/config/automation error, all tracks failed, or validation failed.
- `2`: partial render; at least one source was excluded and at least one rendered.

Available generator scenarios are listed by the `choices` in
`python postaudio.py generate --help`. They cover validation mismatches,
resampling quality, sync offsets/warnings/missing markers, automation errors,
step smoothing and clamping, hot/safe limiting, loudness normalization and
conflict, output formats, deterministic reruns, and failure recovery.

## Manifest

`manifest.json` is read as UTF-8. Source paths are relative to the project root.
All numeric automation times are seconds and gains are dB.

```json
{
  "schema_version": 1,
  "timeline_seconds": 1.0,
  "allowed_sample_rates": [44100, 48000],
  "allowed_bit_depths": [16, 24, 32],
  "output_config": "output_config.json",
  "tracks": [
    {
      "name": "dialogue",
      "file": "sources/dialogue.wav",
      "expected_sample_rate": 44100,
      "expected_bit_depth": 16,
      "expected_channels": 1,
      "sync": {
        "expected_sample_position": 441,
        "threshold_samples": 96,
        "frequency_hz": 2000,
        "cycles": 8,
        "amplitude": 0.1,
        "detection_threshold": 0.35
      },
      "gain_automation": [
        {"time": 0.0, "gain_db": -12.0},
        {"time": 1.0, "gain_db": 0.0}
      ]
    }
  ]
}
```

`expected_*` fields must match the actual header. `allowed_*` adds an optional
global declaration range. `timeline_seconds` is optional; when omitted, the
timeline is the longest aligned track.

Validation reads only the RIFF header and `fmt ` / `data` chunk metadata. It does
not read the data payload. Header mismatches produce codes such as
`SAMPLE_RATE_MISMATCH`, `BIT_DEPTH_MISMATCH`, and
`CHANNEL_COUNT_MISMATCH`.

### Sync alignment

A template matching the declared 2 kHz pulse is correlated against the mean and
each original channel **before** resampling. The reported
`offset_samples = detected_position - expected_sample_position`. A positive
offset removes the extra source lead; a negative offset prepends silence. The
offset is also converted with exact half-even rational rounding and reported as
`offset_samples_output`. Deviations beyond `threshold_samples` (default 96)
emit `ALIGNMENT_WARNING`; no pulse emits `SYNC_MARKER_NOT_FOUND`. Both continue
rendering. The pulse is pulled back onto its expected timeline position before
mixing; channel-layout conversion is linear and commutes with this shift.

### Automation

All automation nodes in the project are validated before audio payloads are
consumed. Times must be finite, non-negative, and non-decreasing; gains must be
within `[-96, +12]` dB. Failures report a zero-based `node_index` and a code
such as `AUTOMATION_TIME_ORDER` or `GAIN_OUT_OF_RANGE`.

Time-to-sample conversion is exact rational round-half-to-even. Samples between
nodes use linear interpolation; values before the first node and after the last
node hold their endpoint gain. A step edge is an adjacent pair less than 5 ms
apart with more than 6 dB difference. Such edges receive a centered raised-cosine
window, default 10 ms, clipped to the automation curve span. The report includes
`smooth_steps`, each edge, and the effective window. A requested window longer
than the span emits `SMOOTH_WINDOW_CLAMPED`.

## Output config

```json
{
  "sample_rate": 48000,
  "channels": 1,
  "encoding": "pcm24",
  "peak_limit_dbfs": -1.0,
  "limiter_lookahead_ms": 5.0,
  "smooth_window_ms": 10.0,
  "loudness_target": null,
  "resampler": {"half_width_samples": 64}
}
```

Encodings are `pcm16`, `pcm24`, `pcm32`, and `float32`. Other or unknown
fields, such as comments, do not enter audio processing or the report.

### Resampling

Source and target rates form an exact reduced integer ratio `L/M`. The
implementation precomputes one FIR phase per numerator position. Each phase is
a deterministic Blackman-Harris-windowed sinc whose cutoff is
`0.5 * min(1, target/source)`. Ratios such as 44100 to 48000 reduce exactly to
`160/147`; no irrational approximation path is entered for integer WAV rates.
Equal rates bypass with stage flags and no filter work. All sample arrays and
filters are float64.

### Layout

Target mono or stereo layouts are supported. Mono to stereo duplicates the
channel. Stereo to mono uses `(L + R) / sqrt(2)`, the -3 dB power-preserving
rule, and records `downmix_applied` plus the rule.

### Limiter

The limiter computes the minimum required gain over the current and future
look-ahead window, then applies that gain at the current sample. Thus no output
sample is shifted: the look-ahead is fully delay-compensated on the source
timeline. A signal already below the ceiling is copied byte-for-byte at the
float stage and reports zero limited samples. `limited_samples`,
`clipping_events`, before/after peaks, and `max_attenuation_db` are recorded.

### Loudness

The K-weighting cascade is derived with bilinear-transform formulas at the
output sample rate (4 dB shelf around 1681.97 Hz and RLB high-pass around
38.14 Hz). The integrated measurement applies 400 ms blocks with 75% overlap,
an absolute -70 LUFS gate, then a relative gate 10 LU below the absolute-gated
mean. If `loudness_target` is set, linear gain is applied before limiting. If
that pushes the peak over the ceiling, the signal is re-limited and
`NORMALIZATION_PEAK_CONFLICT` is emitted; final loudness is measured again and
reported. A target equal to the measured loudness produces exactly 0 dB gain
and no warning.

## Report and determinism

Reports use insertion order, six fixed decimal places for every finite float,
`null` for unmeasurable values, and two-space indentation. There are no
timestamps, random values, or filesystem-order dependencies. Tracks are
normalized to ascending track name for both report order and floating-point
mix accumulation. Therefore manifest reordering cannot change audio. Each
track records input/output rates, resampling count and ratio, channel layout,
original/output offsets, automation statistics, smoothing, and stage peaks/RMS.
Top-level sections cover mix, normalization, limiting, final loudness, warnings,
and errors.

An all-failed render writes only:

```json
{"errors": [{"track": "...", "code": "SOURCE_READ_ERROR", "detail": "..."}]}
```

Errors are sorted by track, code, and detail. Any stale output WAV is removed
before processing, so a total failure cannot leave an old render behind.

## Verification

Run the full acceptance harness:

```powershell
python -m unittest discover -s tests -v
```

The harness covers all ten requested groups end to end, including independent
CLI processes, byte hashes, 24-bit packing, FFT image levels, marker positions,
truncated sources, and report-only failure output. See
`ACCEPTANCE_RESULTS.md` for the measured results and hashes from this workspace.
