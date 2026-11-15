# Configuration

`automix` accepts one JSON object. Relative paths are resolved against the
directory containing the configuration file.

## Top-level fields

| Field | Required | Range / default | Meaning |
| --- | --- | --- | --- |
| `sample_rate` | no | must be `48000` | Declares the fixed internal/output rate. |
| `output` | yes | non-empty path | Final 48 kHz stereo WAV. |
| `report` | no | default `<output stem>.report.json` | JSON statistics report. |
| `log_dir` | no | default `<config dir>/logs` | Rotating log directory. |
| `log_file` | no | default `automix.log` | Plain file name, no path components. |
| `master_gain_db` | no | `-60..12`, default `0` | Gain after summing all tracks. |
| `drc` | no | DRC object | Optional DRC applied to the summed mix. |
| `limiter` | no | limiter object | Peak limiter. Missing object uses defaults. |
| `tracks` | yes | array of 1 to 8 | Input tracks. |

## Track object

| Field | Required | Range / default | Meaning |
| --- | --- | --- | --- |
| `name` | yes | unique non-empty string | Report/log identifier. |
| `input` | yes | non-empty path | PCM16, PCM24, or IEEE-float32 audio input. |
| `gain_db` | yes | `-60..12` | Track gain before DRC. |
| `mute` | no | default `false` | Mute this track. |
| `solo` | no | default `false` | If any track is soloed, only solo tracks are audible. |
| `drc` | no | DRC object | Optional per-track DRC. |

Accepted containers are WAV/WAVEX/RF64/W64, AIFF, AU, and FLAC. The encoded
sample subtype must be PCM16, PCM24, or FLOAT32. Mono and stereo are supported.
Any other subtype or channel count returns `E_FILE_FORMAT`.

Mono tracks are distributed to left and right with equal-power coefficients
`sqrt(0.5)` before track gain and DRC.

## DRC object

When `drc` is present, `enabled` defaults to `true`.

| Field | Required when enabled | Range |
| --- | --- | --- |
| `enabled` | no | boolean, default `true` |
| `threshold_db` | yes | `-40..0` dBFS |
| `ratio` | yes | `1..20` |
| `attack_ms` | yes | `5..200` ms |
| `release_ms` | yes | `5..200` ms |
| `knee_db` | yes | `0..24` dB |

`soft_knee_db` is accepted as an alias for `knee_db`. The detector is a
stereo-linked sample peak detector. Static gain reduction uses a quadratic
soft-knee curve, followed by sample-level attack/release smoothing. All track
DRC is applied before the mix; top-level DRC is applied to the mix.

## Limiter object

| Field | Required when enabled | Range / default |
| --- | --- | --- |
| `enabled` | no | boolean, default `true` |
| `threshold_dbfs` | yes | `-3..0` dBFS |
| `lookahead_ms` | yes | `0.1..50` ms |
| `release_ms` | no | `5..1000` ms, default `80` |

The limiter computes a stereo-linked future-minimum gain envelope over
`round(lookahead_ms * 48)` samples, applies instant attack plus a release
smoothed gain, and clamps the final float samples to the configured ceiling.
The report exposes both the requested lookahead in milliseconds and the exact
integer lookahead used.

## Pipeline

1. Parse and validate the complete configuration.
2. Read every track and verify the container/sample data.
3. Band-limit resample everything to 48 kHz float.
4. Convert to stereo, apply track gain, mute/solo, and track DRC.
5. Sum all tracks, apply master gain and optional master DRC.
6. Apply the lookahead limiter.
7. Measure BS.1770-4 integrated loudness and true peak.
8. Write float WAV and JSON through temporary files, then commit both.

Any input failure aborts before a WAV or report is committed. Existing logs are
left intact.
