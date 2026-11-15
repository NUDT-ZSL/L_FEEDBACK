# Dialogue Baker

Deterministic offline dialogue baking for narrative-game cutscenes. The tool
reads dry WAV performances, a frame-based subtitle script, and camera/character
motion tracks, then produces:

- `dialogue.wav`: 48 kHz, 16-bit, stereo PCM with RIFF metadata.
- `subtitles_frames.json`: one exact integer-sample interval and active dialogue
  id list per project frame.
- `bake_report.json`: stable JSON with spatialization, overlap, clipping,
  loudness, timebase, snapshot, and format-validation records.

No GPU, cloud account, container, or external asset is required.

## Requirements

- Python 3.10+
- `python -m pip install -r requirements.txt`
- `ffmpeg` and `ffprobe` available in `PATH`

NumPy is the DSP implementation dependency. `pydub` is listed because it is part
of the requested runtime environment; the pipeline uses FFmpeg directly for
deterministic decode/resample and FFprobe for final format validation.

## Quick Start

Generate the local deterministic five-line sample (including two overlapping
speakers, a 44.1 kHz source, and a 48 kHz stereo source):

```bash
python tools/generate_demo.py --out demo_project
python -m dialogue_baker bake \
  --project demo_project/project.json \
  --out demo_output
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

The sample uses `target_lufs: -10.0` deliberately. It is a hot stress fixture
that forces pre-limiter clipping and exercises the true-peak limiter. Normal
dialogue work should use an appropriate target such as `-23.0`.

## Input Schema

`project.json` references a script, a scene, and an audio root. Paths are
relative to the project file.

```json
{
  "schema_version": 1,
  "fps": "24",
  "script": "script.json",
  "scene": "scene.json",
  "audio_root": "voices",
  "sample_rate": 48000,
  "target_lufs": -23.0,
  "true_peak_dbtp": -1.0,
  "distance_reference_m": 1.5,
  "pan_strength": 0.9,
  "fade_ms": 5.0,
  "speed_of_sound_mps": 343.0,
  "limiter": {
    "lookahead_ms": 5.0,
    "release_ms": 80.0,
    "true_peak_oversample": 4
  }
}
```

`script.json`:

```json
{
  "schema_version": 1,
  "fps": "24",
  "duration_frames": 120,
  "dialogues": [
    {
      "id": "d01",
      "speaker": "alice",
      "start_frame": 0,
      "end_frame": 48,
      "text": "Line text",
      "audio": "alice.wav"
    }
  ]
}
```

`scene.json` requires a camera and one dense keyframe track per referenced
speaker. Every frame in `0 .. duration_frames-1` must have exactly one
keyframe:

```json
{
  "schema_version": 1,
  "fps": "24",
  "duration_frames": 120,
  "camera": {
    "position": [0.0, 1.5, -6.0],
    "forward": [0.0, 0.0, 1.0],
    "up": [0.0, 1.0, 0.0]
  },
  "tracks": {
    "alice": {
      "keyframes": [
        {
          "frame": 0,
          "position": [-2.0, 1.5, 3.0],
          "forward": [0.0, 0.0, -1.0]
        }
      ]
    }
  }
}
```

Camera `target` may be used instead of `forward`. Half-open dialogue intervals
are `[start_frame, end_frame)`. Different speakers may overlap; two intervals
for the same speaker may not.

## Timebase Contract

All timing decisions use integer samples at 48 kHz:

- Frame boundary to sample: `ceil(frame * 48000 / fps)`.
- Sample boundary to frame: `floor(sample * fps / 48000)`.
- Rational rates such as `30000/1001` are represented as fractions, not floats.
- Frame and dialogue intervals are half-open.
- A line is active in frame `f` exactly when its sample interval intersects
  `[frame_to_sample(f), frame_to_sample(f+1))`.

The ceiling mapping guarantees `sample_to_frame(frame_to_sample(f)) == f` for
all supported boundaries, including 23.976 and 29.97 fps. No floating-point
frame boundary is used for subtitle activity.

Source files are decoded/resampled to mono at 48 kHz by FFmpeg. The resampled
stream is corrected to the exact expected sample count, then zero-padded or
trimmed to the script's integer sample span. Thus source resampling contributes
zero final timeline-length error; the raw FFmpeg length delta is also recorded.

## DSP Model

Each dialogue is split into one-frame gain segments:

- Distance gain: `reference_distance / max(distance, reference_distance)`.
- Speaker orientation directivity:
  `0.65 + 0.35 * max(0, dot(forward, to_camera))`.
- Equal-power left/right pan from projection onto the camera right vector.
- Gain knots are placed at every frame boundary and linearly interpolated
  sample by sample. No dialogue gets a single whole-line gain.
- Half-cosine fades at line boundaries prevent edge steps.
- Per-frame radial velocity is integrated into a variable-rate source read.
  The phase is normalized to the exact script sample span, so Doppler-style
  local pitch movement cannot shift the line start/end or subtitle alignment.

For every line the snapshot retains a per-frame audit table containing frame,
distance, distance gain, pan, Doppler factor, and left/right gains.

Overlapping lines are summed in float64, then loudness gain is applied. A
lookahead/release limiter and a final true-peak safety trim prevent PCM
overflow. The report records both per-line contribution peaks and clipping
counts before and after gain, plus final post-limiter counts.

## Loudness

Integrated loudness uses ITU-R BS.1770-style K-weighting and gating:

- 400 ms blocks, 75% overlap, 100 ms hop.
- 48 kHz K-weighting coefficients.
- Absolute gate `-70 LUFS`.
- Relative gate `-10 LU`.

The exact coefficients, block size, hop, overlap, and gates are written to
`bake_report.json`.

True peak is estimated with a normalized 4x polyphase windowed-sinc
oversampler. The limiter uses 5 ms lookahead and 80 ms release by default. A
`0.05 dB` safety margin is reserved before PCM quantization; the encoded WAV is
read back and re-measured.

If the requested target loudness conflicts with the true-peak ceiling, the
limiter wins and the report exposes both target error and limiter reduction.

## Five Stages and Recovery

Stages run in this order:

1. `parse`
2. `spatialize`
3. `mix`
4. `loudness`
5. `package`

Each completed stage writes an atomic snapshot under
`OUT/.bake_state/<stage>/`. The manifest contains the stage version, input
digest, file hashes, and snapshot digest. `manifest.json` is written last, so a
forced process termination leaves an incomplete stage that is ignored.

Before reuse, the next stage recomputes the expected digest from:

- the project/file input digest,
- the previous stage's snapshot digest,
- the current stage version.

Version mismatch, file hash mismatch, or input mismatch rejects that snapshot
and re-executes the stage. Downstream digests then naturally invalidate later
stages.

Useful commands:

```bash
# Simulate termination after a completed stage.
python -m dialogue_baker bake \
  --project demo_project/project.json \
  --out demo_output \
  --stop-after spatialize

# Resume from the snapshots.
python -m dialogue_baker bake \
  --project demo_project/project.json \
  --out demo_output

# Ignore all snapshots and rebuild.
python -m dialogue_baker bake \
  --project demo_project/project.json \
  --out demo_output \
  --force
```

## Output Metadata

The WAV writer emits a standard PCM `fmt ` and `data` chunk, a `LIST/INFO`
comment, and a custom `bakr` metadata chunk. Both metadata payloads record:

- rational and decimal frame rate,
- 48 kHz sample rate,
- target loudness and true-peak ceiling,
- duration in frames and samples,
- input digest and stage versions.

Report and subtitle JSON use `sort_keys=True` and fixed separators. Audio
samples, report bytes, and subtitle bytes are reproducible for identical
inputs.

## Error Codes

Errors are returned as stable JSON on stderr and in `OUT/bake_error.json`:

| Code | Meaning |
| --- | --- |
| `E_MISSING_INPUT` | Project, script, scene, or WAV file is missing |
| `E_INVALID_JSON` | Input JSON cannot be decoded |
| `E_INVALID_SCHEMA` | Required field/vector/keyframe shape is invalid |
| `E_INVALID_FPS` | Missing, non-numeric, or non-positive frame rate |
| `E_FPS_MISMATCH` | Project, script, and scene frame rates disagree |
| `E_INVALID_FRAME_RANGE` | Empty, negative, or otherwise invalid interval |
| `E_FRAME_OUT_OF_RANGE` | Dialogue or keyframe lies outside the timeline |
| `E_SAME_SPEAKER_OVERLAP` | One speaker has overlapping dialogue |
| `E_MISSING_TRACK` | A dialogue speaker has no keyframe track |
| `E_TRACK_RANGE` | Track does not contain every required frame |
| `E_UNSUPPORTED_SAMPLE_RATE` | Output/sample-processing rate is not 48 kHz |
| `E_AUDIO_DECODE` | FFmpeg cannot decode a dry WAV |
| `E_FFMPEG_MISSING` | `ffmpeg` or `ffprobe` is absent from `PATH` |
| `E_OUTPUT_FORMAT` | Encoded WAV does not match 48 kHz/16-bit/stereo PCM |
| `E_TRUE_PEAK_AFTER_ENCODE` | Encoded WAV exceeds the configured ceiling |
| `E_OUTPUT_CLIPPING` | Float output exceeded full scale before quantization |

See `VALIDATION.md` for the local measurements and hashes produced by the
delivered build.
