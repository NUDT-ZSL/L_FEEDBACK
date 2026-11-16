# Dialogue Timeline Localizer

Local, deterministic command-line tool for replacing cut-scene dialogue with
localized audio while keeping the original video timeline. It reads a JSON
manifest, decodes the selected language's dialogue files with local `ffmpeg`,
trims their edge silence, normalizes each segment to `-23 LUFS` with a
`-1 dBFS` peak ceiling, maps the clips onto the video duration, and writes:

- a `48 kHz`, stereo, 16-bit PCM WAV;
- a structured JSON report with actual times, loudness, peaks and flags.

No network access, accounts, Docker or container runtime are used.

## Requirements

- Windows 10/11 x64, Python 3.10+
- `pip install numpy soundfile`
- `ffmpeg` and `ffprobe` on `PATH`

The implementation uses only NumPy for sample processing. `ffmpeg` handles
decoding, sample-rate conversion to 48 kHz, and channel alignment. `ffprobe`
reads source metadata before decoding.

## Quick Start

Windows PowerShell:

```powershell
py -m pip install -r requirements.txt
py tools\make_demo_assets.py --out artifacts\demo
py localize_dialogue.py `
  --manifest artifacts\demo\manifest.json `
  --target-language zh `
  --output artifacts\demo\output_zh.wav `
  --report artifacts\demo\report_zh.json
py tools\acceptance_check.py `
  --output artifacts\demo\output_zh.wav `
  --report artifacts\demo\report_zh.json `
  --manifest artifacts\demo\manifest.json
```

The package entry point is equivalent:

```text
python -m dialogue_localizer --manifest ... --target-language zh --output ... --report ...
```

Optional arguments:

- `--offset-ms N`: shift the whole target timeline by `N` milliseconds.
- `--crossfade-ms N`: equal-power crossfade length, `0..10` ms; default `10`.
- `--target-lufs -23`: segment loudness target; default `-23`.
- `--peak-dbfs -1`: segment peak ceiling; default `-1`.
- `--ffmpeg PATH`, `--ffprobe PATH`: explicit executable paths.

## Manifest Schema

Paths are resolved relative to the manifest file.

```json
{
  "schema_version": 1,
  "video": {
    "path": "video_720p30.mp4",
    "duration": 10.0,
    "fps": 30,
    "width": 1280,
    "height": 720
  },
  "dialogue_track": {
    "path": "source_dialogue_48k_stereo.wav",
    "sample_rate": 48000,
    "channels": 2,
    "validate_targets": true
  },
  "segments": [
    {
      "id": "scene01_line01",
      "start": 0.5,
      "end": 2.0,
      "text": {
        "zh": "欢迎来到本地化测试场景。",
        "en": "Welcome to the localization test scene."
      },
      "targets": {
        "zh": {
          "path": "targets/zh/segment_01.wav",
          "start": 0.5,
          "end": 2.0,
          "sample_rate": 44100,
          "channels": 1
        },
        "en": {
          "path": "targets/en/segment_01.wav",
          "start": 0.6,
          "end": 2.1
        }
      }
    }
  ]
}
```

Required semantics:

- `video.duration` is the fixed video length in seconds.
- `dialogue_track.sample_rate` and `channels` describe the original dialogue
  track. The source file is probed and rejected on mismatch.
- Segment `start`/`end` are source semantic boundaries. They must be finite,
  non-negative, ordered, and within the video duration.
- The selected language must exist under `targets`.
- `target`, `start` and `end` are optional per target and default to the source
  segment. Target times define the intended start and allocated slot.
- Target `sample_rate`/`channels`, when present, declare the input file format
  and are checked before decode. If omitted, they inherit `dialogue_track`.
  This is how a mismatch is rejected while still allowing explicitly declared
  44.1 kHz or mono targets to be resampled/aligned to 48 kHz stereo.
- Set `dialogue_track.validate_targets` to `false` to allow undeclared target
  input formats; declared per-target formats are always checked.

Times are converted to the nearest 48 kHz sample. The output report uses the
resulting actual sample-derived times so it can be compared against the WAV
without floating-point timeline drift.

## Timeline Rules

1. `target.start + offset` is the intended clip start. A negative resulting
   start is rejected with `OFFSET_OUT_OF_RANGE`.
2. The localized clip's true duration is measured after edge-silence trim.
3. If the clip is longer than its target slot (`target.end - target.start`),
   all following segments are shifted by the overflow. Shifts accumulate.
4. If an overlap exceeds the configured crossfade, the later segment is shifted
   right until the overlap is exactly at most 10 ms. This preserves all target
   audio instead of dropping its head.
5. Any mapped or shifted audio beyond the video length is truncated. The
   segment is marked `truncated: true` and receives a warning.

Overlaps at the final 10 ms boundary use `sin`/`cos` equal-power gains:

```text
fade_out^2 + fade_in^2 = 1
```

## Loudness and Peak

Each segment is measured with gated ITU-R BS.1770 integrated loudness at
48 kHz, normalized to `-23 LUFS`, and limited to `-1 dBFS`. The BS.1770
`400 ms` blocks use `100 ms` hops. Segments shorter than 400 ms use one complete
short block so tiny edited lines still receive a stable measurement.

Linear gain is preferred. High-crest material that cannot satisfy the peak
ceiling uses a short-lookahead gain limiter with iterative loudness correction.
The final mix receives one last peak-safety pass, then is quantized to the exact
PCM16 values before the report is measured. Reported segment values are
therefore values from the published WAV, not pre-write estimates.

## Report

The report contains:

- output format, sample count, and exact sample-derived duration;
- per-segment source and manifest target times;
- `mapped_start_sec`, `actual_start_sec`, `actual_end_sec`;
- `integrated_loudness_lufs`, `peak_dbfs`;
- `shifted`, `caused_shift`, `truncated`, `crossfade_ms`;
- structured warnings with stable codes.

`null` is used when a segment has zero duration after truncation. JSON is
written with `allow_nan=False`.

## Errors and Exit Codes

All expected failures print one JSON object to `stderr` and return an integer:

```json
{
  "status": "error",
  "code": "SAMPLE_RATE_MISMATCH",
  "exit_code": 3,
  "message": "target dialogue sample rate does not match the manifest: C:\\...\\segment.wav",
  "file": "C:\\...\\segment.wav",
  "details": {
    "expected_sample_rate": 48000,
    "actual_sample_rate": 44100,
    "role": "target_dialogue"
  }
}
```

| Exit | Meaning | Representative codes |
|---:|---|---|
| 0 | Success | `OK` |
| 1 | Unexpected internal error | `INTERNAL_ERROR` |
| 2 | CLI or manifest validation | `MISSING_FIELD`, `INVALID_TIME_RANGE`, `TARGET_LANGUAGE_NOT_FOUND` |
| 3 | Missing/undecodable/mismatched audio | `AUDIO_FILE_MISSING`, `AUDIO_DECODE_FAILED`, `SAMPLE_RATE_MISMATCH` |
| 4 | Timeline/offset rule violation | `OFFSET_OUT_OF_RANGE` |
| 5 | Output publication failure | `OUTPUT_WRITE_FAILED` |

Validation and decoding complete before any final output is touched. WAV and
report files are written to same-directory temporary files, verified, and
atomically replaced. On publication failure, partial files are removed and any
previous valid WAV/report pair is restored. Input failures therefore do not
leave incomplete `.wav` or report files.

## Tests

```powershell
py -m unittest discover -s tests -v
```

The suite generates deterministic 720p/30fps samples locally and checks
multi-language mapping, 44.1 kHz mono resampling, silence trim, segment
shifting/truncation, 10 ms crossfade, loudness/peak tolerances, structured
errors, offset rejection and partial-output cleanup.

See `VERIFICATION.md` for the latest recorded run.

