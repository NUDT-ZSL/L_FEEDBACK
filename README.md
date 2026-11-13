# MXF R128 Transcode Pipeline

Deterministic batch transcoding from `.mxf` to internal-delivery MP4 with
EBU R128 loudness normalization and a single QC JSON report. The project is
pure Python 3.11, invokes local FFmpeg 6.x executables, does not use network
services or containers, and does not depend on external media.

## Requirements

- Python 3.11 or newer
- FFmpeg 6.x with `libx264` and AAC support
- Windows, Linux, or WSL

The executables are resolved in this order:

1. `FFMPEG_BIN` / `FFPROBE_BIN`
2. `ffmpeg` / `ffprobe` on `PATH`
3. `ffmpeg(.exe)` / `ffprobe(.exe)` beside the Python executable or in
   the project `bin/` directory

PowerShell example:

```powershell
$env:FFMPEG_BIN = "C:\ffmpeg\bin\ffmpeg.exe"
$env:FFPROBE_BIN = "C:\ffmpeg\bin\ffprobe.exe"
python -m mxf_r128 `
  --input-dir C:\jobs\in `
  --output-dir C:\jobs\out `
  --report C:\jobs\qc.json `
  --max-workers 4
```

## CLI

```text
python -m mxf_r128 --input-dir DIR --output-dir DIR --report FILE
                   [--max-workers N]
```

- Input files are the regular `.mxf` files in `--input-dir`, sorted by
  source file name.
- `--max-workers` defaults to `1`; results are still ordered by source name.
- Output names are `<source-stem>.mp4`.
- The report is written through a temporary file and atomically replaced.

Exit codes:

| Code | Meaning |
|---|---|
| `0` | Every file succeeded |
| `1` | At least one file failed QC or transcoding |
| `2` | Invalid arguments, missing input directory, or missing tools |
| `3` | Uncaught program error or report write failure |

The CLI writes only fixed business symbols such as `E_LOUDNESS_FAIL` to
stderr. FFmpeg banners, stderr logs, and raw exception strings are never
written to the QC report or CLI stderr.

## Output Contract

- Video: H.264 High profile, level 4.1, `yuv420p`, GOP 50.
- Resolution and frame rate are inherited from the source. Frame rate is
  normalized to exactly `25/1` or `30000/1001`.
- Audio: AAC, 48 kHz. Two-channel sources stay stereo. Six-channel sources
  use an explicit `FL FR FC LFE BL BR` pan and are never downmixed.
- Video and audio timestamps are independently reset to zero.
- Text subtitles are mapped to `mov_text` when present.
- Each transcode has a hard 30-second limit. On timeout the whole process
  tree is killed, the partial MP4 is removed, and processing continues.

### Loudness

The source is measured once with FFmpeg `loudnorm` analysis. A constant
gain is calculated for `-23.0 LUFS`, then capped so the predicted true peak
does not exceed `-1.0 dBTP`. The finished MP4 is measured again
independently; success requires:

- integrated loudness within `-23.0 +/- 0.5 LUFS`
- true peak at or below `-1.0 dBTP`

If the second check fails, the MP4 is retained, the file is not marked as
success, and `E_LOUDNESS_FAIL` is recorded. This deliberately exposes
signals whose loudness and peak constraints cannot both be met.

### Frame-rate recovery

1. Use ffprobe `avg_frame_rate` when valid.
2. Fall back to `r_frame_rate` and record `W_FPS_INFERRED`.
3. Infer from frame count/duration or video time base, also recording
   `W_FPS_INFERRED`.
4. If no inference is possible, use 25 fps and record `W_FPS_DEFAULT`.

Frame-rate recovery never directly fails a file.

### Audio delay

The source video/audio `start_time` values are preserved in the QC record.
Encoded audio and video both start at zero, and the output start-time delta
must be at most 10 ms. A source delay greater than one second also records
`W_AUDIO_DELAY`; the original measured delay remains unchanged in QC.

## QC JSON

The top level contains `input_dir`, `output_dir`, `max_workers`,
`thresholds`, `summary`, and the sorted `files` list. Each file record
contains:

`source_file`, `status`, `output_path`, `video_codec`, `video_profile`,
`video_profile_level`, `video_level`, `pix_fmt`, `frame_rate`, `width`,
`height`, `gop_size`, `audio_codec`, `loudness_lufs`, `loudness_lra`,
`true_peak_dbtp`, `channel_layout`, `audio_channels`,
`source_video_start_time`, `source_audio_start_time`,
`source_audio_delay_sec`, `output_video_start_time`,
`output_audio_start_time`, `output_start_delta_ms`, `warnings`, and
`error_code`.

Fixed symbols:

- Errors: `E_MEDIA_CORRUPT`, `E_LOUDNESS_FAIL`, `E_SYNC_FAIL`,
  `E_TIMEOUT`, `E_DIR_NOT_FOUND`, `E_FFMPEG_NOT_FOUND`,
  `E_OUTPUT_WRITE`, `E_REPORT_WRITE`, `E_ARGUMENT`, `E_UNEXPECTED`
- Warnings: `W_FPS_INFERRED`, `W_FPS_DEFAULT`, `W_AUDIO_DELAY`

## Deterministic Samples

Generate the complete local sample set:

```bash
FFMPEG_BIN=/path/to/ffmpeg FFPROBE_BIN=/path/to/ffprobe \
python -m tools.generate_samples \
  --output-dir artifacts/acceptance/samples \
  --duration 4 --include-subtitle --include-long --long-duration 400
```

The generator uses only FFmpeg `lavfi` sources. Real MXF OP1a files are
used for MPEG-2, AVC-Intra 29.97, 5.1, transient loudness, and missing-fps
cases. Timestamp-offset, subtitle, and overlong cases use a deterministic
container rewrite (MP4 payload under the `.mxf` input filename) so those
metadata conditions can be reproduced without external assets.

## Verification

Run unit and real-FFmpeg integration tests:

```bash
python -m unittest discover -s tests -v
```

Run the full acceptance batch and independent verifier:

```bash
FFMPEG_BIN=/path/to/ffmpeg FFPROBE_BIN=/path/to/ffprobe \
python -m tools.run_acceptance --work-dir artifacts/acceptance
```

Actual checked-in results are in `artifacts/acceptance/`:

- `qc.json`
- `verification.json`
- `run-result.json`
- `samples/samples-manifest.json`
- `output/`

The latest run used FFmpeg 6.0.1, 12 source files, and four workers:

- exit code `1` because failures were intentionally present
- 7 success, 5 expected failures
- independent verification: `all_checks_pass: true`
- no partial files and no remaining FFmpeg processes
