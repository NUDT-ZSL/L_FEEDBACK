# ReplayForge

ReplayForge is a .NET 8 command-line renderer for esports replay post-production.
It accepts a manifest describing several H.264/MP4 video layers and a separate
48 kHz stereo AAC commentary track, synchronizes them by presentation time,
composes a bilinear-scaled picture-in-picture frame, normalizes commentary
loudness, muxes a standards-compliant MP4, and resumes safely after an
unreported process termination.

FFmpeg is used only as the deterministic decode/encode engine. Container and
bitstream inspection, timeline alignment, durability, MP4 sample tables, muxing,
and output integrity validation are implemented in C#.

## Build

Requires the .NET 8 SDK.

```powershell
# Windows x64 (FFmpeg is restored from NuGet and copied beside the app)
dotnet publish src/ReplayForge/ReplayForge.csproj -c Release -r win-x64 --self-contained false
.\src\ReplayForge\bin\Release\net8.0\win-x64\publish\replayforge.exe help
```

```bash
# Linux x64 development build
dotnet publish src/ReplayForge/ReplayForge.csproj -c Release -r linux-x64 --self-contained false
dotnet src/ReplayForge/bin/Release/net8.0/linux-x64/replayforge.dll help
```

The app locates FFmpeg in this order: `--ffmpeg`, manifest `ffmpegPath`,
`FFMPEG_PATH`, the publish directory, then `PATH`. Runtime network access is not
required.

## Commands

```text
replayforge generate --dir <dir> [--duration 20] [--seed 1337] [--overwrite]
replayforge run --manifest <manifest.json> [--force-restart] [--ffmpeg <path>]
replayforge demo --dir <dir> [--duration 20] [--seed 1337] [--overwrite]
replayforge validate --file <output.mp4>
```

`generate` creates one reusable deterministic fixture:

- Main POV: 320x180, 24 fps, H.264 High, first PTS 10.000 s.
- Player POV: 256x144, 30 fps, H.264 Main, first PTS 10.500 s.
- Commentary camera: 192x108, 25 fps, H.264 Constrained Baseline, first PTS 10.600 s.
- Commentary: standalone ADTS AAC-LC, 48 kHz, stereo.

The generated manifest aligns those media to timeline starts 0.0, 0.5, and 0.6 s
against a 10.0 s master PTS anchor. The output fixture is 640x360 at 30 fps with
one-second GOP segments.

All command outcomes are JSON on stdout. Program exit code is always `0`; use
the JSON `ok` and `error.code` fields for business outcomes.

## Manifest

The generated `manifest.json` is the complete example. Important fields:

```json
{
  "timeline": {
    "masterStartPtsSeconds": 10.0,
    "durationSeconds": 20.0,
    "outputFps": 30,
    "width": 640,
    "height": 360,
    "syncToleranceSeconds": 0.02,
    "targetLoudnessLufs": -16.0,
    "segmentDurationSeconds": 1.0
  },
  "videos": [
    {
      "id": "main",
      "path": "media/main-high-24fps.mp4",
      "timelineStartSeconds": 0.0,
      "layout": { "x": 0, "y": 0, "width": 640, "height": 360, "opacity": 1.0 }
    }
  ],
  "audio": {
    "path": "media/commentary-48k-stereo.aac",
    "sampleRate": 48000,
    "channels": 2
  },
  "outputPath": "output/final.mp4",
  "workDirectory": "work",
  "summaryPath": "summary.json"
}
```

## Processing details

- MP4 `moov`/sample tables and edit lists are parsed directly. The first visible
  presentation PTS must equal `masterStartPtsSeconds + timelineStartSeconds`
  within tolerance; otherwise the report is `E_SYNC_FAILED` with layer, path,
  actual PTS, expected PTS, and tolerance.
- H.264 `avcC` SPS/PPS are parsed. Baseline (profile 66), Main (77), and High
  (100) are accepted. Missing tracks, embedded/`avcC` SPS or PPS mismatches,
  malformed ADTS, unsupported AAC object types, and other H.264 profiles return
  `E_INVALID_STREAM`.
- Each layer is timestamp-resampled to the output rate with FFmpeg's `fps`
  filter, scaled with `flags=bilinear`, made RGBA, premultiplied through
  `colorchannelmixer`, and composited with `overlay=alpha=premultiplied`.
- Commentary is fully decoded and scanned using a BS.1770-style K-weighting and
  gated integrated-loudness calculation. One global gain is applied to the whole
  track, so GOP boundaries cannot cause gain steps. The output AAC remains
  48 kHz stereo.
- Video is encoded as H.264 High, `yuv420p`, with an IDR at the start of every
  segment. A custom muxer appends samples to a 64-bit `mdat`, writes one sample
  per chunk (`co64`), and writes a complete `moov` after all segments.

## Resume protocol

The work directory contains:

- `resume.json`: atomically replaced snapshot with manifest SHA-256, every input
  SHA-256, completed segment index, safe output byte length, audio cursor, and
  cumulative sample counts.
- `journals/seg-NNNNNN.json`: atomically written sample offsets/sizes/keyframe
  flags for one committed GOP.
- `chunks/`: transient encoded GOPs, removed only after commit.

For each GOP the renderer appends samples, fsyncs the MP4, atomically writes the
journal, then atomically publishes the snapshot. On startup it hashes all inputs,
rolls back output bytes beyond the snapshot length, removes journals beyond the
last committed GOP, and appends from that safe point. A previous snapshot with
`cleanShutdown=false` is reported as `forced-termination-recovery`; a clean
complete rerun is `normal-already-complete`.

Before reporting success, the MP4 is re-parsed. Validation checks moov/disk EOF,
mdat large-size bounds, track durations, sample counts, every chunk offset,
keyframe IDR bytes, and the AAC sample entry. Failure returns
`E_OUTPUT_CORRUPT` and preserves the work directory.

## Verification

Reproduce the full normal render, SIGKILL/forced-termination recovery, resumed
MP4 validation, and a negative synchronization test:

```bash
DOTNET=/path/to/dotnet scripts/e2e.sh
```

```powershell
.\scripts\e2e.ps1
```

Measured in this workspace with a 20-second fixture:

- Generated inputs reported 24/30/25 fps, first PTS 10.000/10.500/10.600 s, and
  High/Main/Constrained Baseline profiles.
- A renderer process was killed with `SIGKILL` after 3 of 20 GOPs. Its durable
  snapshot was at 237,036 bytes with `cleanShutdown=false`.
- 5,376 simulated partial bytes were appended. Restart selected
  `forced-termination-recovery`, rolled back exactly 5,376 bytes, resumed at
  segment 3, and produced 600 frames / 20.000 s.
- Final output: 1,678,616 bytes, H.264 High 640x360 at 30 fps, AAC-LC 48 kHz
  stereo, 20 keyframes, valid `moov`, and clean external FFmpeg decode.
- External `ebur128` measured the mixed commentary at `-16.0 LUFS`, matching
  the manifest target.
- `E_SYNC_FAILED`, `E_INVALID_STREAM` (missing track, High 4:4:4 profile,
  inconsistent SPS), and `E_OUTPUT_CORRUPT` were each exercised with exit code 0
  and JSON reports.
- Changing one input file's SHA-256 caused `rebuilt-input-changed`; the stale
  snapshot/output were discarded and all 120 frames were rendered again.

The end-to-end runtime tests above were executed on the workspace's Linux x64
host (WSL2). A Windows x64 publish was also produced and inspected: the app is a
PE32+ x86-64 executable and the publish directory contains `ffmpeg.exe`, its
native DLLs, and `ffprobe.exe`. The Windows runner script mirrors the same
`taskkill /T /F` recovery test, but could not be executed on this host.

The final AAC frame is indivisible, so the media sample table may contain up to
about 21.3 ms of encoder padding. The muxer adds an audio edit list that presents
the track exactly to the video duration; external `ffprobe` reports both tracks
and the format as exactly 20.000 s for the fixture.
