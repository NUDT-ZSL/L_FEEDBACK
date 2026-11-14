# voice-replay

`voice-replay` is a dependency-free C++17 command-line tool for deterministic
post-match production of an esports commentary mix. It reads local 48 kHz /
16-bit mono raw commentary tracks, aligns them to a match event timeline in
1024-frame blocks, and writes a stereo master WAV, one mono stem WAV per
commentator, and a stable JSON manifest.

The exporter supports process interruption at any block boundary. A resume
validates the input digest, job digest, completed output-prefix digests, and
clipping counters before continuing. The final files are byte-for-byte identical
to an uninterrupted export.

## Build

Requirements: CMake 3.20+, a C++17 compiler (Visual Studio 2022 or MinGW-w64),
and no external libraries or network access.

Linux / MinGW-w64:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
```

Visual Studio 2022:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

The executable is `build/bin/voice-replay` for single-config generators. For
Visual Studio the per-config directory is normally
`build/bin/Release/voice-replay.exe`.

## CLI

```text
voice-replay export --input <dir> --output <dir> [--resume]
voice-replay --input <dir> --output <dir> [--resume]
voice-replay generate --output <dir>
```

`generate` writes a deterministic local fixture: four raw tracks, four metadata
files, and `events.json`. Its 8100 ms match produces 388800 frames, so the last
1024-frame block contains only 704 valid frames. The fixture also covers a track
truncated by the raw length, a track clipped above 0 dBFS, a track whose speech
crosses the match end, and a track with zero available samples.

`--abort-after-blocks <n>` and the environment variable
`VOICE_REPLAY_ABORT_AFTER_BLOCKS=<n>` are test hooks. They terminate with exit
code 75 immediately after the state checkpoint for block `n`.

On success the tool exits 0 and writes one `SHA256` line per WAV plus a
`MANIFEST_SHA256` line to stderr.

## Input format

The input directory contains exactly one `events.json`, one metadata JSON per
commentary track, and the raw files referenced by that metadata.

`events.json`:

```json
{
  "sample_rate": 48000,
  "duration_ms": 8000,
  "events": [
    {"type": "kill", "match_time_ms": 1000},
    {"type": "tower", "match_time_ms": 3000},
    {"type": "kill", "match_time_ms": 6000},
    {"type": "tower", "match_time_ms": 7000}
  ]
}
```

A commentary metadata file:

```json
{
  "commentary_id": "caster_alpha",
  "raw_file": "caster_alpha.raw",
  "sample_rate": 48000,
  "start_frame": 0,
  "end_frame": 220000,
  "gain_db": -6.0,
  "events": [
    {"type": "kill", "match_time_ms": 1000, "local_frame": 0},
    {"type": "tower", "match_time_ms": 3000, "local_frame": 96000}
  ]
}
```

Accepted aliases include `id`/`commentaryId` for the ID, `file`/`audio_file`
for the raw name, and camelCase variants for the frame fields. If `local_frame`
is omitted on a marker, `start_frame` is used as that marker's local position.

### Alignment rule

For each track, markers are ordered by `(match_time_ms, type, local_frame)`. The
first marker is the anchor. It is matched to `events.json` by event type and
nearest match time, preferring an exact same-type match. Then:

```text
alignment_offset = round(anchor_event_time_ms * 48000 / 1000) - anchor_local_frame
global_frame = raw_frame + alignment_offset
```

The signed offset is recorded in the manifest and is applied inside every
1024-frame chunk. Audio is never shifted wholesale after block output.

## Processing and quantization

- Master and stems are always 48 kHz, 16-bit PCM. The master is stereo and each
  stem is mono.
- Every output has exactly `round(duration_ms * 48000 / 1000)` frames, including
  trailing silence outside a commentary segment.
- All tracks are summed equally into both master channels because the input
  protocol has no pan field. The two channels are therefore deterministic but
  identical.
- Gain is `pow(10, gain_db / 20)` and is accumulated as `double` float samples
  across tracks sorted by commentary ID.
- The sum is rounded with `llround`, then hard-clipped to the signed 16-bit
  range. Track clipping counts are counted on each isolated stem before the
  master sum; the master count is the number of stereo frames where either
  channel clipped.
- A missing or empty raw file still gets a full-length silent stem and does not
  contribute to the master.
- `used_frames` is the intersection of the available raw speech range with the
  match timeline. It is 0 for a completely unavailable track.

## Output and manifest

Outputs:

```text
master.wav
tracks/track_0001_<id>.wav
...
manifest.json
```

`manifest.json` has fixed key order. It records total frames, each WAV's file
name, channel count, frame count, SHA-256, each track's alignment offset,
available/requested/used frames, gain, clipping count, and anchor event. It
does not contain its own digest.

When an existing manifest and all WAV files still match the current input, a
repeat export prints `REUSE existing byte-identical outputs` and does not rewrite
the WAVs or manifest, so their timestamps remain unchanged.

## Resume protocol

After each block the tool flushes all WAVs and atomically replaces
`.voice-replay-state.json`. The state contains:

- completed block count and the current input/job SHA-256;
- each track's clipped-sample count;
- each track's monotonic PCM prefix SHA-256;
- the master clipped-frame count and PCM prefix SHA-256.

All block writes for `--resume` are validated against that state. Extra bytes
from an interrupted block are truncated; missing bytes, a WAV header mismatch,
a prefix digest mismatch, changed inputs, or a changed job configuration cause
`E_STATE_CORRUPT`. The tool never silently restarts from block zero. The state
file is removed only after the final manifest is atomically committed.

## Business errors

All metadata and event validation happens before any output file is created.

| Exit | Code | Meaning |
|---:|---|---|
| 0 | `OK` | Success |
| 2 | `E_USAGE` | Missing/invalid CLI arguments |
| 3 | `E_IO` | Input/output read or write failure |
| 4 | `E_META_INVALID` | Malformed metadata, negative frames, non-finite gain, duplicate ID, event out of range, or bad raw size |
| 5 | `E_RATE_MISMATCH` | Declared rate is not 48000 |
| 6 | `E_OUTPUT_UNWRITABLE` | Output directory/path cannot be created or written |
| 7 | `E_STATE_MISSING` | `--resume` requested without a state file |
| 8 | `E_STATE_CORRUPT` | State/input/job/output-prefix mismatch |
| 9 | `E_INTERNAL` | Unexpected internal failure |

Errors are emitted on stderr as:

```text
voice-replay: error=E_META_INVALID exit=4 message=...
```

## Verification

Linux:

```sh
./tests/run_acceptance.sh ./build/bin/voice-replay
```

Windows PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tests\run_acceptance.ps1 `
  .\build\bin\Release\voice-replay.exe
```

CMake also registers the same script as the `acceptance` test when the required
shell is available (`ctest --test-dir build --output-on-failure`).

The local verification run on the supplied fixture produced:

```text
master.wav    frames=388800 sha256=3838e64b7f061c4061d393aadbeb92b519b781ec85cecc5d3593f06d826dde4b
manifest.json              sha256=223eab51380a30d94793b1ae82eab6938b46801064ea4cd73525b5e6342c739d
```

The acceptance run confirmed:

- full export and a 17-block interrupted/resumed export are byte-identical for
  master, all four stems, and manifest;
- the final block is partial (704 of 1024 frames) and the WAV/manifest frame
  count remains exactly 388800, with no padding counted as valid audio;
- a second full export is byte-identical to the first;
- repeated export reuses files with unchanged WAV/manifest timestamps;
- negative frames, `NaN`/`Infinity` gain, 44100 Hz, out-of-range events, and
  duplicate IDs produce `E_META_INVALID` or `E_RATE_MISMATCH` with no partial
  master;
- changed input and a corrupted state prefix digest produce `E_STATE_CORRUPT`.

See `docs/DESIGN.md` for the state/commit ordering and additional edge-case
notes.
