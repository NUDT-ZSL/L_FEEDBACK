# GamePost

GamePost is a deterministic, resumable post-production pipeline for 48 kHz /
16-bit mono or stereo game recordings. It takes a game soundtrack, a narration
track, and a millisecond event table, then produces:

- a 16-bit PCM WAV with event-driven narration speed changes, silence gating,
  clipping repair, gated loudness normalization, and peak protection;
- one JSON export manifest containing input fingerprints, stage statistics,
  final loudness/peak measurements, and the chunk completion table;
- one JSON persistent state file used to resume after cancellation or an
  unclean process termination.

The project uses only Python 3.11, NumPy, and the Python standard library. It
does not use Docker, a service, a database, or network assets.

## Quick Start

Linux/macOS shell:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python -m gamepost gen-samples --out-dir samples
.venv/bin/python -m gamepost validate-events \
  --events samples/events.json \
  --mic samples/mic.wav \
  --report out/events.json

.venv/bin/python -m gamepost export \
  --game samples/game.wav \
  --mic samples/mic.wav \
  --events samples/events.json \
  --out out/mix.wav \
  --manifest out/manifest.json
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

.\.venv\Scripts\python -m gamepost gen-samples --out-dir samples
.\.venv\Scripts\python -m gamepost export `
  --game samples\game.wav `
  --mic samples\mic.wav `
  --events samples\events.json `
  --out out\mix.wav `
  --manifest out\manifest.json
```

The export command prints a compact JSON result. The detailed manifest is at
`out/manifest.json`, state is at `out/mix.wav.state.json`, and the partial WAV
while work is in progress is `out/mix.wav.part.wav`.

## Event Table

The event table is either a JSON array or an object with an `events` array:

```json
{
  "events": [
    {"start_ms": 1500, "duration_ms": 600, "category": "kill", "rate": 1.5},
    {"start_ms": 4000, "duration_ms": 1000, "category": "teamfight", "rate": 0.8},
    {"start_ms": 7800, "duration_ms": 500, "category": "crate", "rate": 2.0}
  ]
}
```

Aliases are accepted for integration with existing annotation tools:
`timestamp_ms`, `time_ms`, `t_ms` for start; `type`/`kind` for category;
`speed` for rate; and `duration` for `duration_ms`. A top-level
`default_rate` applies when an item has no rate.

The default allowed categories are `kill`, `teamfight`, `crate`, `击杀`,
`团战`, and `开箱`; use
`--event-categories "kill,teamfight,crate,boss"` to extend the allow-list.
Events must be ascending, non-overlapping, inside the narration track, have a
positive duration, and use a rate in the configured range (default 0.25 to 4).

Validation failures write a JSON report and exit with code `2`. Examples:

- `EVENT_OVERLAP` (1005): windows overlap in sample time.
- `EVENT_OUT_OF_RANGE` (1006): a window extends past the narration WAV.
- `EVENT_UNKNOWN_CATEGORY` (1007): category is not allowed.
- `EVENT_ORDER` (1004): timestamps are not ascending.
- `EVENT_RATE_INVALID` (1008): rate is missing, non-positive, or unsupported.
- `EVENT_TIMESTAMP_INVALID`, `EVENT_DURATION_INVALID`, `EVENT_ITEM_INVALID`,
  and `EVENT_JSON_INVALID` cover malformed fields and JSON.

Every manifest and error report contains the complete business-code table.
Other process-level codes are `INPUT_WAV_INVALID` (2001, exit 3),
`STATE_INVALID` (2003, exit 3), `MEMORY_QUOTA_INVALID` (2004, exit 3),
`CANCELLED` (3001, exit 130), and `INTERNAL_ERROR` (9001, exit 1).

## Timeline And Varispeed

Millisecond timestamps are rounded half-up to 48 kHz sample boundaries. Each
event window's nominal output length is:

```text
output_samples = round_half_up(source_window_samples / rate)
```

Output boundaries are accumulated as integers. Source positions inside a
window are computed as the exact rational
`local_output_index * source_window_samples / output_window_samples`, so there
is no floating-point drift and no later event can move by an accumulated
fractional sample. The manifest records both source and output boundaries for
every event and segment.

Normal-speed segments are copied sample for sample. Event segments use a
32-tap, 1024-phase windowed-Lanczos polyphase interpolator. Decimating windows
use a low-pass cutoff of `0.94 / rate` before sampling. The coefficient table is
built through a fixed polynomial sine path, and taps are summed in a fixed
order. The verification sample's 23 kHz tone decimated by 2x is attenuated by
more than 60 dB.

The timeline consequence is intentional: compressing one event window shortens
the content after that window. All later source events are mapped through the
cumulative integer boundaries; they are not looked up by the original wall
clock time.

## Processing Stages

1. **Clipping repair**: full-scale runs (`>= 32767` by default, at least two
   samples) are replaced with local Catmull-Rom reconstruction using neighboring
   non-clipped samples. Samples outside a detected run are compared and left
   byte-for-byte unchanged. Every run and the original peak are reported.
2. **Silence gate**: sliding RMS windows (20 ms window / 10 ms hop by default)
   detect quiet regions. Short gaps are merged, very short regions ignored, and
   the gate applies cosine fades (8 ms default) at both edges. Silence is
   muted, not deleted, so the timeline remains fixed. Logs contain each
   `start_sample`, `end_sample`, duration, linear gain, dB gain, and fade length.
3. **Event-driven varispeed**: applied to the gated narration as described
   above. The output length is exactly the timeline length.
4. **Loudness normalization**: game and narration are measured independently
   with ITU-R BS.1770-4-style K-weighting, 400 ms blocks at 75% overlap, and
   absolute/relative gates (-70 LUFS and -10 dB relative). Mono sources are
   measured in the final output channel layout. Default target is -16 LUFS for
   both sources, with a -3 dB game mix trim and +0 dB narration trim. Gain is
   clamped to +/-24 dB and the requested/actual gain is reported.
5. **Mix and peak protection**: sources are mixed in float64, then a single
   global peak scale limits the mix to -1 dBFS. The final mix is measured again
   for integrated loudness and peak.
6. **Deterministic PCM**: output uses explicit little-endian int16 conversion
   with `rint`, clipping to [-32768, 32767], and canonical chunk bytes.

## Resumable Export

Processing parameters and inputs are hashed into `parameter_fingerprint`:

- SHA-256 of the game WAV, microphone WAV, and event table;
- canonical normalized configuration;
- timeline fingerprint;
- pipeline and algorithm versions.

Each fixed-size output block is:

1. mixed from globally determined gains and peak scale;
2. written at its final offset in `mix.wav.part.wav`;
3. flushed and `fsync`-ed;
4. hashed into a chunk record (`index`, `start_frame`, `frames`,
   `byte_length`, `sha256`);
5. committed by atomically replacing the state JSON.

The state JSON is written to a same-directory temporary file, flushed,
`fsync`-ed, and moved with `os.replace`; directory sync is attempted where the
platform supports it. A partially written chunk is never recorded. On restart,
each recorded chunk hash is checked against the partial file. The header,
schema version, parameter fingerprint, and timeline fingerprint are checked
first.

If the state version or fingerprint does not match, the old state, partial
WAV, final WAV, and manifest are discarded and the complete export is recomputed.
If they match, verified chunks are skipped and only missing chunks are
rendered. Cancellation is checked before every chunk write, so it takes effect
at a block boundary. A `SIGKILL` needs no cleanup: the last atomically committed
state is sufficient to resume.

## Parallel Jobs And Resource Control

`batch` accepts a jobs file:

```json
{
  "jobs": [
    {
      "id": "episode-a",
      "game": "inputs/a-game.wav",
      "mic": "inputs/a-mic.wav",
      "events": "inputs/a-events.json",
      "output": "out/a.wav",
      "config": {"chunk_frames": 4800}
    },
    {
      "id": "episode-b",
      "game": "inputs/b-game.wav",
      "mic": "inputs/b-mic.wav",
      "events": "inputs/b-events.json",
      "output": "out/b.wav"
    }
  ]
}
```

Run it with shared limits:

```bash
python -m gamepost batch \
  --jobs jobs.json \
  --report batch-report.json \
  --workers 2 \
  --memory-mb 110
```

`ThreadPoolExecutor(max_workers=...)` bounds active workers. Every task leases a
conservative working-set estimate from one shared `MemoryQuota`; exhausted
quota blocks at the job boundary and resumes when another job releases memory.
It does not fail merely because another task is running. The batch report is a
single JSON file containing per-job status, output hashes, lease sizes, queue
and cancellation rules. If one job's conservative estimate exceeds the entire
quota, it waits for the quota to become idle, leases the whole quota exclusively,
and runs; this is a serialization fallback rather than an error.

## Test And Verification

```bash
python -m unittest discover -s tests -v
python scripts/verify_local.py
```

`scripts/verify_local.py` creates deterministic fixtures in a temporary
directory, runs independent exports, an invalid-table CLI check, a real
`SIGKILL` / resume test, and a two-job batch run under a 110 MiB shared quota.
It writes the observed result to `artifacts/verification_local.json`.

Observed in this workspace (Python 3.11.18, NumPy 1.26.4):

- 576,000 input frames mapped to 566,400 output frames with three event windows.
- Three clipped runs, 108 full-scale samples repaired; non-clipped samples
  remained unchanged.
- Seven silence segments gated with logged sample boundaries and fades.
- Final integrated loudness `-13.527538155 LUFS`, peak `-5.009443572 dBFS`.
- Two independent exports both produced WAV SHA-256
  `787ff5f3fdf05eee74ebb13c2a3360a8acd67337aa04f0adac1ba3ff445b6250`.
- A `SIGKILL` after 20 completed chunks left the final file absent and state
  `in_progress`; the resumed run was byte-identical.
- Two batch jobs completed under a `115,343,360`-byte quota with per-job leases
  of `99,731,704` bytes, proving the tasks were serialized by the shared quota
  rather than rejected.
- All 9 acceptance tests passed.

The manifest is intentionally a single JSON document. It includes all stage
diagnostics rather than writing a separate report file.
