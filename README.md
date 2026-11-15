# Dialogue Baker

Deterministic offline baker for multilingual narrative-game dialogue. It reads
JSON planning tables, synthesizes 48 kHz mono PCM without a TTS service, aligns
all locales on one absolute sample timeline, protects overlapping-role mixes
from clipping, and emits:

- `index.json`: resource version, input/config hashes, absolute timeline,
  chunk table, subtitle timing, stretch details, and per-locale mix statistics.
- `dialogue-<version>.pack`: raw little-endian signed 16-bit PCM chunks.
- `bake_report.json`: per-locale counts, skipped entries, stretch details,
  alignment statistics, peak-protection flags, error codes, and final status.

The implementation uses only Python 3.11+, NumPy, and the standard library.
FFmpeg is not needed to bake; the test suite uses it when available to verify
that a chunk decodes as 48 kHz mono PCM.

## Quick Start

Linux/macOS:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/generate_sample.py

python -m dialogue_baker \
  --project examples/clean/project.json \
  --config examples/config.json \
  --output-dir build/clean \
  --mode full

python -m dialogue_baker \
  --project examples/clean/project.json \
  --config examples/config.json \
  --output-dir build/clean \
  --mode incremental
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python scripts\generate_sample.py

python -m dialogue_baker `
  --project examples\clean\project.json `
  --config examples\config.json `
  --output-dir build\clean `
  --mode full
```

After `pip install -e .`, `dialogue-baker` is also available as a console
script. The examples are already generated in the repository; running
`scripts/generate_sample.py` recreates them byte-for-byte.

Run verification:

```bash
python -m pytest -q
ruff check . --exclude .venv --exclude build
```

## Exit Codes

| Code | Report status | Meaning |
| ---: | --- | --- |
| `0` | `ok` | Every expected entry was baked or safely reused. |
| `2` | `degraded` | At least one entry or table was rejected, but valid entries were committed and the report is complete. |
| `3` | `failed` | Fatal input or output failure. A previous `index.json` and its referenced pack remain intact. |

The command exit code is always the `exit_code` stored in
`bake_report.json`.

## Input Model

`config.json` is global and is selected in full by `config_hash`:

```json
{
  "schema_version": 1,
  "sample_rate": 48000,
  "channels": 1,
  "sample_format": "pcm_s16le",
  "stretch": {"min_ratio": 0.75, "max_ratio": 1.25},
  "synthesis": {
    "crossfade_ms": 4.0,
    "attack_ms": 6.0,
    "release_ms": 9.0
  },
  "mix": {"target_peak_dbfs": -1.0},
  "pack": {"chunk_alignment": 64}
}
```

The sample rate must be 48000 and channels must be 1. The pack format is
`pcm_s16le`.

The project manifest maps locales, roles, voice parameters, and planning-table
paths:

```json
{
  "schema_version": 1,
  "project_id": "harbor_night",
  "locales": ["zh-CN", "en-US", "ja-JP"],
  "roles": {
    "heroine": {
      "gain_db": 1.5,
      "voice": {
        "fundamental_hz": 219.0,
        "amplitude": 0.32,
        "vibrato_hz": 4.6,
        "vibrato_depth_rad": 0.006,
        "harmonics": [
          {"multiple": 1.0, "gain": 1.0, "phase_deg": 0.0},
          {"multiple": 2.0, "gain": 0.28, "phase_deg": 31.0}
        ]
      }
    }
  },
  "tables": ["tables/scene01.json"]
}
```

A planning table contains slots. A slot has one absolute start point and slot
duration shared by every locale:

```json
{
  "schema_version": 1,
  "table_id": "scene01",
  "slots": [
    {
      "slot_id": "market_heroine",
      "role": "heroine",
      "start_sample": 12000,
      "duration_samples": 30000,
      "lines": {
        "zh-CN": {
          "text": "快看那边的灯",
          "syllables": [
            {"text": "快", "duration_samples": 4800},
            {"text": "看", "duration_samples": 5400},
            {"text": "那", "duration_samples": 3600},
            {"text": "边", "duration_samples": 5400},
            {"text": "的", "duration_samples": 3000},
            {"text": "灯", "duration_samples": 7800}
          ]
        }
      }
    }
  ]
}
```

Syllable durations are onset-to-onset annotations. Internal syllables overlap
the following syllable for a short linear cross-fade, but the final synthesized
line length remains exactly the sum of the annotated durations.

## Synthesis and Timing

For each syllable, the baker sums the configured sine harmonics, applies a
deterministic vibrato phase, and applies attack/release/cross-fade envelopes.
The waveform contains no randomness and no time-dependent choice. The line is
then linearly resampled to the slot duration when required.

The stretch ratio is:

```
target_slot_samples / annotated_natural_samples
```

If it falls outside `[stretch.min_ratio, stretch.max_ratio]`, that entry is
rejected with `STRETCH_LIMIT_EXCEEDED`; other entries continue.

Every final entry has exactly `duration_samples` PCM samples. Its subtitle
boundary is:

```
start_sample = absolute slot start
end_sample   = start_sample + duration_samples
```

Boundaries are represented as integer samples and as seconds rounded to six
decimal places. Integer samples are always the source of truth. The baker
round-trips each displayed seconds value back to samples and records
`start_error_samples`, `end_error_samples`, and `max_error_samples`. With a
48 kHz rate, six decimal places are sufficient to round-trip every sample;
baked entries in the acceptance fixtures have zero error.

All locales share the timeline's absolute sample origin. The engine selects one
locale, places its chunks by `timeline.start_sample`, and sums overlapping
role entries.

## Mixing and Peak Protection

Each line receives its role's linear `gain_db` adjustment. For each locale, the
baker sums all overlapping role entries in float64. If the pre-protection peak
exceeds `10^(target_peak_dbfs/20)`, one locale-wide scale factor is applied to
every entry in that locale before quantization. This preserves all chunk
lengths and absolute start/end samples.

Final per-locale `sample_peak`, `peak_dbfs`, `rms_dbfs`, `timeline_samples`, and
`clipped_sample_count` are calculated by reconstructing the integer mix from the
actual packed chunks over the full timeline, including silence. The index also
stores the SHA-256 of that int32 mix.
`peak_protection_triggered` and `mix_scale` are present in both the index and
report.

## Binary Pack

The pack starts with a 64-byte header. The packed header structure is
`<8sIIHHIII`:

1. Magic `DLGPACK1`
2. Pack format version
3. Sample rate
4. Channel count
5. Format code (`1` = little-endian signed 16-bit PCM)
6. Chunk count
7. Header byte size
8. Timeline sample count

Header strings occupy the remaining bytes as zero padding. Each chunk is raw
`s16le` PCM. Its absolute file offset is aligned to `pack.chunk_alignment`
(64 bytes in the sample), with zero padding between chunks. `index.json`
contains the complete chunk table with offset, byte length, sample count, and
SHA-256.

The pack filename contains the full resource version. It is fully
installed before `index.json` is atomically replaced. Therefore an old
`index.json` continues to reference its old, untouched pack until the commit
point. Unreferenced older versioned packs are removed only after the new index
is durable.

## Incremental Rules

`project_hash` covers the raw manifest and table file hashes plus parsed table
metadata. `config_hash` covers canonical config content. `input_version`
combines both with the baker version. `resource_version` covers the final index
metadata, chunk table, mix statistics, and pack hash.

An entry may be reused only when all of these hold:

- The previous index and pack are readable.
- Previous `baker_version` and `config_hash` match.
- The locale's full input signature matches, including every valid or rejected
  slot/locale entry.
- The entry's own `input_hash` matches.
- Its old chunk is in bounds, has the expected sample count, and matches its
  stored SHA-256.
- The old pack matches the index's full-file size and SHA-256.
- Reconstructed entry and locale loudness statistics match the old metadata.

Reused PCM is read by the old verified offset only as input data. The new pack
layout and every new offset are recomputed from scratch; old offsets are never
copied into the new index. Any input or config change invalidates the relevant
locale and re-bakes its entries. Unaffected locales can still be skipped.

## Degradation and Recovery

- A malformed manifest or config is fatal. No new pack or index is written.
- A malformed or missing planning table is a table-level `error`; entries from
  other tables continue.
- Missing lines, empty text, bad syllable data, unknown roles, missing stretch
  bounds, and oversized/undersized entries are per-entry errors.
- Per-entry errors do not stop unrelated entries. They make the final status
  `degraded` with exit code `2`.
- A fatal output failure writes a failure report and leaves the previous index
  and its referenced pack intact. New versioned pack files are never substituted
  into the old index.

Common error codes:

| Code | Scope |
| --- | --- |
| `INPUT_JSON_INVALID` | Fatal manifest/config JSON, or degraded table JSON |
| `INPUT_FILE_MISSING` / `INPUT_FILE_UNREADABLE` | Missing or unreadable table |
| `MANIFEST_INVALID` / `CONFIG_INVALID` | Fatal schema or configuration error |
| `TABLE_INVALID` / `SLOT_INVALID` / `DUPLICATE_SLOT` | Planning structure error |
| `MISSING_LINE` | Locale line absent |
| `EMPTY_TEXT` | Whitespace-only or empty dialogue text |
| `INVALID_LINE` / `INVALID_SYLLABLES` | Malformed line or syllable annotations |
| `VOICE_NOT_FOUND` / `VOICE_INVALID` | Role/voice mapping failure |
| `STRETCH_LIMIT_EXCEEDED` | Slot duration outside configured ratio bounds |
| `OUTPUT_WRITE_FAILED` | Fatal commit failure |
| `OLD_PACK_INVALID` | Warning; previous pack cannot be reused |

## Report Shape

The report includes:

- `summary.locale_entry_counts` and `locale_skipped_counts`
- total baked/skipped entries
- per-locale `stretch_adjusted_entries`
- one `stretch_details` record per accepted or rejected entry
- one `alignment_details` record per accepted entry
- `peak_protection_trigger_count` and per-locale mix statistics
- sorted `error_codes`, each detailed error with context, and warnings
- `resource_version`, `input_version`, `config_hash`, and `final_status`

## Local Verification

The supplied fixtures include:

- `examples/clean`: 3 locales, 5 slots, 4 roles, overlapping speech, mixed
  stretch ratios, and a triggered peak-protection case in `ja-JP`.
- `examples/degraded`: valid entries plus missing, empty, and over-limit
  entries.

Measured on the local Linux verification environment with Python 3.14.4,
NumPy 2.5.5, and FFmpeg 8.0.1:

- Full clean bake: exit `0`, 15 entries, 3 locales, all alignment errors `0`.
- `ja-JP` peak before protection: about `-0.397 dBFS`; after protection:
  about `-1.000 dBFS`, `clipped_sample_count = 0`, protection flag `true`.
- Immediate incremental rerun: `15` skipped, `0` baked, same resource version.
- One Chinese line text change: `5` zh-CN entries re-baked, `10` en-US/ja-JP
  entries skipped, both input and resource versions changed.
- Degraded fixture: exit `2`, 6 valid entries committed, and one each of
  `EMPTY_TEXT`, `MISSING_LINE`, and `STRETCH_LIMIT_EXCEEDED`.
- Corrupt manifest: exit `3`; SHA-256 of the previous index and pack unchanged.
- `pytest`: 14 passed.
- `ruff`: all checks passed.
- FFmpeg/ffprobe decode check: `48000 Hz`, `1 channel`, PCM codec.

The exact `resource_version`, pack size, and pack hash are deterministic for a
given input/config pair and baker version; inspect the generated
`bake_report.json` and `index.json` for the current values.
