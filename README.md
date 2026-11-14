# Radio Feedback Pipeline

Deterministic offline receiver simulation for a game radio: timestamped 20 ms
packets, jitter-buffer reordering, duplicate suppression, timeout discard, PLC,
sample-accurate crossfade/de-click, FFmpeg WAV/Ogg output, log-derived reporting,
and independent verification.

The packet payload is a decoded 20 ms / 960-sample mono PCM frame at 48 kHz. This
is the receive-side representation requested by the task, so a real Opus decoder
is not needed in the offline harness. The transcode stage uses FFmpeg's real
`libopus` encoder with a 64 kbps target for the Ogg output.

## Pipeline

```text
JSON config
   |
   v
deterministic source PCM --> deterministic network event JSONL
                                      |
                                      v
                           receive + dedupe + timeout window
                                      |
                                      v
                         jitter buffer / buffer watermark
                         / fast-trim / low-water PLC
                                      |
                                      v
                 fixed-length PLC + pitch repeat + comfort noise
                      + sample-level boundary fade/de-click
                                      |
                                      v
                exact-length PCM --> FFmpeg --> WAV / Ogg(Opus 64k)
                                      |
                                      v
              packet logs / frame logs / report.json / verify
```

The output timeline is always exactly `packet_count * 960` samples. Missing
slots are concealment-filled with the selected PLC policy, or muted under the
`mute` strategy. They never shorten or extend the timeline.

## Windows Setup

1. Install Python 3.10 or newer. In PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Install FFmpeg 6.0 or newer with `libopus` support, add its `bin` directory
   to `PATH`, and confirm:

```powershell
ffmpeg -version
ffprobe -version
```

Alternatively set `FFMPEG_BIN` and `FFPROBE_BIN` to absolute executable paths.

3. Run the validation script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_validation.ps1
```

## Commands

```powershell
python -m radio_pipeline run --config config\default.json --output-dir artifacts\demo
python -m radio_pipeline report --run-dir artifacts\demo --output artifacts\demo\report.rebuilt.json
python -m radio_pipeline verify --run-dir artifacts\demo
```

`run` generates `logs\input_packets.jsonl` when `--input` is omitted. To replay a
preconstructed sequence:

```powershell
python -m radio_pipeline run --config config\default.json `
  --input path\to\packets.jsonl --output-dir artifacts\replay
```

`report` reads only `logs\processing_log.json` and rebuilds all counters. `verify`
uses `soundfile`, FFmpeg, and `ffprobe` to decode the actual outputs and compares
them with the logs and report.

Exit codes:

- `0`: success
- `2`: invalid arguments/config/environment/input
- `3`: report reconstruction or output verification failure

## Artifacts

```text
artifacts/demo/
  radio.wav                  PCM_S16LE, 48 kHz, mono
  radio.ogg                  Ogg/Opus, 64 kbps target
  processed.pcm              exact processed int16 PCM
  report.json                log-derived report
  run_summary.json           compact command result
  audio_metadata.json        embedded metadata copy
  logs/
    input_packets.jsonl      deterministic wire sequence
    input_manifest.json      generation seed/config summary
    accepted_packets.jsonl   accepted packets, sequence/time monotonic
    packet_events.jsonl      received/duplicate/lost/timeout/invalid events
    frame_log.jsonl          one entry per 20 ms timeline frame
    processing_log.json      complete input to report reconstruction log
```

Packet events contain sequence, timestamp, arrival time, status, buffer water
level, fast-consumption marker, PLC trigger, strategy, fade regions, and
diagnostics. `invalid` is an additional arrival status used for malformed
payloads; it never interrupts processing. The source packet itself is then
`received`, `timeout`, or `lost`.

Accepted packets are written in ascending sequence order. Their timestamps are
strictly increasing because every source sequence owns one fixed 20 ms slot.

## Config

See `config/default.json`, `config/sequential.json`, and `config/extreme.json`.
All unspecified keys inherit from the built-in defaults.

### Network generation

| Key | Meaning |
| --- | --- |
| `seed` | Fixed PCG64 seed for source and network generation |
| `network.ordering` | `reordered` or `sequential` |
| `network.reorder_probability` | Extra-delay probability |
| `network.reorder_min_extra_ms`, `reorder_max_extra_ms` | Reorder delay range |
| `network.loss_rate` | Independent loss probability |
| `network.duplicate_rate` | Duplicate-copy probability |
| `network.burst_length`, `burst_probability` | Random burst-loss behavior |
| `network.arrival_jitter.distribution` | `normal`, `uniform`, `exponential`, or `none` |
| `network.arrival_jitter.scale_ms`, `max_abs_ms` | Jitter spread and clipping |
| `network.base_latency_ms` | Baseline transport delay |
| `network.forced_bursts` | Deterministic burst-loss runs |
| `network.forced_late_sequences` | Deterministic timeout arrivals |
| `network.forced_corrupt_sequences` | Invalid-payload diagnostic arrivals |

The generated JSONL is sorted by arrival time with deterministic tie-breaks.
Duplicate packets are post-arrival copies of the same sequence; the receiver
retains only the first valid copy.

### Jitter buffer and PLC

| Key | Meaning |
| --- | --- |
| `jitter_buffer.min_delay_ms` | Initial playout lead and low-water threshold |
| `jitter_buffer.max_delay_ms` | Watermark above which playback lead is trimmed |
| `jitter_buffer.timeout_ms` | Arrival must not exceed `timestamp + timeout_ms` |
| `jitter_buffer.fast_consume_step_ms` | Lead shortening step when over max |
| `jitter_buffer.conservative_max_delay_ms` | Raised limit after degradation |
| `plc.strategy` | `adaptive`, `pitch`, `comfort`, or `mute` |
| `plc.fade_ms` | Fade-out/fade-in length at PLC/normal boundaries |
| `plc.pitch_decay`, `pitch_min_hz`, `pitch_max_hz` | Pitch-repeat behavior |
| `plc.isolated_decay`, `comfort_noise_gain` | Comfort-noise decay/level |
| `plc.conservative_aggression` | Reduced PLC gain after degradation |

`adaptive` uses pitch-period repetition with progressive decay for runs of two
or more missing frames, and short decaying comfort noise for isolated losses.
Pitch is estimated by normalized autocorrelation over the last good frames.
Every concealed frame remains exactly 960 samples.

Boundary handling applies a configurable linear fade-out on the last normal-frame
samples and fade-in on the first PLC-frame samples (and the reverse at PLC exit).
Both samples directly adjacent to the boundary approach zero, avoiding a
discontinuity/click while preserving sample count.

### Degradation

The initial buffer depth in frames is `ceil(max_delay_ms / 20)`. If a missing
run exceeds this depth, the pipeline automatically:

- raises the effective maximum buffer limit to `conservative_max_delay_ms`,
- reduces PLC aggression to `conservative_aggression`,
- records a structured degradation event in `report.json`.

Separately, a watermark above `max_delay_ms` increases `trim_total_ms` in
`fast_consume_step_ms` increments, shortening the playout lead and recording a
fast-consumption event.

## Metadata and Ogg duration

The embedded comment tag contains machine-readable text beginning with
`RADIO_PIPELINE_METADATA=`. It includes sample rate, channels, duration,
total samples, packet count, missing packet count/ranges, and timeline policy.
A JSON sidecar is also written.

Ogg/Opus stores a decoder pre-skip (312 samples with FFmpeg/libopus). This is
correctly removed on decode, but container-level `ffprobe` duration therefore
reads 312 samples longer than the audible timeline. `verify` therefore checks
the decoded PCM sample count for the required one-sample tolerance. The WAV
container and decoded PCM are exact.

Output loudness is an approximate integrated LUFS value:

```text
-0.691 + 10*log10(sum(x^2) / (N * 32768^2))
```

The same sum-of-squares is logged per frame, so `report` is reproducible from
logs alone. `verify` recomputes loudness from both decoded formats and compares
against the report using `loudness_tolerance_lufs`.

## Determinism

- Source audio uses a fixed seed.
- Network generation uses a fixed PCG64 seed.
- JSONL ordering and numerical formatting are canonical.
- FFmpeg runs with bitexact flags and a fixed Ogg serial offset.
- PCM, WAV, and Ogg SHA-256 hashes are identical across repeated runs of the
  same config on the same FFmpeg build.

## Tests

```powershell
python -m unittest discover -v
```

The suite covers config validation, generator determinism, malformed payload
diagnostics, duplicate/timeout/loss handling, fixed-length pitch PLC, fade
boundaries, lossy report reconstruction, audio verification, and repeated-run
binary determinism.

## Validation Record

The current workspace was validated with Python 3.14, NumPy 2.5.4,
soundfile 0.14.0, and FFmpeg 7.0.2 (static, libopus enabled). The code paths are
Windows-compatible and the PowerShell validation script is included. Exact
results are recorded in `VALIDATION.md`.
