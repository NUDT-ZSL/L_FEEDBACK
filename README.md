# Deterministic Object-Based Panoramic Audio Pipeline

This repository implements an offline, deterministic pack/render pipeline for
object-based panoramic audio. It parses a scene JSON, decodes and mixes mono
object buses, writes a custom `IAPANOR1` container with aligned per-frame
metadata, and renders an 8-channel reference WAV using VBAP.

The implementation is pure Python plus NumPy and soundfile. No network,
container runtime, audio device, or system service is used.

## Requirements

- Python 3.11 or newer
- NumPy 1.24 or newer
- soundfile 0.12 or newer
- FFmpeg 6.0 or newer with `ffprobe` in `PATH` for external WAV validation

```bash
python -m pip install -r requirements.txt
python -m immersive_audio --help
```

The current environment was validated on Linux/WSL with Python 3.14.4,
NumPy 2.5.4, soundfile 0.14.0, and FFmpeg/ffprobe 7.0.2. Windows was separately
validated with Python 3.11.9, NumPy 1.26.4, soundfile 0.12.1, and
FFmpeg/ffprobe 8.0 on a Windows local volume.

## Quick Start

```bash
python tools/generate_demo.py --output-dir examples/demo

python -m immersive_audio pack \
  examples/demo/scene.json \
  examples/demo/scene.iap \
  --stats examples/demo/pack-stats.json

python -m immersive_audio render \
  examples/demo/scene.iap \
  examples/demo/scene-8ch.wav \
  --stats examples/demo/render-stats.json

ffprobe -v error \
  -show_entries stream=codec_name,sample_rate,channels,duration \
  -of json examples/demo/scene-8ch.wav
```

Both positional arguments and `--scene`/`--output` or `--input`/`--output`
forms are supported.

## Scene JSON

Canonical example:

```json
{
  "sample_rate": 48000,
  "frame_length": 1024,
  "duration_seconds": 2.0,
  "objects": [
    {
      "id": "voice",
      "keyframes": [
        {
          "time": 0.0,
          "azimuth": 0.0,
          "elevation": 0.0,
          "distance": 1.0,
          "gain": 1.0,
          "size": 0.25
        },
        {
          "time": 2.0,
          "azimuth": 90.0,
          "elevation": 15.0,
          "distance": 2.0,
          "gain": 0.5,
          "size": 0.5
        }
      ],
      "clips": [
        {
          "path": "voice.wav",
          "start": 0.0,
          "loops": 1
        }
      ]
    }
  ]
}
```

Time values are seconds and are independent of the sample rate. Clip paths are
resolved relative to the scene JSON file. `loops` is the total number of plays,
so `loops: 1` plays once and `loops: 2` plays twice.

Supported aliases include:

- Root: `sampleRate`, `frameLength`, `duration`
- Keyframes: `time_seconds`, `t`, `azimuth_deg`, `az`, `elevation_deg`, `el`,
  `radius`
- Separate curves: `position_keyframes`/`positions` plus `gain_curve`
- Clips: `file`, `start_time`, `offset`, `loop_count`, `repeat`

Validation rules:

- `1 <= objects <= 64`
- `8000 <= sample_rate <= 384000`
- `1 <= frame_length <= 1048576`
- Position keyframe times are finite, non-negative, and strictly increasing
- Gain curve times, when present, are finite, non-negative, and strictly increasing
- Object IDs are unique; distances and sizes are non-negative

## Timeline Semantics

A frame is a render block of `frame_length` audio sample frames. One metadata
record exists for every object in every render block:

```text
metadata_frames = ceil(total_audio_samples / frame_length)
```

Metadata is sampled at the first sample of each block. Keyframes use linear
interpolation. Values before the first keyframe and after the last keyframe are
clamped to that endpoint. Conversion from seconds to sample frames uses exact
decimal arithmetic and round-half-up:

```text
frame = floor(seconds * sample_rate + 0.5)
```

Source clips with a different sample rate are linearly resampled. Multi-channel
clips are downmixed by channel mean. Clip overlaps are added, loops repeat the
whole resampled clip, and samples past the scene duration are cropped.

## Renderer

The renderer uses eight ring speakers at azimuths `0, 45, ..., 315` degrees.
Each source is panned to its adjacent speaker pair with vector-base amplitude
panning and L2-normalized pan gains. The two-pair amplitudes are:

```text
g_left  = sin(angle_right - angle) / sin(angle_right - angle_left)
g_right = sin(angle - angle_left) / sin(angle_right - angle_left)
```

Elevation is represented by horizontal projection attenuation
`max(0, cos(elevation))`; at and beyond +/-90 degrees the ring contribution is
zero. Distance attenuation uses the inverse square law:

```text
distance_gain = 1 / max(distance, minimum_distance)^2
```

The default minimum distance is `0.1` m and is configurable with
`--min-distance`. Object gain multiplies the distance and VBAP gains. Samples
are kept as float32 and are not clipped, so the container-to-render transform
can be compared exactly with:

```python
expected = object_audio @ spatial_gain_matrix(metadata)
```

The WAV output is 8-channel IEEE float32 (`pcm_f32le`) at the scene sample rate.

## Container

The binary format is documented in [docs/CONTAINER_FORMAT.md](docs/CONTAINER_FORMAT.md).
In brief:

- Fixed 124-byte little-endian header with magic, version, sample rate, frame
  length, channel/object count, sizes, and payload SHA-256.
- Block-major metadata: five float32 fields per object
  (`azimuth`, `elevation`, `distance`, `gain`, `size`).
- Key-sorted UTF-8 comment JSON with every clip SHA-256.
- Sample-interleaved IEEE float32 PCM in object-index channel order.

The reader validates magic, version, formulas, offsets, and exact data length.
The render command verifies the SHA-256 payload before creating the WAV.

## CLI Statistics

Successful commands print a JSON object and optionally write the same object
with `--stats PATH`. Important fields:

- `peak_memory_mb`: process peak RSS, using `GetProcessMemoryInfo` on Windows,
  `/proc/self/status` or `resource` on Unix
- `within_memory_limit`: peak is at or below 512 MiB
- `stages_seconds`: load/analyze/mix/write/verify/render stage timings
- `counts`: sample rate, frame length, objects, clips, samples, metadata blocks
- `object_bus_ranges`: valid start/end sample frames for every packed object bus

## Error Codes

Errors are written to stderr as `CODE: message` and return a non-zero status.

| Code | Meaning |
|---|---|
| `E1001` | Scene file missing/unreadable |
| `E1002` | Invalid JSON |
| `E1003` | Invalid scene schema |
| `E1004` | Invalid sample rate |
| `E1005` | Invalid frame length |
| `E1006` | Object count outside `1..64` |
| `E1007` | Non-monotonic keyframe timestamps |
| `E1008` | Invalid duration |
| `E1009` | Duplicate object ID |
| `E2001` | Missing audio clip |
| `E2002` | Clip decode failure |
| `E2003` | Invalid audio format/channels |
| `E2004` | Audio I/O failure |
| `E3001` | Bad container magic |
| `E3002` | Unsupported container version |
| `E3003` | Truncated container |
| `E3004` | Invalid container layout/data length |
| `E3005` | Payload checksum mismatch |
| `E3006` | Invalid comment block |
| `E4001` | WAV render I/O failure |
| `E4002` | Invalid render metadata/audio |
| `E9999` | Unexpected internal failure |

## Memory Design

The pack path never holds all object buses or all audio in memory:

1. Clip headers and file hashes are analyzed with bounded buffers.
2. Each object bus is mixed into a temporary float32 file on the output volume.
3. Metadata is generated and written one block at a time.
4. Object buses are interleaved to the container in bounded chunks.
5. Render verifies and reads the container in bounded chunks.

The 10-minute benchmark uses 64 objects and 100 clip references per object.
Both measured runs stayed below 512 MiB:

| Environment | Pack | Render |
|---|---:|---:|
| Linux/WSL | 65.48 s, 70.922 MiB peak | 21.64 s, 41.336 MiB peak |
| Windows 3.11 local volume | 187.47 s, 66.305 MiB peak | 72.87 s, 39.438 MiB peak |

Both runs produced a 600-second, 64-object container and a 600-second,
8-channel, `pcm_f32le` WAV. Exact statistics are in
`artifacts/benchmark-64x100-10min.json` and
`artifacts/benchmark-windows-64x100-10min.json`.

## Tests and Validation

```bash
python -m unittest discover -s tests -v
python tools/validate.py
python tools/benchmark_10min.py
```

The suite contains 42 deterministic tests covering scene validation, frame
rounding/interpolation, overlap, looping, zero-length clips, resampling,
minimum/maximum object counts, extreme coordinates, container round-trip and
corruption, VBAP, WAV/container equality, stable CLI errors, ffprobe parsing,
and the 64-object/6400-clip-reference path.

`tools/validate.py` writes `artifacts/validation-report.json` with the actual
test result, environment versions, checksum/layout checks, ffprobe output, and
runtime statistics. The actual Linux and Windows reports are
`artifacts/validation-report.json` and
`artifacts/validation-report-windows.json`.

## Windows Reproducibility

- All integer fields are explicitly little-endian.
- JSON is emitted with sorted keys and fixed separators.
- SHA-256 is computed from raw source bytes and raw container payload bytes.
- Frame rounding uses `Decimal` with `ROUND_HALF_UP`.
- Temporary files are created next to the output on the same local volume.
- Paths are passed through `os.fspath`; JSON stores normalized forward slashes.
- Peak memory uses the Windows process memory counters.

Given identical input bytes, scene JSON, and implementation version, the pack
output bytes are deterministic. The test suite verifies byte-for-byte repeat
packing on both Linux/WSL and Windows Python. The Windows acceptance run
reported 38.91 MiB pack peak and 33.5 MiB render peak for the demo scene.

## Project Layout

```text
immersive_audio/
  audio.py        clip analysis, resampling, mixing, object buses
  cli.py          pack/render commands and JSON statistics
  container.py    IAPANOR1 reader/writer and validation
  metrics.py      cross-platform peak RSS measurement
  renderer.py     VBAP, distance gain, 8-channel WAV rendering
  scene.py        JSON model and validation
  timeline.py     decimal frame conversion and interpolation
tests/            deterministic unittest suite
tools/            demo generator, acceptance runner, 10-minute benchmark
docs/             binary format specification
```
