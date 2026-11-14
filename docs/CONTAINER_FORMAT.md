# IAPANOR1 Container Format

Version 1 is a little-endian binary format. All multi-byte integers use an
explicit little-endian layout, independent of host byte order.

## File Layout

```text
+----------------------+ offset 0
| Header (124 bytes)   |
+----------------------+ header_size = 124
| Metadata blocks      | frame_count * object_count * 20 bytes
+----------------------+ metadata_offset + metadata_size
| Comment JSON (UTF-8) | comment_size bytes
+----------------------+ comment_offset + comment_size
| Interleaved audio    | sample_count * channel_count * 4 bytes
+----------------------+ audio_offset + audio_size = file size
```

There are no gaps or trailing bytes. `channel_count` equals `object_count`.
The metadata block count is:

```text
frame_count = ceil(sample_count / frame_length)
```

A frame is a render block containing up to `frame_length` sample frames. The
last frame may contain fewer samples. Metadata frame `i` describes the audio
samples whose absolute indices are in:

```text
[i * frame_length, min((i + 1) * frame_length, sample_count))
```

## Header

The fixed header is `struct.Struct("<8sHHIIIHHQQQQQQQQ32s")`.

| Field | Type | Bytes | Meaning |
|---|---:|---:|---|
| `magic` | 8 bytes | 8 | ASCII `IAPANOR1` |
| `version` | uint16 | 2 | Currently `1` |
| `header_size` | uint16 | 2 | Exactly `124` |
| `flags` | uint32 | 4 | Must be `0` in version 1 |
| `sample_rate` | uint32 | 4 | Samples per second |
| `frame_length` | uint32 | 4 | Samples per metadata/render block |
| `channel_count` | uint16 | 2 | Must equal `object_count` |
| `object_count` | uint16 | 2 | `1..64` |
| `sample_count` | uint64 | 8 | Total audio sample frames |
| `frame_count` | uint64 | 8 | Number of metadata blocks |
| `metadata_offset` | uint64 | 8 | Must be `124` |
| `metadata_size` | uint64 | 8 | `frame_count * object_count * 20` |
| `audio_offset` | uint64 | 8 | Start of interleaved audio |
| `audio_size` | uint64 | 8 | `sample_count * object_count * 4` |
| `comment_offset` | uint64 | 8 | Start of comment JSON |
| `comment_size` | uint64 | 8 | UTF-8 comment length |
| `payload_sha256` | 32 bytes | 32 | SHA-256 of metadata, comments, and audio |

## Metadata Blocks

Metadata is block-major:

```text
frame 0 object 0, frame 0 object 1, ..., frame 1 object 0, ...
```

Every object record is five float32 values in this order:

| Field | Unit |
|---|---|
| `azimuth` | degrees, positive toward the right |
| `elevation` | degrees, positive above the horizontal plane |
| `distance` | meters |
| `gain` | linear amplitude |
| `size` | application-defined size scalar |

The object index is the channel index. Object metadata is sampled at the first
sample of its render block:

```text
t = block_index * frame_length / sample_rate
```

## Audio Data

Samples are IEEE float32 PCM, interleaved by sample frame:

```text
sample 0 object 0, sample 0 object 1, ..., sample 1 object 0, ...
```

The channel order is exactly the object order in the scene. Float samples are
not clipped by the pack or render stages.

## Comment Block

The comment block is compact, key-sorted UTF-8 JSON. It records the SHA-256 of
every source clip as raw file bytes:

```json
{
  "format": "IAPANOR1",
  "objects": [
    {
      "clips": [
        {
          "frames": 288000,
          "loops": 1,
          "path": "voice.wav",
          "sha256": "hex digest",
          "start": "0.0"
        }
      ],
      "id": "voice",
      "index": 0
    }
  ],
  "version": 1
}
```

## Validation

Opening a container verifies:

- Magic and version.
- Fixed header size and flags.
- Object/channel count relation and `1..64` object range.
- `frame_count = ceil(sample_count / frame_length)`.
- Metadata and audio byte-size formulas.
- Contiguous offsets and exact file length.
- UTF-8 comment JSON.

`render` additionally verifies the SHA-256 payload digest before writing the
output WAV. This detects metadata, comment, or audio corruption.
