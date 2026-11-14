# Snapshot Format

Cloud snapshots use little-endian integer and IEEE-754 binary32 fields. The magic
is the eight ASCII bytes `CLDSNAP0`. CRC32 uses the reflected IEEE polynomial
`0xEDB88320` (the same variant verified by the `123456789 -> 0xCBF43926` test).

## Header

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | magic `CLDSNAP0` |
| 8 | 4 | format version, 1 or 2 |
| 12 | 8 | payload byte count |
| 20 | 4 | CRC32 over payload only |
| 24 | payload size | version-specific payload |

The loader checks magic, then version selection, then payload length, then CRC32.
An unknown version is rejected as `E_VERSION_UNSUPPORTED` before interpreting any
version-specific payload offsets.

## V1 Payload Table

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | density seed (`u64`) |
| 8 | 4 | cloud type, 0=cumulus, 1=stratus |
| 12 | 4 | density resolution (`128`, `64`, or `32`) |
| 16 | 4 | last completed frame index |
| 20 | 4 | framebuffer width |
| 24 | 4 | framebuffer height |
| 28 | 4 | camera-history count; must be 1 |
| 32 | 64 | previous view matrix, 16 row-major `f32` |
| 96 | 64 | previous projection matrix, 16 row-major `f32` |
| 160 | `4*N` | previous ray-depth buffer, row-major `f32` |
| `160+4N` | `12*N` | accumulated HDR color, row-major RGB `f32` |
| `160+16N` | `4*N` | per-pixel accumulation count, row-major `u32` |

`N = width * height`. The payload is exactly `160 + 20N` bytes.

## V2 Payload Table

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | density seed (`u64`) |
| 8 | 4 | cloud type |
| 12 | 4 | density resolution |
| 16 | 4 | last completed frame index |
| 20 | 4 | framebuffer width |
| 24 | 4 | framebuffer height |
| 28 | 4 | camera-history count; must be 1 |
| 32 | 64 | previous view matrix |
| 96 | 64 | previous projection matrix |
| 160 | 12 | next-frame sun direction XYZ (`f32`) |
| 172 | 12 | next-frame sun color RGB (`f32`) |
| 184 | 4 | next-frame ambient light (`f32`) |
| 188 | 4 | maximum reprojection history length (`u32`) |
| 192 | `4*N` | previous ray-depth buffer |
| `192+4N` | `12*N` | accumulated HDR color |
| `192+16N` | `4*N` | per-pixel accumulation count |

The payload is exactly `192 + 20N` bytes.

The v2 data buffers start 32 bytes later than v1. The implementation has a
separate `kV1Layout` and `kV2Layout`; all scalar and buffer offsets are read from
the selected table.

## V1 Upgrade Defaults

A v1 load produces an in-memory v2-compatible state with:

- lighting enabled,
- sun direction `(0.45, 0.82, 0.25)` normalized,
- sun color `(1.0, 0.95, 0.86)`,
- ambient `0.16`,
- reprojection history length `8`.

The warning JSON event is:

```json
{"event":"W_SNAPSHOT_UPGRADED","code":"W_SNAPSHOT_UPGRADED", ...}
```

Snapshots are written to a temporary sibling file, flushed, and then published by
rename. Loading validates that all three arrays match the dimensions and version
layout before copying them into renderer state.

