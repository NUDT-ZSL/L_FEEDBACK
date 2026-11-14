# Binary formats

All integers and floats are little-endian. `f32` is IEEE-754 binary32 and
`u32`/`u64` are unsigned integers.

## Scene: `QBSC`, version 1

| Offset | Type | Meaning |
|---:|---|---|
| 0 | 4 bytes | magic `QBSC` |
| 4 | u32 | version, currently `1` |
| 8 | u32 | output width |
| 12 | u32 | output height |
| 16 | u32 | layer count |
| 20 | u32 | square texture size |

Immediately after the header:

```text
f32 texels[tex_size * tex_size * 4]   # R,G,B,A per texel, each in [0,1]
```

Then each layer repeats:

```text
u32 vertex_key_count                   # >= 1
repeat vertex_key_count:
    f32 time                           # [0,1], nondecreasing
    f32 displacement[4][2]             # x/y displacement for 4 vertices

u32 opacity_key_count                  # >= 1
repeat opacity_key_count:
    f32 time                           # [0,1], nondecreasing
    f32 opacity                        # [0,1]

u32 playback                           # 0 = clamp, 1 = loop
f32 base[4][3]                         # x,y,z per vertex
f32 uv[4][2]                           # u,v per vertex
f32 albedo[3]                          # [0,1]
f32 roughness                          # [0,1]
f32 specular                           # [0,1]
u32 premultiplied                      # 0 or 1
```

The loader rejects non-finite values, out-of-range material values, unsorted
key times, unknown flags, impossible dimensions, truncated records, and any
trailing bytes. A complete valid byte string is hashed with 64-bit FNV-1a; that
hash is stored in snapshots.

## Snapshot: `QSNP`, versions 1 and 2

Fixed header:

| Offset | Type | Meaning |
|---:|---|---|
| 0 | 4 bytes | magic `QSNP` |
| 4 | u32 | version |
| 8 | u64 | scene FNV-1a hash |
| 16 | u32 | width |
| 20 | u32 | height |
| 24 | u32 | scene layer count |
| 28 | u32 | completed layer count |

Version 2 additionally stores:

| Offset | Type | Meaning |
|---:|---|---|
| 32 | u32 | raw bits of render time |
| 36 | u32 | degenerate layer count |

Then:

```text
u64 layer_pixels[layer_count]
f32 premultiplied_rgba[width * height * 4]
```

Parsing is dispatched by the version field. Version 1 predates the explicit
time/degenerate fields, so it is accepted only for time `0.0`; its historical
last-statistics high word is interpreted as the optional degenerate count.
Unknown versions are discarded. Every field is read through the version's
layout rather than a fixed reused file offset.

## Output: `QRBA`, version 1

| Offset | Type | Meaning |
|---:|---|---|
| 0 | 4 bytes | magic `QRBA` |
| 4 | u32 | version, currently `1` |
| 8 | u32 | width |
| 12 | u32 | height |
| 16 | bytes | premultiplied R,G,B,A, row-major |

Each channel is `uint8(floor(clamp(float_value, 0, 1) * 255))`.

## Atomic replacement

Temporary names have the form:

```text
TARGET.tmp.PROCESS_ID.SEQUENCE
```

They are created with exclusive-create semantics in the target directory.
After all bytes are written and flushed, the temporary is atomically renamed
over the target (`rename` on POSIX, `MoveFileExW` with replace/write-through on
Windows). A crash can leave a temporary file, but never exposes its bytes under
the target name.
