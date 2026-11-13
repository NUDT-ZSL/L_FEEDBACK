# Snapshot Binary Format

All integer and floating-point fields are little-endian. `float` follows IEEE-754
binary32. The writer never serializes host structs directly.

## Header (`64` bytes)

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 8 | Magic `SKNBKE01` |
| 8 | 4 | Snapshot/business version (`uint32`) |
| 12 | 4 | Frames per second (`uint32`) |
| 16 | 8 | Scene + transition fingerprint (`uint64`) |
| 24 | 4 | Total frame count (`uint32`) |
| 28 | 4 | Committed frame count (`uint32`) |
| 32 | 4 | Bone count (`uint32`) |
| 36 | 4 | Vertex count (`uint32`) |
| 40 | 4 | Flags; bit 0 = complete |
| 44 | 4 | CRC32 of the complete file with this field zeroed |
| 48 | 16 | Reserved, zero |

## Frame record

```text
uint32 frameIndex
uint32 timeMs                 // llround(frameIndex * 1000 / fps)
float  bones[boneCount][12]   // row-major 3x4 affine matrices
float  vertices[vertexCount][2]
```

The 3x4 matrix layout is:

```text
m00 m01 m02 0
m10 m11 m12 0
0   0   1   0
```

The padding keeps each bone matrix at 48 bytes and makes GPU upload to a
`mat4x3`-style buffer straightforward. A renderer only needs the first three
columns of the first two rows for 2D point transformation.

Record size is `8 + boneCount * 48 + vertexCount * 8` bytes. The built-in 7-bone,
11-vertex sample therefore uses 432 bytes per frame. A complete 64-frame file is
`64 + 64 * 432 = 27712` bytes; the `.progress` file has the same layout while
committing and differs only in its frame count and complete flag.

## Checksum

CRC32 uses the reflected polynomial `0xedb88320`. The checksum field itself is
zeroed while calculating the checksum over the entire file. This covers the
header as well as the payload, so a damaged frame count or version field is
also detected.

## Load order

1. Read the entire file into memory.
2. Validate magic.
3. Read and compare the expected version, if supplied. A mismatch reports both
   `actualVersion` and `expectedVersion`.
4. Validate CRC32.
5. Validate flags, dimensions, declared byte length, frame contiguity, and finite
   float values.
6. Only then construct and return the `Snapshot`.

An error at any step throws before a partially populated snapshot escapes.

## Atomic progress

Writers use the same serialization for progress and final output:

1. Write all bytes to `path.tmp`.
2. Flush the file and call `fsync` (Windows `FlushFileBuffers`).
3. Atomically replace `path` (POSIX `rename`; Windows
   `MoveFileExW(MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)`).

The prior committed file remains intact if the process dies before step 3.
Every committed frame performs this sequence before the next frame is exposed.

