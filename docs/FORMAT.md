# SKA1 asset format

All integer and floating-point values are little-endian. Matrices are stored
column-major. Strings are fixed 32-byte arrays, zero-padded, and must contain a
NUL terminator at the end of the used text.

## Header (32 bytes)

| Offset | Type | Field |
|---:|---|---|
| 0 | `char[4]` | magic `SKA1` |
| 4 | `u32` | version, currently `1` |
| 8 | `u32` | composition order |
| 12 | `f32` | frames per second |
| 16 | `u32` | frame count |
| 20 | `u32` | node count |
| 24 | `u32` | mesh count |
| 28 | `u32` | sample count |

Composition orders:

| Value | Order |
|---:|---|
| 0 | `T * R * S` |
| 1 | `T * S * R` |
| 2 | `R * T * S` |
| 3 | `R * S * T` |
| 4 | `S * T * R` |
| 5 | `S * R * T` |

## Node records (108 bytes each)

| Field | Type |
|---|---|
| name | `char[32]` |
| parent | `i32`, `-1` for a root |
| firstSample | `u32` |
| sampleCount | `u32` |
| inverseBind | `f32[16]`, column-major |

The reader permits arbitrary physical ordering of sample records as long as
every node's `[firstSample, firstSample + sampleCount)` range is in bounds,
ranges are disjoint and every sample is claimed. Times within each node range
must be finite and strictly increasing.

## Sample records (44 bytes each)

| Field | Type |
|---|---|
| time | `f32` seconds |
| translation | `f32[3]` |
| rotation | `f32[4]`, order `x,y,z,w` |
| scale | `f32[3]` |

Quaternions may have any positive length. They are normalized before use.
A zero or non-finite quaternion rejects the asset.

## Mesh records

Mesh prefix:

| Field | Type |
|---|---|
| name | `char[32]` |
| vertexCount | `u32` |
| indexCount | `u32` |

Vertex records (72 bytes each):

| Field | Type |
|---|---|
| position | `f32[3]` |
| normal | `f32[3]` |
| color | `f32[3]` |
| boneIndices | `u32[4]` |
| weightCount | `u32`, in `[1,4]` |
| weights | `f32[4]` |

Weights outside `[0, weightCount)` must be zero. All active weights must be
finite and non-negative, and their sum must be positive. The reader preserves
the four bone slots and the original topology. Unitization happens during
frame evaluation and is counted in the report.

`indexCount` must be divisible by three. Each index must be less than the
mesh vertex count. There are exactly `indexCount / 3` triangles.

## Safety limits

The reader rejects assets over 512 MiB, more than 65,536 nodes, 4,194,304
samples, 256 meshes, 4,000,000 vertices per mesh or 24,000,000 indices per
mesh. It also rejects trailing bytes, truncated records, non-finite numeric
attributes, malformed track ranges, duplicate track claims, parent cycles and
out-of-range bone/triangle indices.

## Stable error codes

| Code | Name |
|---:|---|
| 10 | `OPEN_FAILED` |
| 11 | `READ_FAILED` |
| 12 | `INVALID_MAGIC` |
| 13 | `UNSUPPORTED_VERSION` |
| 14 | `TRUNCATED` |
| 15 | `INVALID_COUNT` |
| 16 | `INVALID_NODE_INDEX` |
| 17 | `INVALID_PARENT_INDEX` |
| 18 | `PARENT_CYCLE` |
| 19 | `INVALID_TRACK_NODE` |
| 20 | `NON_MONOTONIC_TIME` |
| 21 | `INVALID_NUMERIC_VALUE` |
| 22 | `ZERO_QUATERNION` |
| 23 | `INVALID_WEIGHT_COUNT` |
| 24 | `INVALID_WEIGHT` |
| 25 | `WEIGHT_SUM_ZERO` |
| 26 | `INVALID_BONE_INDEX` |
| 27 | `INVALID_TRIANGLE_INDEX` |
| 28 | `INVALID_COMPOSITION_ORDER` |
| 29 | `INVALID_JSON` |
| 30 | `RESOURCE_TOO_LARGE` |
| 31 | `WRITE_FAILED` |
| 32 | `INVALID_FRAME_RANGE` |
| 33 | `INVALID_ARGUMENT` |
| 50 | `OPENGL_UNAVAILABLE` |
| 70 | `NAN_PRODUCED` |

On parsing failure, the output `Asset` is reset before return. No partially
loaded state is exposed.

