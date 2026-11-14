# Design Notes

## Density Field

Each voxel is a pure function of `(x,y,z,seed,cloudType)`. The field combines a
four-octave 3D Perlin FBM with a three-octave Worley F1 FBM. All lattice hashes are
integer avalanche hashes; no standard-library random generator is used.

Cumulus applies a radial x/z envelope, a noise-modulated billowy top, and a
low-altitude base. Stratus uses a flat vertical band and an x/z edge fade. The
result is quantized to R8 with round-to-nearest.

The full field is generated as independent z-slices. Worker tasks write disjoint
memory and never perform a cross-slice reduction, so changing worker count or
switching to synchronous execution cannot change a byte. Downsampling uses fixed
2x2x2 integer sums and `(sum + 4) / 8` rounding.

## Frame Pipeline

1. Build a camera ray for each output pixel from the inverse view-projection.
2. Intersect the unit cloud volume and march fixed-count ray samples.
3. At each nonzero density sample, march toward the sun and integrate optical
   depth.
4. Accumulate single-scattered radiance with Beer-Lambert transmittance and
   composited sky.
5. For scale 2, render a half-resolution color/depth buffer. Upsample with a
   foreground-biased depth test and bilateral spatial weights.
6. Reproject the previous accumulation through the previous camera matrices.
7. Blend current and historical color; on a disocclusion miss use zero history
   weight. History weight is also capped by the source sample count.
8. Increment every count by exactly one and tone-map the accumulated HDR color to
   RGBA8.

The renderer never parallelizes ray traversal or temporal accumulation, keeping
the floating-point operation order fixed.

## Reprojection and Counts

The state stores only the previous frame's camera matrices and depth, because the
next update needs exactly one previous frame. A cloud hit is accepted when the
current world position projects inside the previous frame and the reconstructed
previous depth point is within a distance-scaled tolerance. Sky-to-sky projection
is accepted; cloud/sky mismatches are disocclusions.

The accumulation count is a cumulative frame sample count, not a count of
successful history hits. It therefore increases by one for every pixel on every
serial frame, including disocclusions. Separately, every frame logs ray attempts,
hits, misses, and `hits / attempts`; `hits + misses == attempts`.

## Async Synchronization

`WorkerPool::waitIdle()` waits on a `std::condition_variable` until the submitted
task count reaches zero. There is no polling loop. Logs record
`mainThreadSpinMilliseconds: 0`, `waitEvent: "condition_variable"`, and worker
count. The pool destructor sets its stop flag, notifies all workers, and joins
them.

## Snapshot Continuity

All state needed for the next frame is serialized with exact float bit patterns:
previous matrices, previous depth, accumulation color, counts, density level,
seed, shape, history length, and next lighting. On restore, density is generated
again from the same integer seed and level, so no texture payload is needed. The
next camera is the deterministic frame `snapshot.frameIndex + 1`.

The acceptance harness stops after v2 frame 1, starts a fresh process from the
snapshot, and compares the final RGBA8 bytes against a three-frame uninterrupted
run. The observed relative error is zero.

## Resource Failure Chain

Memory budgeting happens before allocation. If all levels fail, the process
throws `E_OUT_OF_MEMORY`; a host `std::bad_alloc` is mapped to the same business
code. Snapshot open failures, CRC failures, and unknown versions are raised
before rendering. The main catch block logs the JSON business record, the stack
unwinds through all RAII worker/buffer owners, and the process returns 10, 11,
12, or 13 as appropriate.

