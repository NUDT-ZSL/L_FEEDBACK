# voice-replay design notes

## Pipeline

1. Parse and validate `events.json` and every commentary metadata file.
2. Hash all referenced input bytes, sort tracks by commentary ID, derive the
   event-anchor offsets, and build output names.
3. Reuse a complete manifest if all output digests still match.
4. Otherwise create/resume WAVs and process blocks in fixed 1024-frame units.
5. For each block, map each track's raw range through its global offset into
   the chunk. Mix unclipped floating-point contributions, quantize the master,
   quantize each stem, flush, then atomically checkpoint state.
6. Hash the completed WAVs, atomically commit the manifest, and remove state.

## Why output-prefix digests are enough

The state is written only after all WAVs have been flushed for a completed
block. A restart can therefore:

1. truncate any bytes beyond the saved block boundary;
2. verify the 44-byte canonical WAV header;
3. re-hash the saved PCM prefix;
4. compare it with the state's monotonic SHA-256;
5. seed the same incremental SHA-256 for subsequent checkpoints.

This catches both changed input and output-file corruption. It also makes the
resume path independent of wall-clock time, temporary block-file names, or
process buffering details.

## Commit order

For every block:

```text
write master and all stems -> flush all streams -> atomic state replace
```

For completion:

```text
close WAVs -> hash WAVs -> atomic manifest replace -> remove state
```

If a process dies before the state replace, the next resume truncates the
unacknowledged bytes and recomputes that block. If it dies after the state
replace, the checkpoint is accepted. If it dies after the manifest commit but
before state removal, a subsequent run can reuse the manifest and remove the
stale state.

## Edge cases

- A source range is intersected with both the validated available raw length
  and the match timeline before it is read.
- The final partial block writes only the remaining valid frames; no padding
  is included in the WAV data chunk or manifest frame count.
- A track with no available frames creates no contribution to the master, but
  its stem writer receives a zero block for the full timeline.
- Markers do not cut audio. They provide the anchor used to compute the one
  signed global offset that is applied consistently inside every chunk.
- Clipping is tracked separately for isolated stems and the stereo master.
- Floating-point summation order is fixed by commentary-ID order, and no
  parallel reduction or fast-math option is used.

## Security and locality

The tool has no network, GPU, container, or external-library dependency. It
only reads the declared input files/referenced raw files and writes to the
declared output directory. Manifest file paths are relative and rejected if
they contain an absolute path or `..` component during reuse validation.

