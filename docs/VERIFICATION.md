# Verification Record

Environment used for this run:

- Python 3.11.18
- NumPy 1.26.4
- opencv-python 4.10.0.84
- Linux x86_64 test host; code and in-repo PNG writer avoid platform-specific
  metadata and time stamps.

## Automated Tests

Command:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Result:

```text
Ran 8 tests in 3.892s
OK
```

Covered assertions:

- Two fresh runs over the same scene produce identical hashes for every file,
  including checkpoint state.
- `--chunk-rows 7` and `--chunk-rows 37` produce identical float caches,
  diagnostics, baked PNG and preview PNG.
- A run interrupted after two committed chunks exits 70; a second invocation
  reports 2 already-complete chunks, computes two new chunks, and its complete
  output tree is byte-identical to an uninterrupted run.
- Missing input returns 11 (`E001`), invalid scene field returns 14 (`E004`),
  RGBA normal returns 13 (`E003`), and a non-unit normal returns 15 (`E005`)
  with the offending pixel.
- Mirror, green-channel/UV and hard-edge seam fixtures produce localized
  anomaly entries.
- 16-bit output, custom naming, custom preview resolution and omitted scalar
  maps with scene defaults all work.
- Reversing scene light order leaves baked pixels and previews unchanged.
- Built-in TGA decoding handles RLE packets and bottom-left-origin images.

## Fixture Diagnostics

The generated clean 256x192 fixture reports:

- status `ok`;
- 0 normal-space anomalies;
- 192/192 seam boundary pairs classified `mirror_consistent`.

Injected cases:

| Fixture | Result |
|---|---|
| `seam-flip` | warning; normal-space mismatches localized; seam entries classified `mirror_seam` with handedness change |
| `uv-flip` | warning; localized `uv_mismatch` pixels and anomalous mirror-seam entries |
| `space-mismatch` | warning; localized tangent/object-space mismatch entries |
| `hard-edge` | warning; 101 seam pairs classified `hard_edge_seam` with high linear-color contrast |
| `non-unit` | exit 15, report `E005`, worst-pixel coordinate and affected-pixel count |

## Reproducibility Observation

Manual runs to separate output directories produced the same 25-file hash
tree. This includes `.work/state.json`, the chunk files, float caches, PNGs,
asset manifest, validation report and final run manifest.

The final manifest intentionally excludes run-time details such as a recovery
count, elapsed time or wall-clock date. Recovery information is observable in
stderr and in `.work/state.json` while the final artifact tree remains
deterministic.

## Known Boundaries

- UV islands and geometric axes are scene metadata because the task provides
  raster material maps and a scene description, not a mesh.
- The albedo-derived normal reference is intentionally a diagnostic heuristic;
  seam-pair checks use object-space paired normals directly.
- TGA 16-bit input is decoded through 5-5-5 quantization to 8-bit RGB. TGA
  24/32-bit and grayscale variants are supported.
