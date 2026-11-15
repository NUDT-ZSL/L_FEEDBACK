# Validation Record

Commands used:

```bash
python tools/generate_demo.py --out demo_project
python -m dialogue_baker bake \
  --project demo_project/project.json \
  --out demo_output \
  --force
python -m unittest discover -s tests -v
```

The test suite covers:

- exact integer round-trip at 24, 25, 60, 23.976, and 29.97 fps;
- full five-line bake with four overlapping speaker pairs;
- 44.1 kHz and stereo-input resampling paths;
- per-frame spatial gain/pan/Doppler records;
- pre-limiter clipping counters and safest true-peak limiting;
- per-frame subtitle activity derived from half-open sample intervals;
- byte-identical WAV/report/subtitle rebuilds;
- resume after `parse`/`spatialize` snapshots;
- snapshot version and file-digest rejection;
- missing-file, FPS-mismatch, missing-track, frame-range, and same-speaker
  overlap business errors;
- CLI business-error JSON and `bake_error.json`.

Automated result:

```text
Ran 7 tests in about 5s

OK
```

## Demo Bake Metrics

Observed report values for the deterministic local fixture:

- Duration: 120 frames / 240000 samples / 5.000 seconds.
- Timeline boundary round-trip: `true`.
- Subtitle alignment check: `true` for all 5 lines and 120 frames.
- Overlap pairs: 4.
- Integrated loudness before processing: `-26.811880120 LUFS`.
- Requested linear gain to `-10.0 LUFS`: `+16.811880120 dB`.
- Pre-limiter samples over full scale after gain: `40`.
- Limiter active samples: `134202`.
- Maximum limiter reduction: `1.695721257 dB`.
- Final true-peak safety trim: `-0.065350878 dB`.
- Final integrated loudness after limiting: `-10.183067621 LUFS`.
- Encoded true peak: `-1.049865239 dBTP` (ceiling `-1.0 dBTP`).
- Encoded format: `pcm_s16le`, 48000 Hz, 2 channels, 16 bits.
- Final post-limiter clipped samples: `0`.

The target is intentionally above the sustainable peak ceiling for this fixture,
so the report correctly shows a `-0.183 LU` target miss while maintaining the
true-peak limit.

## Reproducibility Check

Two clean builds into different output directories produced identical bytes:

```text
dialogue.wav          b6796034272cefe56bdcc3f6175e0fcd74a5a32e6ad160731ec5a129df446da6
bake_report.json      32d892be6b8197850a00f18982af1ea53861d6b46e6723e2b0435e1fd5d3b690
subtitles_frames.json 9011ee62e6945c247127443376b2824464448a9eaaa089fa9550e03b26d98b48
```

The hashes above are from the final verified build. A second clean build into a
different output directory was byte-identical.

Snapshot resume check:

```text
first run  -> parse=executed, spatialize=executed, status=stopped
second run -> parse=reused, spatialize=reused, mix=executed,
              loudness=executed, package=executed, status=complete
```

The resumed WAV and report were byte-identical to the clean build.

