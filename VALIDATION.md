# Validation Record

## Environment

- Date: 2026-11-15
- Python: 3.11.18
- NumPy: 2.4.6 primary; compatibility suite also passed with 1.24.4
- Platform: Linux x86_64 (also uses only cross-platform Python + NumPy APIs)

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/generate_samples.py --out-dir samples
python -m telemetry_synth.cli \
  --telemetry samples/telemetry.json \
  --params samples/params.json \
  --output artifacts/run1/race.wav \
  --seed 20261115 \
  --block-size 4096
python -m unittest discover -s tests -v
```

## Generated Inputs

| File | Bytes | SHA-256 |
|---|---:|---|
| `samples/telemetry.json` | 404789 | `3e181d76e5c9f0fcd819a01106805f84e31512ca5f2a0c3d9181d34b621a79b1` |
| `samples/params.json` | 2898 | `bda67e6fd503615dafc9d72989d4a6029ff3cc9220bef76c22fecd3ff4185a11` |

样本包含 2400 帧（12.000 秒）、3 次 RPM 阶跃换挡、4 种路面、16 次滑移状态切换和 13 个离散事件。其中 3 个事件落入重叠时间窗并触发衰减。

## Audio Result

`artifacts/run1/race.wav`

| Field | Value |
|---|---|
| RIFF/WAVE | valid |
| Format | PCM, 2 channels, 48000 Hz, 16 bit |
| Byte rate / block align | 192000 / 4 |
| Frame count | 576000 |
| Data bytes | 2304000 |
| File bytes | 2304044 |
| SHA-256 | `6d5bbe87d70675a950ba5800fd70ea73db0875491e291974a221b5d0ca9f077f` |

Alignment check:

```text
round(2400 frames * 0.005 s * 48000) = 576000 samples
2400 frames * 240 samples/frame = 576000 samples
WAV data chunk = 576000 * 2 channels * 2 bytes = 2304000 bytes
```

Header validation through `wave` and an independent RIFF chunk parser both passed. Decoded PCM remained within `[-32768, 32767]`.

## Report Result

`artifacts/run1/race.report.json` contains:

```text
frame_count               = 2400
written_sample_count      = 576000
sample_count_match        = true
bus_peaks.engine          = 1.0362981439312198
bus_peaks.tire            = 0.2333358860587017
bus_peaks.events          = 0.37492864436193934
pre_limit_peak            = 0.7805020582451854
post_limit_peak           = 0.7805020582451854
event_trigger_count       = 13
events_attenuated         = 3
engine_shift_count        = 3
tire_state_switch_count   = 16
```

The first successful run emitted:

```text
peak_memory_estimate_bytes=8224640 tracemalloc_peak_bytes=3729358 process_peak_rss_bytes=168448000 method=modeled_streaming_working_set
```

## Determinism

- Two runs with seed `20261115` and block size `4096` produced identical WAV SHA-256:
  `6d5bbe87d70675a950ba5800fd70ea73db0875491e291974a221b5d0ca9f077f`.
- The full 19-test suite passed under both NumPy 2.4.6 and the declared minimum
  NumPy 1.24.4. A 1.24.4-generated WAV also matched the 2.4.6 SHA-256 exactly.
- A run with block size `257` produced the same byte-for-byte WAV, confirming filter state, phase state, event envelopes and index-addressed noise survive block boundaries.
- A run with seed `7` produced a different WAV SHA-256:
  `d88d9c6b584a37855d923f17c44b2d61fd2574b9ab3bd0f89047425a52cf2bdc`.

## Failure-Path Checks

Automated tests verified nonzero exits and no generated WAV/report for:

- Removed frame or frame-index/time gap: `E_TELEMETRY_MISSING_FRAME`.
- Duplicate/non-monotonic timestamp: `E_TELEMETRY_TIMESTAMP`.
- RPM out of range or unknown surface: `E_TELEMETRY_RANGE`.
- Malformed JSON: `E_TELEMETRY_JSON`.
- Invalid parameter type: `E_PARAMS_SCHEMA`.
- Block size zero: `E_USAGE`.

An existing sentinel output file remained unchanged after a failed validation run. No hidden `*.tmp` files remained after successful or failed runs.

## Automated Test Result

```text
Ran 19 tests in ~20 s
OK
```

The test suite includes full 12-second synthesis plus reproducibility and alternate block-size runs. See `tests/test_pipeline.py`.

## Notes

- The event overlap policy is deterministic start-window ducking, not a full polyphonic loudness normalizer.
- Tire noise uses a time-varying one-pole low-pass; engine harmonics above the anti-alias guard band fade smoothly.
- All output is model-based; no external samples, network access, Docker, or special hardware are used.

