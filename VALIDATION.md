# Validation Record

Runtime used for the recorded runs:

- Python 3.14.4
- NumPy 2.5.4
- soundfile 0.14.0
- FFmpeg / ffprobe 7.0.2 static build with libopus
- Linux/WSL execution environment; all runtime paths and subprocess calls are
  Windows-compatible, and `scripts/run_validation.ps1` is the Windows entry.

## Automated Tests

Command:

```text
python -m unittest discover -v
```

Result: `Ran 9 tests ... OK`

Coverage includes config rejection, deterministic packet generation, malformed
payload parsing, duplicate/timeout/loss classification, pitch PLC and fixed
timeline length, fade boundaries, report reconstruction, FFmpeg verification,
and binary deterministic reruns.

## Default Lossy Scenario

Commands:

```text
python -m radio_pipeline run --config config/default.json --output-dir artifacts/default
python -m radio_pipeline report --run-dir artifacts/default --output artifacts/default/report_from_log.json
python -m radio_pipeline verify --run-dir artifacts/default
```

Observed results:

| Metric | Value |
| --- | ---: |
| Source packets | 400 |
| Duration | 8000 ms |
| Expected/decoded samples | 384000 / 384000 |
| Received | 307 |
| Lost | 34 |
| Timeout | 59 |
| Duplicate arrivals | 13 |
| Invalid-payload arrivals | 1 |
| PLC frames | 93 (35 pitch, 58 comfort) |
| Effective lost/PLC ratio | 0.2325 |
| Max buffer watermark | 120 ms |
| Fast-consumption events | 0 |
| Conservative-mode events | 1 at sequence 238 |
| Report loudness | -20.998 LUFS-approx |
| WAV loudness | -20.998 LUFS-approx |
| Ogg loudness | -21.007 LUFS-approx |
| WAV sample error | 0 samples |
| Ogg decoded sample error | 0 samples |
| WAV check | `pcm_s16le`, 48000 Hz, mono, 384000 samples |
| Ogg check | `opus`, 48000 Hz, mono, 384000 decoded samples |
| report rebuild | data equal to `report.json` |
| verify exit code | `0` |

The conservative-mode event records the buffer depth increase from 160 ms to
320 ms and PLC aggression reduction from 1.0 to 0.45 after a 9-frame loss run
exceeded the 8-frame initial depth.

## Extreme Jitter Scenario

Commands:

```text
python -m radio_pipeline run --config config/extreme.json --output-dir artifacts/extreme
python -m radio_pipeline report --run-dir artifacts/extreme --output artifacts/extreme/report_from_log.json
python -m radio_pipeline verify --run-dir artifacts/extreme
```

Observed results:

| Metric | Value |
| --- | ---: |
| Source packets | 300 |
| Duration | 6000 ms |
| Expected/decoded samples | 288000 / 288000 |
| Received | 108 |
| Lost | 39 |
| Timeout | 153 |
| Duplicate arrivals | 23 |
| Invalid-payload arrivals | 1 |
| PLC frames | 192 (172 pitch, 20 comfort) |
| Effective lost/PLC ratio | 0.640000 |
| Max buffer watermark | 200 ms |
| Fast-consumption events | 3, total trim 60 ms |
| Conservative-mode events | 1 at sequence 191 |
| Report loudness | -22.902 LUFS-approx |
| WAV loudness | -22.902 LUFS-approx |
| Ogg loudness | -22.896 LUFS-approx |
| WAV/Ogg sample error | 0 / 0 samples |
| report rebuild | data equal to `report.json` |
| verify exit code | `0` |

This scenario exercises over-watermark fast consumption, a long burst loss,
adaptive PLC strategy switching, and graceful conservative degradation while
retaining the full 6.000 s timeline.

## Clean Sequential Scenario

Commands:

```text
python -m radio_pipeline run --config config/sequential.json --output-dir artifacts/clean
python -m radio_pipeline verify --run-dir artifacts/clean
```

Observed: 100/100 packets received, 0 PLC frames, 96000/96000 decoded samples
in both formats, report/WAV loudness `-19.793`, Ogg `-19.780`, sample errors
`0`, verify exit code `0`.

## Determinism

`config/default.json` was run twice. SHA-256 values were identical:

```text
WAV: 1aaec65d0e8dfdb2cc79cbb721a84f33619da59850074a111bb72182877ce531
Ogg: 4a4d499c9bad6dca89a470dd9630c4c118e7f856d5c8f7f5ca1941c1867ce56d
PCM: d5a7b43d40fe9bc33cf079a8b919c7eb3a61a634c6114557b185d80dc3353c4a
```

Generated `input_packets.jsonl` was also identical:

```text
3bcab1c08e7d6afc0bbe9a46de40f67d7e85b4bf8e3aa5dbee089199036dd3d2
```

## Exit-Code Checks

- Missing config / invalid parameter: exit `2`
- Missing processing log for `report`: exit `3`
- Tampered WAV in a copied run directory: `verify` exit `3`, failing check
  `wav_hash`
- Clean output: `run`, `report`, and `verify` exit `0`
