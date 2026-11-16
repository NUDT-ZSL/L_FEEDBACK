# Verification Record

Environment used for this run:

- Linux x86_64 development shell (same Python/ffmpeg implementation used for
  Windows execution)
- Python 3.14.4
- NumPy and soundfile available
- ffmpeg 8.0.1 / ffprobe 8.0.1

## Commands

```text
python tools/make_demo_assets.py --out artifacts/demo
python localize_dialogue.py --manifest artifacts/demo/manifest.json --target-language zh --output artifacts/demo/output_zh.wav --report artifacts/demo/report_zh.json
python localize_dialogue.py --manifest artifacts/demo/manifest.json --target-language en --output artifacts/demo/output_en.wav --report artifacts/demo/report_en.json
python tools/acceptance_check.py --output artifacts/demo/output_zh.wav --report artifacts/demo/report_zh.json --manifest artifacts/demo/manifest.json
python tools/acceptance_check.py --output artifacts/demo/output_en.wav --report artifacts/demo/report_en.json --manifest artifacts/demo/manifest.json
python -m unittest discover -s tests -v
```

## Observed Results

- Demo video: H.264, 1280x720, 30/1 fps, 10.000 s.
- Both output WAVs: 48,000 Hz, 2 channels, PCM16, 480,000 samples (10.000 s).
- Chinese timeline:
  - line 2 caused a 554 ms overflow shift;
  - line 3 was shifted and crossfaded over 10 ms;
  - line 4 was shifted by 594 ms total and truncated by 198 ms at 10.000 s.
- Chinese segment integrated loudness: approximately `-23.000`, `-23.000`,
  `-22.997`, `-23.000` LUFS.
- Chinese segment peaks: approximately `-19.28`, `-19.12`, `-19.19`,
  `-19.41` dBFS.
- English timeline: line 2 caused a 454 ms overflow shift; lines 3 and 4 were
  shifted, with line 4 ending at 9.908 s and remaining inside the video.
- English segment integrated loudness: approximately `-23.000`,
  `-23.000`, `-23.000`, `-23.000` LUFS.
- English segment peaks: approximately `-18.96`, `-19.18`, `-19.20`,
  `-19.24` dBFS.
- Automated acceptance checks: 29/29 passed for Chinese, 29/29 passed for
  English.
- Two consecutive Chinese runs produced byte-identical WAV files
  (`SHA-256 5594d0ec8b286d2c3294418ab8ad9aaa675c82511a6fb805417015c53c260eab`)
  and identical report values.
- `python -m unittest discover -s tests -v`: 16 tests passed.

The exact report values are in `artifacts/demo/report_zh.json` and
`artifacts/demo/report_en.json`; acceptance details are in
`artifacts/demo/acceptance_zh.json` and `artifacts/demo/acceptance_en.json`.
