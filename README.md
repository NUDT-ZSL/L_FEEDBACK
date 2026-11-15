# automix

`automix` is a Windows command-line, offline automotive audio mixer. It reads
one JSON configuration plus up to eight local audio files, resamples all inputs
to 48 kHz, mixes them to stereo, applies track/master gain, sample-level DRC, a
lookahead peak limiter, and BS.1770-4 loudness/true-peak measurement, then
writes a deterministic 48 kHz stereo float WAV and a JSON statistics report.

No device, account, or network resource is used at runtime. The included
Python sample generator is deterministic and uses only the standard library.

## Build on Windows

Prerequisites:

- Visual Studio 2022 or Build Tools with the C++17 workload
- CMake 3.20 or newer
- libsndfile 1.2.0 Windows binaries, with `LIBSNDFILE_DIR` pointing to the
  prefix containing `include/sndfile.h`, `lib/*.lib`, and `bin/*.dll`
- FFTW 3.3.10 Windows DLL package, with `FFTW_DIR` pointing to the prefix
  containing `fftw3.h`, the import libraries, and the FFTW DLLs

```bat
set LIBSNDFILE_DIR=C:\deps\libsndfile-1.2.0
set FFTW_DIR=C:\deps\fftw-3.3.10

cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

CMake copies the libsndfile/FFTW DLLs next to the executables. FFTW is used by
the spectral resampler verification test; the production DSP uses the specified
BS.1770 FIR and a deterministic polyphase Kaiser-windowed sinc converter.

## Run

```bat
automix.exe --config examples\config.json
automix.exe --config examples\config.json --log-dir C:\logs\automix
```

Generate the deterministic samples first:

```bat
python scripts\generate_samples.py --out-dir examples\samples
```

The example config resolves its paths relative to `examples/`.

## Processing behavior

- Read support: WAV/WAVEX/RF64/W64, AIFF, AU, and FLAC; PCM16, PCM24, and
  IEEE-float32 samples; mono or stereo; any libsndfile-supported source rate.
- Resampling: Kaiser-windowed band-limited polyphase interpolation, normalized
  per phase and designed for 90 dB attenuation. Measured 44.1-to-48 kHz
  passband error was below 0.0003 dB and 96-to-48 kHz alias rejection was
  110.8 dB.
- Mixing: deterministic sequential accumulation; mono tracks use equal-power
  `sqrt(0.5)` left/right coefficients; mute and solo are supported.
- DRC: instantaneous stereo-linked peak detection, quadratic soft knee,
  sample-level attack/release smoothing.
- Limiter: offline future-minimum lookahead envelope with linked stereo gain and
  release smoothing. The final float output is clamped to the configured
  ceiling, so every output sample satisfies `abs(sample) <= threshold`.
- Loudness: ITU-R BS.1770-4 K-weighting, 400 ms blocks with 75% overlap,
  absolute and relative gating, and the standard 48 kHz four-phase 12-tap true
  peak interpolator.
- Output: `WAV/FLOAT_32`, 48 kHz, stereo. Float output preserves the limiter
  ceiling exactly and avoids PCM quantization changing the measured levels.

## Exit codes and errors

Errors are printed to stderr as `<CODE>: <message>`.

| Business code | Exit | Meaning |
| --- | ---: | --- |
| `E_USAGE` | 2 | Invalid command line. |
| `E_CFG_SYNTAX` | 2 | Malformed JSON; no audio processing starts. |
| `E_CFG_MISSING` | 2 | Missing required field; no audio processing starts. |
| `E_CFG_VALUE` | 2 | Field outside its allowed range or inconsistent. |
| `E_FILE_MISSING` | 3 | Input (or configuration) file does not exist. |
| `E_FILE_FORMAT` | 3 | Unsupported container/subtype/channel count/rate. |
| `E_FILE_CORRUPT` | 3 | Truncated or invalid sample data. |
| `E_OUTPUT_IO` | 4 | Output/report could not be committed. |
| `E_INTERNAL` | 5 | Unexpected internal failure. |

All tracks are validated/rendered before output files are written. A track
failure aborts the entire run with exit code 3, removes temporary files, and
never commits a partial WAV. Logs produced up to the failure are retained.

## Logging and reports

Logs include startup arguments, per-track read/resample/gain/DRC summaries,
limiter/loudness results, and stage timings. The active log rotates before it
would exceed 10 MiB and retains the active file plus two archives (three files
total). `SFC_SET_ADD_PEAK_CHUNK` is disabled on output so repeated runs are
byte-identical. If the log directory is not writable, the tool prints one
stderr warning, continues to render, and still commits the WAV/report.

See `docs/CONFIG.md` for the complete schema and `docs/TEST_RESULTS.md` for
actual measurements.
