"""Sample-rate conversion with an FFmpeg/soxr front end and a NumPy fallback.

The FFmpeg path uses the SoX resampler (28-bit precision) when available and
falls back to the SWResampler windowed-sinc engine. If the ffmpeg binary is
missing entirely, a self-contained windowed-sinc (Kaiser window) resampler in
NumPy is used so the tool still works offline.

Every conversion reports the theoretical/actual output lengths and their
residual so the report can be audited sample-by-sample.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np

from .errors import ErrorCode, MixerError

_FFMPEG_CACHE: dict[str, object] = {"engine": None}


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _probe_soxr() -> str:
    if _FFMPEG_CACHE["engine"] is not None:
        return str(_FFMPEG_CACHE["engine"])
    if not ffmpeg_available():
        _FFMPEG_CACHE["engine"] = "numpy_windowed_sinc"
        return str(_FFMPEG_CACHE["engine"])
    probe = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-f", "lavfi", "-i",
            "anullsrc=r=48000:cl=mono:d=0.05",
            "-af", "aresample=resampler=soxr", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    _FFMPEG_CACHE["engine"] = "ffmpeg_soxr" if probe.returncode == 0 else "ffmpeg_swr"
    return str(_FFMPEG_CACHE["engine"])


def expected_output_length(input_frames: int, src_rate: int, dst_rate: int) -> int:
    return int(round(input_frames * dst_rate / src_rate))


def _resample_ffmpeg(mono: np.ndarray, src_rate: int, dst_rate: int, engine: str) -> np.ndarray:
    resampler = "soxr" if engine == "ffmpeg_soxr" else "swr"
    precision = ":precision=28" if resampler == "soxr" else ":filter_size=32:phase_shift=10"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "f32le", "-ar", str(src_rate), "-ac", "1", "-i", "pipe:0",
        "-af", f"aresample={dst_rate}:resampler={resampler}{precision}",
        "-f", "f32le", "-ac", "1", "pipe:1",
    ]
    proc = subprocess.run(
        command, input=mono.astype("<f4").tobytes(), capture_output=True, check=False
    )
    if proc.returncode != 0:
        raise MixerError(
            ErrorCode.INTERNAL,
            "ffmpeg resampling failed",
            {"stderr": proc.stderr.decode("utf-8", "replace")[:2000]},
        )
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float64)


def _resample_numpy(mono: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Band-limited interpolation with a Kaiser-windowed sinc (cutoff 0.9 pi)."""
    output_frames = expected_output_length(mono.shape[0], src_rate, dst_rate)
    ratio = dst_rate / src_rate
    # Half-width of the sinc kernel expressed in input samples.
    half_width = 32
    beta = 14.7696565
    cutoff = min(1.0, ratio) * 0.9
    out = np.zeros(output_frames, dtype=np.float64)
    # Chunked vectorised convolution keeps memory bounded for long assets.
    source_indices = np.arange(output_frames, dtype=np.float64) * (src_rate / dst_rate)
    for start in range(0, output_frames, 4096):
        block_idx = source_indices[start : start + 4096]
        center = np.floor(block_idx).astype(np.int64)
        frac = block_idx - center
        taps = np.arange(-half_width, half_width + 1, dtype=np.float64)
        # Distance from each output time to every candidate input sample.
        distances = taps[None, :] - frac[:, None]
        x = np.pi * cutoff * distances
        sinc = np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / np.where(x == 0, 1.0, x))
        window = np.i0(beta * np.sqrt(np.clip(1.0 - (distances / half_width) ** 2, 0.0, 1.0))) / np.i0(beta)
        kernel = cutoff * sinc * window
        sample_positions = center[:, None] + taps.astype(np.int64)[None, :]
        valid = (sample_positions >= 0) & (sample_positions < mono.shape[0])
        gathered = np.where(valid, np.clip(sample_positions, 0, mono.shape[0] - 1), 0)
        values = mono[gathered]
        out[start : start + block_idx.shape[0]] = np.sum(values * kernel * valid, axis=1)
    return out


def resample_mono(
    mono: np.ndarray, src_rate: int, dst_rate: int
) -> tuple[np.ndarray, dict]:
    """Resample a mono float64 signal and return ``(signal, info)``."""
    if src_rate == dst_rate:
        return mono.astype(np.float64, copy=True), {
            "src_rate": src_rate,
            "dst_rate": dst_rate,
            "engine": "passthrough",
            "input_frames": int(mono.shape[0]),
            "expected_output_frames": int(mono.shape[0]),
            "raw_output_frames": int(mono.shape[0]),
            "final_output_frames": int(mono.shape[0]),
            "length_residual_samples": 0,
            "trimmed_samples": 0,
            "padded_samples": 0,
        }

    engine = _probe_soxr()
    input_frames = int(mono.shape[0])
    expected = expected_output_length(input_frames, src_rate, dst_rate)

    if engine.startswith("ffmpeg"):
        converted = _resample_ffmpeg(mono, src_rate, dst_rate, engine)
    else:
        converted = _resample_numpy(mono, src_rate, dst_rate)

    raw_frames = int(converted.shape[0])
    trimmed = padded = 0
    if raw_frames > expected:
        converted = converted[:expected]
        trimmed = raw_frames - expected
    elif raw_frames < expected:
        converted = np.pad(converted, (0, expected - raw_frames))
        padded = expected - raw_frames

    info = {
        "src_rate": int(src_rate),
        "dst_rate": int(dst_rate),
        "engine": engine,
        "input_frames": input_frames,
        "expected_output_frames": expected,
        "raw_output_frames": raw_frames,
        "final_output_frames": int(converted.shape[0]),
        "length_residual_samples": raw_frames - expected,
        "trimmed_samples": trimmed,
        "padded_samples": padded,
        "ratio": dst_rate / src_rate,
    }
    return converted, info


def engine_description() -> str:
    return _probe_soxr()


def ffmpeg_version() -> str | None:
    if not ffmpeg_available():
        return None
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-version"], capture_output=True, text=True, check=False
    )
    return proc.stdout.splitlines()[0] if proc.stdout else None


def ffmpeg_ebur128(wav_path: str) -> dict:
    """Cross-check helper: run FFmpeg's ebur128 filter and parse its summary."""
    if not ffmpeg_available():
        raise MixerError(ErrorCode.INTERNAL, "ffmpeg unavailable for ebur128 cross-check")
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-i", wav_path,
            "-af", "ebur128=peak=true", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    summary: dict[str, float | None] = {
        "integrated_lufs": None,
        "sample_peak_dbfs": None,
        "true_peak_dbfs": None,
    }
    for line in proc.stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith("I:") and "LUFS" in stripped:
            summary["integrated_lufs"] = _parse_ebur_number(stripped)
        elif stripped.startswith("Peak:") and "dBFS" in stripped:
            value = _parse_ebur_number(stripped)
            if summary["sample_peak_dbfs"] is None:
                summary["sample_peak_dbfs"] = value
            else:
                summary["true_peak_dbfs"] = value
    return summary


def _parse_ebur_number(line: str) -> float | None:
    token = line.split()
    for idx, part in enumerate(token):
        if part in {"I:", "Peak:"}:
            for candidate in token[idx + 1 :]:
                try:
                    return float(candidate)
                except ValueError:
                    continue
    return None
