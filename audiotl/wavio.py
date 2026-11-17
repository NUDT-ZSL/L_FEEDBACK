"""Deterministic WAV PCM I/O.

Reads PCM integer WAV (8/16/24/32-bit) into float32 arrays shaped
(frames, channels), normalized to [-1.0, 1.0). Writes 16-bit PCM WAV.
Quantization is round-half-to-even via numpy.rint (IEEE deterministic).
"""

import os
import wave

import numpy as np

from .errors import BusinessError, ErrorCode, ExitCode

_INT_SCALE = {1: 128.0, 2: 32768.0, 3: 8388608.0, 4: 2147483648.0}


def read_wav(path):
    """Return (data float32 (frames, channels), sample_rate)."""
    try:
        wf = wave.open(str(path), "rb")
    except FileNotFoundError:
        raise BusinessError(
            ErrorCode.E_INPUT_MISSING,
            f"input file not found: {path}",
            ExitCode.INPUT_MISSING,
        )
    except (wave.Error, EOFError, OSError) as exc:
        raise BusinessError(
            ErrorCode.E_UNSUPPORTED_FORMAT,
            f"cannot parse WAV file {path}: {exc}",
            ExitCode.RENDER_ERROR,
        )
    with wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if channels not in (1, 2):
        raise BusinessError(
            ErrorCode.E_UNSUPPORTED_FORMAT,
            f"unsupported channel count {channels} in {path} (only mono/stereo)",
            ExitCode.RENDER_ERROR,
        )
    if sampwidth not in _INT_SCALE:
        raise BusinessError(
            ErrorCode.E_UNSUPPORTED_FORMAT,
            f"unsupported sample width {sampwidth} bytes in {path} (PCM int only)",
            ExitCode.RENDER_ERROR,
        )

    if sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        v = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        v = np.where(v & 0x800000, v - 0x1000000, v)
        data = v.astype(np.float32) / _INT_SCALE[3]
    else:
        dtype = "<i2" if sampwidth == 2 else "<i4"
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32) / _INT_SCALE[sampwidth]

    return data.reshape(-1, channels), rate


def quantize_int16(data):
    """float (frames, channels) in [-1,1] -> int16, deterministic rounding."""
    clipped = np.clip(data, -1.0, 1.0)
    return np.rint(clipped * 32767.0).astype("<i2")


def write_wav16(path, data, rate):
    """Write float (frames, channels) data as 16-bit PCM WAV."""
    q = quantize_int16(data)
    channels = q.shape[1] if q.ndim == 2 else 1
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(q.tobytes())


def write_wav16_atomic(path, data, rate):
    """Write WAV via temp file + os.replace so readers never see a partial file."""
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        write_wav16(tmp, data, rate)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
