"""Strict PCM16 mono WAV reader/writer using only the standard library.

Validation rejects (never silently truncates or pads):
  * missing RIFF/WAVE signature
  * unknown/duplicate chunks or illegal chunk lengths
  * non-PCM (wFormatTag != 1) or non-16-bit or non-mono files
  * data bytes that are not an exact multiple of the frame size
"""

import hashlib
import struct
import wave

from .errors import WavFormatError

WAVE_FORMAT_PCM = 1
PCM16_BITS = 16
PCM16_CHANNELS = 1
PCM16_FRAME_BYTES = 2  # mono, 16-bit


class WavData:
    __slots__ = ("sample_rate", "samples", "path", "data_bytes")

    def __init__(self, sample_rate, samples, path=None, data_bytes=None):
        self.sample_rate = sample_rate
        self.samples = samples  # list[int] in signed PCM16 range
        self.path = path
        self.data_bytes = data_bytes


def _read_exact(f, n, what, path):
    buf = f.read(n)
    if len(buf) != n:
        raise WavFormatError(
            "unexpected end of file while reading %s" % what,
            details={"path": path, "wanted": n, "got": len(buf)},
        )
    return buf


def read_wav(path):
    """Parse and validate a PCM16 mono WAV, returning WavData."""
    try:
        f = open(path, "rb")
    except OSError as exc:
        raise WavFormatError(
            "cannot open wav: %s" % exc,
            code="E_WAV_OPEN",
            details={"path": path},
        )

    with f:
        size = _file_size(f)
        if size < 12:
            raise WavFormatError("file smaller than RIFF header", details={"path": path})

        riff = _read_exact(f, 4, "RIFF id", path)
        if riff != b"RIFF":
            raise WavFormatError("missing RIFF signature", details={"path": path})
        riff_size = struct.unpack("<I", _read_exact(f, 4, "RIFF size", path))[0]
        wave_id = _read_exact(f, 4, "WAVE id", path)
        if wave_id != b"WAVE":
            raise WavFormatError("missing WAVE signature", details={"path": path})
        if riff_size != size - 8:
            raise WavFormatError(
                "illegal RIFF chunk length",
                details={"path": path, "header": riff_size, "actual": size - 8},
            )

        fmt = None
        data_bytes = None
        fact_samples = None
        seen = set()
        # Walk every chunk; the declared RIFF size bounds the chunk region.
        while f.tell() < size:
            chunk_id = f.read(4)
            if len(chunk_id) == 0:
                break
            if len(chunk_id) != 4:
                raise WavFormatError("truncated chunk id", details={"path": path})
            if chunk_id in seen:
                raise WavFormatError(
                    "duplicate chunk %r" % chunk_id, details={"path": path}
                )
            seen.add(chunk_id)
            chunk_len = struct.unpack(
                "<I", _read_exact(f, 4, "chunk length", path)
            )[0]
            payload_start = f.tell()
            if payload_start + chunk_len > size:
                raise WavFormatError(
                    "illegal chunk length for %r" % chunk_id,
                    details={"path": path, "chunk": chunk_id.decode("latin1")},
                )

            if chunk_id == b"fmt ":
                fmt = _read_exact(f, chunk_len, "fmt payload", path)
            elif chunk_id == b"fact":
                fact_payload = _read_exact(f, chunk_len, "fact payload", path)
                if len(fact_payload) >= 4:
                    fact_samples = struct.unpack("<I", fact_payload[:4])[0]
            elif chunk_id == b"data":
                data_bytes = _read_exact(f, chunk_len, "data payload", path)
            else:
                # Unknown chunks are tolerated only if fully bounded.
                _read_exact(f, chunk_len, "chunk payload", path)

            # Chunks are word-aligned; a single pad byte follows odd lengths.
            if chunk_len & 1:
                pad = f.read(1)
                if len(pad) != 1:
                    raise WavFormatError("missing chunk pad byte", details={"path": path})

        if fmt is None:
            raise WavFormatError("missing fmt chunk", details={"path": path})
        if data_bytes is None:
            raise WavFormatError("missing data chunk", details={"path": path})
        if len(fmt) < 16:
            raise WavFormatError("fmt chunk too small", details={"path": path})

        format_tag, channels, sample_rate, _byte_rate, _block_align, bits = (
            struct.unpack("<HHIIHH", fmt[:16])
        )
        if format_tag != WAVE_FORMAT_PCM:
            raise WavFormatError(
                "non-PCM format tag %d" % format_tag, details={"path": path}
            )
        if channels != PCM16_CHANNELS:
            raise WavFormatError(
                "expected mono, got %d channels" % channels, details={"path": path}
            )
        if bits != PCM16_BITS:
            raise WavFormatError(
                "expected 16-bit PCM, got %d bits" % bits, details={"path": path}
            )
        if sample_rate <= 0:
            raise WavFormatError("illegal sample rate", details={"path": path})

        if len(data_bytes) % PCM16_FRAME_BYTES != 0:
            raise WavFormatError(
                "data length not a whole number of mono PCM16 frames",
                details={"path": path, "data_bytes": len(data_bytes)},
            )

        count = len(data_bytes) // PCM16_FRAME_BYTES
        # A fact chunk, when present for PCM, must agree with the frame count.
        if fact_samples is not None and fact_samples != count:
            raise WavFormatError(
                "fact sample count does not match data frames",
                details={
                    "path": path,
                    "fact_samples": fact_samples,
                    "data_frames": count,
                },
            )
        samples = list(struct.unpack("<%dh" % count, data_bytes))
        return WavData(sample_rate, samples, path=path, data_bytes=data_bytes)


def _file_size(f):
    pos = f.tell()
    f.seek(0, 2)
    size = f.tell()
    f.seek(pos)
    return size


def write_wav(path, sample_rate, samples):
    """Write deterministic PCM16 mono WAV (no LIST/INFO chunks)."""
    if sample_rate <= 0:
        raise WavFormatError("illegal sample rate on write", details={"path": path})
    frame_bytes = len(samples) * PCM16_FRAME_BYTES
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + frame_bytes))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(
            struct.pack(
                "<HHIIHH",
                WAVE_FORMAT_PCM,
                PCM16_CHANNELS,
                sample_rate,
                sample_rate * PCM16_FRAME_BYTES,
                PCM16_FRAME_BYTES,
                PCM16_BITS,
            )
        )
        f.write(b"data")
        f.write(struct.pack("<I", frame_bytes))
        packed = struct.pack("<%dh" % len(samples), *samples)
        f.write(packed)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def bytes_sha256(data):
    return hashlib.sha256(data).hexdigest()


# Backwards-compatible re-export for callers that prefer the stdlib wave size.
def wave_sample_count(path):
    with wave.open(path, "rb") as w:
        return w.getnframes()
