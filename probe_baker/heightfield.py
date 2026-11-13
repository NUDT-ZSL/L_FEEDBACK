"""Heightfield binary format: deterministic generator + validating reader.

Layout (little-endian):
  magic          8 bytes  b"HFPROBE1"
  format_version uint32   (= 1)
  width          uint32
  height         uint32
  reserved       uint32   (= 0)
  version_hash   16 bytes blake2b(data, digest_size=16)
  data           width*height float32, row-major, values in [0, 1]
"""
import hashlib
import struct

import numpy as np

from .errors import HeaderMismatch, HeightInvalid

MAGIC = b"HFPROBE1"
FORMAT_VERSION = 1
HEADER_STRUCT = struct.Struct("<8sIIII16s")
HEADER_SIZE = HEADER_STRUCT.size
MAX_DIM = 1 << 16


def version_hash_of(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=16).digest()


def version_hash_hex(data: bytes) -> str:
    return version_hash_of(data).hex()


def write_heightfield(path: str, heights: np.ndarray) -> str:
    """Write a validated float32 heightfield; returns the version hash hex."""
    arr = np.ascontiguousarray(heights, dtype="<f4")
    if arr.ndim != 2:
        raise HeightInvalid("heightfield must be a 2-D array")
    h, w = arr.shape
    if not (0 < w <= MAX_DIM and 0 < h <= MAX_DIM):
        raise HeightInvalid(f"bad dimensions {w}x{h}")
    if not np.all(np.isfinite(arr)) or float(arr.min()) < 0.0 or float(arr.max()) > 1.0:
        raise HeightInvalid("samples must be finite and within [0, 1]")
    data = arr.tobytes()
    header = HEADER_STRUCT.pack(MAGIC, FORMAT_VERSION, w, h, 0, version_hash_of(data))
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(data)
    return version_hash_hex(data)


def read_heightfield(path: str):
    """Read and fully validate a heightfield file.

    Returns (heights float32 ndarray (H, W), version_hash hex str).
    Raises HeaderMismatch / HeightInvalid on any inconsistency.
    """
    with open(path, "rb") as fh:
        blob = fh.read()
    if len(blob) < HEADER_SIZE:
        raise HeaderMismatch("file smaller than header", file=path)
    magic, fmt_ver, width, height, reserved, vhash = HEADER_STRUCT.unpack(
        blob[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise HeaderMismatch("bad magic", file=path)
    if fmt_ver != FORMAT_VERSION:
        raise HeaderMismatch(f"unsupported format version {fmt_ver}", file=path)
    if reserved != 0:
        raise HeaderMismatch("reserved header field must be zero", file=path)
    if not (0 < width <= MAX_DIM and 0 < height <= MAX_DIM):
        raise HeaderMismatch(f"bad dimensions {width}x{height}", file=path)
    expected = HEADER_SIZE + width * height * 4
    if len(blob) != expected:
        raise HeaderMismatch(
            f"data length mismatch: header says {expected} bytes, file has {len(blob)}",
            file=path,
        )
    data = blob[HEADER_SIZE:]
    if version_hash_of(data) != vhash:
        raise HeaderMismatch("version hash does not match payload", file=path)
    arr = np.frombuffer(data, dtype="<f4").reshape(height, width)
    if not np.all(np.isfinite(arr)):
        raise HeightInvalid("heightfield contains NaN or Inf samples", file=path)
    lo = float(arr.min())
    hi = float(arr.max())
    if lo < 0.0 or hi > 1.0:
        raise HeightInvalid(
            f"samples out of range [0, 1]: min={lo}, max={hi}", file=path
        )
    return arr, vhash.hex()


# ---------------------------------------------------------------------------
# Deterministic sample generator (no external assets, fixed algorithm).
# ---------------------------------------------------------------------------

def _value_noise(seed: int, freq: int, shape):
    """Deterministic value noise: hash-based lattice + smoothstep interp."""
    rng = np.random.default_rng(seed * 2654435761 % (2**32))
    lattice = rng.random((freq + 2, freq + 2), dtype=np.float64)
    h, w = shape
    ys = np.linspace(0, freq, h)
    xs = np.linspace(0, freq, w)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    sy = fy * fy * (3.0 - 2.0 * fy)
    sx = fx * fx * (3.0 - 2.0 * fx)
    v00 = lattice[y0[:, None], x0[None, :]]
    v10 = lattice[y0[:, None], x0[None, :] + 1]
    v01 = lattice[y0[:, None] + 1, x0[None, :]]
    v11 = lattice[y0[:, None] + 1, x0[None, :] + 1]
    top = v00 + (v10 - v00) * sx
    bot = v01 + (v11 - v01) * sx
    return top + (bot - top) * sy


def generate_sample(width: int, height: int, seed: int) -> np.ndarray:
    """Deterministic terrain in [0, 1]: layered value noise, fixed octaves."""
    shape = (height, width)
    acc = np.zeros(shape, dtype=np.float64)
    amp = 1.0
    total = 0.0
    for octave in range(4):
        freq = 4 << octave
        acc += amp * _value_noise(seed + octave * 101, freq, shape)
        total += amp
        amp *= 0.5
    acc /= total
    lo = float(acc.min())
    hi = float(acc.max())
    out = (acc - lo) / (hi - lo) if hi > lo else np.zeros_like(acc)
    return out.astype(np.float32)


def apply_edit(heights: np.ndarray, rect, delta: float) -> np.ndarray:
    """Deterministic local edit: raise/lower samples inside pixel rect.

    rect = (x, y, w, h) in pixel coordinates. Result is clamped to [0, 1].
    """
    x, y, w, h = (int(v) for v in rect)
    H, W = heights.shape
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    out = heights.astype(np.float32).copy()
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = np.clip(out[y0:y1, x0:x1] + np.float32(delta), 0.0, 1.0)
    return out
