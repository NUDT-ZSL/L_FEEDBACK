"""Deterministic heightfield generation and self-describing terrain files."""

from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import TerrainConfig
from .errors import terrain_error


_FILE_MAGIC = b"TRNF"
_FILE_VERSION = 1
_HEADER = struct.Struct("<4sIIIQQ32s")
_U64 = np.uint64


@dataclass(frozen=True)
class TerrainData:
    heights: np.ndarray
    normals: np.ndarray
    sha256: str
    path: Path


def _mix64(value: np.ndarray) -> np.ndarray:
    value = value.copy()
    value ^= value >> _U64(30)
    value *= _U64(0xBF58476D1CE4E5B9)
    value ^= value >> _U64(27)
    value *= _U64(0x94D049BB133111EB)
    value ^= value >> _U64(31)
    return value


def _lattice_values(ix: np.ndarray, iy: np.ndarray, seed: int, octave: int) -> np.ndarray:
    x = ix.astype(np.uint64)
    y = iy.astype(np.uint64)
    mask = (1 << 64) - 1
    key_value = (
        int(seed) ^ ((octave + 1) * 0xD6E8FEB86659FD93)
    ) & mask
    key = _U64(key_value)
    mixed = _mix64(
        x * _U64(0x9E3779B185EBCA87)
        ^ y * _U64(0xC2B2AE3D27D4EB4F)
        ^ key
    )
    return ((mixed >> _U64(11)).astype(np.float64)) * (1.0 / 9007199254740992.0)


def _smoothstep01(value: np.ndarray) -> np.ndarray:
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def _value_noise(
    u: np.ndarray,
    v: np.ndarray,
    frequency: float,
    seed: int,
    octave: int,
) -> np.ndarray:
    x = u * frequency
    y = v * frequency
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    fx = _smoothstep01(x - ix)
    fy = _smoothstep01(y - iy)

    v00 = _lattice_values(ix, iy, seed, octave)
    v10 = _lattice_values(ix + 1, iy, seed, octave)
    v01 = _lattice_values(ix, iy + 1, seed, octave)
    v11 = _lattice_values(ix + 1, iy + 1, seed, octave)

    lower = v00 + (v10 - v00) * fx
    upper = v01 + (v11 - v01) * fx
    return lower + (upper - lower) * fy


def _smoothstep_edges(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def generate_heightfield(config: TerrainConfig) -> np.ndarray:
    """Generate a normalized [0, 1] heightfield with deterministic noise."""

    rows, cols = config.rows, config.cols
    u = np.linspace(0.0, 1.0, cols, dtype=np.float64)[None, :]
    v = np.linspace(0.0, 1.0, rows, dtype=np.float64)[:, None]
    u_grid = np.broadcast_to(u, (rows, cols))
    v_grid = np.broadcast_to(v, (rows, cols))

    total = np.zeros((rows, cols), dtype=np.float64)
    amplitude_sum = 0.0
    frequency = config.noise.frequency
    amplitude = 1.0
    for octave in range(config.noise.octaves):
        total += amplitude * _value_noise(
            u_grid, v_grid, frequency, config.seed, octave
        )
        amplitude_sum += amplitude
        frequency *= config.noise.lacunarity
        amplitude *= config.noise.persistence

    total /= max(amplitude_sum, 1e-15)
    minimum = float(np.min(total))
    maximum = float(np.max(total))
    if maximum - minimum <= 1e-15:
        raise terrain_error("generated heightfield is constant; adjust noise parameters")
    heights = (total - minimum) / (maximum - minimum)

    edge_u = _smoothstep_edges(0.0, 0.12, u_grid) * _smoothstep_edges(0.0, 0.12, 1.0 - u_grid)
    edge_v = _smoothstep_edges(0.0, 0.12, v_grid) * _smoothstep_edges(0.0, 0.12, 1.0 - v_grid)
    edge = edge_u * edge_v
    heights *= 0.55 + 0.45 * edge
    heights = (heights - float(np.min(heights))) / max(
        float(np.max(heights) - np.min(heights)), 1e-15
    )
    if config.height_patches:
        patch_scale = max(config.world_width, config.world_depth)
        world_x = u_grid * config.world_width
        world_z = v_grid * config.world_depth
        for patch in config.height_patches:
            center_x = patch.center[0] * config.world_width
            center_z = patch.center[1] * config.world_depth
            radius = patch.radius * patch_scale
            distance_squared = (world_x - center_x) ** 2 + (world_z - center_z) ** 2
            normalized = distance_squared / max(radius * radius, 1e-15)
            falloff = np.maximum(1.0 - normalized, 0.0) ** 2
            heights += patch.delta * falloff
        heights = np.clip(heights, 0.0, 1.0)
    return np.ascontiguousarray(heights.astype(np.float32))


def compute_normals(
    heights: np.ndarray,
    world_width: float,
    world_depth: float,
    height_scale: float,
) -> np.ndarray:
    """Compute per-vertex normals for y-up terrain in x/z world coordinates."""

    h = np.asarray(heights, dtype=np.float64) * height_scale
    rows, cols = h.shape
    dz = world_depth / max(rows - 1, 1)
    dx = world_width / max(cols - 1, 1)
    dh_dz, dh_dx = np.gradient(h, dz, dx, edge_order=1)
    normals = np.stack((-dh_dx, np.ones_like(h), -dh_dz), axis=-1)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    lengths = np.maximum(lengths, 1e-15)
    return np.ascontiguousarray((normals / lengths).astype(np.float32))


def terrain_payload(heights: np.ndarray) -> bytes:
    array = np.ascontiguousarray(heights, dtype="<f4")
    return array.tobytes(order="C")


def terrain_sha256(heights: np.ndarray) -> str:
    return hashlib.sha256(terrain_payload(heights)).hexdigest()


def write_terrain_binary(
    path: str | Path, heights: np.ndarray, seed: int
) -> str:
    destination = Path(path)
    rows, cols = heights.shape
    payload = terrain_payload(heights)
    digest = hashlib.sha256(payload).digest()
    header = _HEADER.pack(
        _FILE_MAGIC,
        _FILE_VERSION,
        rows,
        cols,
        len(payload),
        int(seed),
        digest,
    )
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as handle:
            handle.write(header)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise terrain_error(f"cannot write terrain file {destination}: {exc}") from None
    return digest.hex()


def read_terrain_binary(path: str | Path) -> tuple[np.ndarray, int]:
    source = Path(path)
    try:
        with source.open("rb") as handle:
            header = handle.read(_HEADER.size)
            if len(header) != _HEADER.size:
                raise terrain_error(f"terrain file is truncated: {source}")
            magic, version, rows, cols, payload_len, seed, digest = _HEADER.unpack(header)
            if magic != _FILE_MAGIC:
                raise terrain_error(f"terrain file has invalid magic: {source}")
            if version != _FILE_VERSION:
                raise terrain_error(f"unsupported terrain file version {version}: {source}")
            if rows < 2 or cols < 2:
                raise terrain_error(f"terrain file has invalid dimensions: {source}")
            if payload_len != rows * cols * 4:
                raise terrain_error(f"terrain file payload length is invalid: {source}")
            payload = handle.read(payload_len)
            trailing = handle.read(1)
            if len(payload) != payload_len or trailing:
                raise terrain_error(f"terrain file payload is truncated or oversized: {source}")
    except OSError as exc:
        raise terrain_error(f"cannot read terrain file {source}: {exc}") from None

    if hashlib.sha256(payload).digest() != digest:
        raise terrain_error(f"terrain file checksum mismatch: {source}")
    heights = np.frombuffer(payload, dtype="<f4").reshape(rows, cols).copy()
    return heights, seed


def build_terrain(config: TerrainConfig, path: str | Path) -> TerrainData:
    heights = generate_heightfield(config)
    digest = write_terrain_binary(path, heights, config.seed)
    normals = compute_normals(
        heights,
        config.world_width,
        config.world_depth,
        config.height_scale,
    )
    return TerrainData(
        heights=heights,
        normals=normals,
        sha256=digest,
        path=Path(path),
    )

