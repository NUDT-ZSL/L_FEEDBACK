"""Configuration loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import BakeError, ErrorCode


@dataclass
class NoiseConfig:
    octaves: int = 5
    base_frequency: float = 3.0   # lattice cells across the whole heightfield
    gain: float = 0.5
    lacunarity: float = 2.0


@dataclass
class HeightfieldConfig:
    resolution: int = 257         # samples per side (vertices)
    chunk_size: int = 65          # samples per chunk side, including shared border
    world_size: float = 1000.0    # meters per side
    min_height: float = 0.0
    max_height: float = 120.0
    noise: NoiseConfig = field(default_factory=NoiseConfig)


@dataclass
class ViewConfig:
    position: tuple = (500.0, 500.0, 300.0)
    fov_y_deg: float = 60.0
    viewport_height_px: int = 1080
    screen_error_threshold_px: float = 2.0


@dataclass
class Config:
    seed: int
    heightfield: HeightfieldConfig
    lod_levels: int
    normal_threshold_deg: float
    view: ViewConfig

    @property
    def chunks_per_side(self) -> int:
        hf = self.heightfield
        return (hf.resolution - 1) // (hf.chunk_size - 1)

    @property
    def spacing(self) -> float:
        hf = self.heightfield
        return hf.world_size / (hf.resolution - 1)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise BakeError(ErrorCode.E_CONFIG, msg)


def load_config(path: str | Path) -> Config:
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BakeError(ErrorCode.E_CONFIG, f"cannot read config {p}: {exc}")
    except json.JSONDecodeError as exc:
        raise BakeError(ErrorCode.E_CONFIG, f"invalid JSON in {p}: {exc}")

    try:
        seed = int(raw["seed"])
        hf_raw = raw.get("heightfield", {})
        noise_raw = hf_raw.get("noise", {})
        noise = NoiseConfig(
            octaves=int(noise_raw.get("octaves", 5)),
            base_frequency=float(noise_raw.get("base_frequency", 3.0)),
            gain=float(noise_raw.get("gain", 0.5)),
            lacunarity=float(noise_raw.get("lacunarity", 2.0)),
        )
        hf = HeightfieldConfig(
            resolution=int(hf_raw.get("resolution", 257)),
            chunk_size=int(hf_raw.get("chunk_size", 65)),
            world_size=float(hf_raw.get("world_size", 1000.0)),
            min_height=float(hf_raw.get("min_height", 0.0)),
            max_height=float(hf_raw.get("max_height", 120.0)),
            noise=noise,
        )
        lod_raw = raw.get("lods", {})
        lod_levels = int(lod_raw.get("levels", 3))
        threshold = float(lod_raw.get("normal_threshold_deg", 2.0))
        view_raw = raw.get("view", {})
        view = ViewConfig(
            position=tuple(view_raw.get("position", (500.0, 500.0, 300.0))),
            fov_y_deg=float(view_raw.get("fov_y_deg", 60.0)),
            viewport_height_px=int(view_raw.get("viewport_height_px", 1080)),
            screen_error_threshold_px=float(view_raw.get("screen_error_threshold_px", 2.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BakeError(ErrorCode.E_CONFIG, f"bad config value: {exc}")

    _require(hf.resolution >= 3, "heightfield.resolution must be >= 3")
    _require(hf.chunk_size >= 3, "heightfield.chunk_size must be >= 3")
    _require((hf.resolution - 1) % (hf.chunk_size - 1) == 0,
             "(resolution - 1) must be divisible by (chunk_size - 1)")
    _require(lod_levels >= 1, "lods.levels must be >= 1")
    max_step = 1 << (lod_levels - 1)
    _require((hf.chunk_size - 1) % max_step == 0,
             "(chunk_size - 1) must be divisible by 2^(levels - 1)")
    _require(hf.max_height > hf.min_height, "max_height must exceed min_height")
    _require(noise.octaves >= 1, "noise.octaves must be >= 1")
    _require(threshold > 0.0, "normal_threshold_deg must be positive")
    return Config(seed=seed, heightfield=hf, lod_levels=lod_levels,
                  normal_threshold_deg=threshold, view=view)
