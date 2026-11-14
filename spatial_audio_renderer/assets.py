"""Deterministic in-memory audio asset generation."""

from __future__ import annotations

import hashlib

import numpy as np

from .scene import AssetSpec, Scene


def _source_sample_count(asset: AssetSpec, scene: Scene) -> int:
    if asset.duration is None:
        return scene.total_samples
    return max(1, int(round(asset.duration * scene.sample_rate)))


def generate_asset(asset: AssetSpec, scene: Scene) -> np.ndarray:
    """Generate an asset using only deterministic local math.

    The returned signal always has exactly the scene's sample count. Short
    assets are zero-padded unless loop=true.
    """

    source_count = _source_sample_count(asset, scene)
    sample_rate = scene.sample_rate
    if asset.kind == "sine":
        time = np.arange(source_count, dtype=np.float64) / float(sample_rate)
        signal = asset.amplitude * np.sin(
            2.0 * np.pi * asset.frequency_hz * time + asset.phase_radians
        )
    elif asset.kind == "noise":
        rng = np.random.Generator(np.random.PCG64(int(asset.seed)))
        signal = asset.amplitude * rng.standard_normal(source_count)
        # Keep generated peaks in a comparable range without linking assets.
        peak = float(np.max(np.abs(signal))) if signal.size else 0.0
        if peak > 1.0:
            signal = signal / peak
    else:
        raise AssertionError("unsupported validated asset kind: %s" % asset.kind)

    if source_count == scene.total_samples:
        return np.asarray(signal, dtype=np.float64)

    if asset.loop:
        repeats = int(np.ceil(scene.total_samples / float(source_count)))
        return np.tile(signal, repeats)[:scene.total_samples].astype(np.float64, copy=False)

    output = np.zeros(scene.total_samples, dtype=np.float64)
    usable = min(source_count, scene.total_samples)
    output[:usable] = signal[:usable]
    return output


def asset_sha256(signal: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(signal, dtype=np.float64).tobytes()).hexdigest()

