"""Shared fixtures/helpers for the test-suite."""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import numpy as np
import soundfile as sf

from audiopack.config import Config
from audiopack.samples import LANGS, generate_sample_assets, write_demo_manifest


def make_workspace(seed: int = 20261114) -> tuple[str, str, str, str, Config]:
    root = tempfile.mkdtemp(prefix="audiopack_test_")
    assets = os.path.join(root, "assets")
    manifest_path = os.path.join(root, "manifest.json")
    work = os.path.join(root, "work")
    out = os.path.join(root, "out")
    config = Config()
    generate_sample_assets(assets, seed=seed, config=config)
    write_demo_manifest(manifest_path)
    return root, assets, manifest_path, work, out, config


def cleanup(root: str) -> None:
    shutil.rmtree(root, ignore_errors=True)


def write_manifest(path: str, events: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"schema_version": 1, "events": events}, handle)


def write_wav(path: str, samples: np.ndarray, sample_rate: int = 48000) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pcm = np.round(np.clip(samples, -1, 1) * 32767).astype("<i2")
    sf.write(path, pcm, sample_rate, subtype="PCM_16", format="WAV")


def all_asset_pairs() -> list[tuple[str, str]]:
    from audiopack.samples import CLIP_DURATIONS_MS

    return [(key, lang) for key in CLIP_DURATIONS_MS for lang in LANGS]
