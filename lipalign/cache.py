"""Incremental stretched-fragment cache."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .jsonio import atomic_write_json
from .stretch import StretchSettings
from .wavio import read_wav, sha256_int16, write_pcm16_wav


PathLike = Union[str, os.PathLike]
CACHE_SCHEMA_VERSION = 1


class FragmentCache:
    def __init__(
        self,
        cache_dir: PathLike,
        sample_rate: int,
        stretch_settings: StretchSettings,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.sample_rate = int(sample_rate)
        self.stretch_settings = stretch_settings
        self.entries_dir = self.cache_dir / "stretched"
        self.entries_dir.mkdir(parents=True, exist_ok=True)

    def key_for(
        self,
        source_sha256: str,
        source_samples: int,
        target_samples: int,
    ) -> str:
        """Build a key that ignores sentence position but tracks stretch inputs."""

        identity = {
            "schema": CACHE_SCHEMA_VERSION,
            "source_sha256": source_sha256,
            "source_samples": int(source_samples),
            "target_samples": int(target_samples),
            "sample_rate": self.sample_rate,
            "stretch_settings": self.stretch_settings.cache_identity(),
        }
        from hashlib import sha256

        encoded = json.dumps(
            identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def _paths(self, key: str) -> Dict[str, Path]:
        return {
            "wav": self.entries_dir / "{}.wav".format(key),
            "meta": self.entries_dir / "{}.json".format(key),
        }

    def get(
        self,
        key: str,
        source_sha256: str,
        source_samples: int,
        target_samples: int,
    ) -> Optional[np.ndarray]:
        paths = self._paths(key)
        if not paths["wav"].is_file() or not paths["meta"].is_file():
            return None
        try:
            meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
            if meta.get("key") != key:
                return None
            if meta.get("source_sha256") != source_sha256:
                return None
            if int(meta.get("source_samples", -1)) != int(source_samples):
                return None
            if int(meta.get("target_samples", -1)) != int(target_samples):
                return None
            if int(meta.get("sample_rate", -1)) != self.sample_rate:
                return None
            if meta.get("stretch_settings") != self.stretch_settings.cache_identity():
                return None
            audio = read_wav(paths["wav"])
            if audio.channels != 1 or audio.sample_rate != self.sample_rate:
                return None
            if audio.sample_count != target_samples:
                return None
            pcm = audio.samples[:, 0].copy()
            if meta.get("output_sha256") and meta["output_sha256"] != sha256_int16(pcm):
                return None
            return pcm
        except Exception:
            return None

    def put(
        self,
        key: str,
        source_sha256: str,
        source_samples: int,
        target_samples: int,
        pcm: np.ndarray,
    ) -> str:
        paths = self._paths(key)
        output = np.asarray(pcm, dtype=np.int16).reshape(-1)
        if len(output) != target_samples:
            raise ValueError("cached fragment length does not match target")
        output_hash = sha256_int16(output)
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "key": key,
            "source_sha256": source_sha256,
            "source_samples": int(source_samples),
            "target_samples": int(target_samples),
            "sample_rate": self.sample_rate,
            "stretch_settings": self.stretch_settings.cache_identity(),
            "output_sha256": output_hash,
            "output_samples": int(len(output)),
            "pcm_encoding": "16-bit signed little-endian mono",
        }
        write_pcm16_wav(paths["wav"], output, self.sample_rate, channels=1)
        atomic_write_json(paths["meta"], metadata)
        return output_hash


def cache_summary(cache_dir: PathLike) -> Dict[str, Any]:
    root = Path(cache_dir)
    entries_dir = root / "stretched"
    entries: List[Dict[str, Any]] = []
    invalid_sidecars = 0
    total_bytes = 0
    if entries_dir.is_dir():
        for sidecar in sorted(entries_dir.glob("*.json")):
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("sidecar is not an object")
                entries.append(payload)
                wav_path = sidecar.with_suffix(".wav")
                if wav_path.is_file():
                    total_bytes += wav_path.stat().st_size + sidecar.stat().st_size
            except Exception:
                invalid_sidecars += 1
    algorithms: Dict[str, int] = {}
    sample_rates: Dict[str, int] = {}
    for entry in entries:
        settings = entry.get("stretch_settings") or {}
        algorithm = str(settings.get("algorithm", "unknown"))
        algorithms[algorithm] = algorithms.get(algorithm, 0) + 1
        rate = str(entry.get("sample_rate", "unknown"))
        sample_rates[rate] = sample_rates.get(rate, 0) + 1
    return {
        "cache_dir": str(root),
        "entries_dir": str(entries_dir),
        "entry_count": len(entries),
        "invalid_sidecar_count": invalid_sidecars,
        "total_bytes": total_bytes,
        "algorithms": algorithms,
        "sample_rates": sample_rates,
    }

