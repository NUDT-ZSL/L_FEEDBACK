from __future__ import annotations

from dataclasses import dataclass
import json
import os
import tempfile

import numpy as np

from .errors import ValidationError


CACHE_VERSION = 1


@dataclass
class RenderCache:
    height: np.ndarray
    source_weights: np.ndarray
    color_linear: np.ndarray
    depth_linear: np.ndarray
    visibility: np.ndarray
    shadow_maps: np.ndarray
    structural_hash: str
    height_hash: str
    weights_hash: str
    structural_config: dict
    cascade_splits: list[float]
    warning_codes: list[str]

    def save(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, exist_ok=True)
        metadata = {
            "cache_version": CACHE_VERSION,
            "structural_hash": self.structural_hash,
            "height_hash": self.height_hash,
            "weights_hash": self.weights_hash,
            "structural_config": self.structural_config,
            "cascade_splits": self.cascade_splits,
            "warning_codes": self.warning_codes,
        }
        descriptor, temporary = tempfile.mkstemp(prefix=".terrain-cache-", suffix=".npz", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(
                    handle,
                    height=np.asarray(self.height, dtype=np.float64),
                    source_weights=np.asarray(self.source_weights, dtype=np.float64),
                    color_linear=np.asarray(self.color_linear, dtype=np.float32),
                    depth_linear=np.asarray(self.depth_linear, dtype=np.float32),
                    visibility=np.asarray(self.visibility, dtype=np.float32),
                    shadow_maps=np.asarray(self.shadow_maps, dtype=np.float32),
                    metadata=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(cls, path: str) -> "RenderCache":
        if not os.path.isfile(path):
            raise ValidationError(f"缓存文件不存在: {path}")
        try:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
                if int(metadata.get("cache_version", -1)) != CACHE_VERSION:
                    raise ValidationError("缓存版本不兼容，需要整体重建")
                return cls(
                    height=np.asarray(archive["height"], dtype=np.float64),
                    source_weights=np.asarray(archive["source_weights"], dtype=np.float64),
                    color_linear=np.asarray(archive["color_linear"], dtype=np.float32),
                    depth_linear=np.asarray(archive["depth_linear"], dtype=np.float32),
                    visibility=np.asarray(archive["visibility"], dtype=np.float32),
                    shadow_maps=np.asarray(archive["shadow_maps"], dtype=np.float32),
                    structural_hash=str(metadata["structural_hash"]),
                    height_hash=str(metadata["height_hash"]),
                    weights_hash=str(metadata["weights_hash"]),
                    structural_config=dict(metadata["structural_config"]),
                    cascade_splits=[float(v) for v in metadata["cascade_splits"]],
                    warning_codes=[str(v) for v in metadata.get("warning_codes", [])],
                )
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(f"无法读取渲染缓存: {path}: {exc}") from exc
