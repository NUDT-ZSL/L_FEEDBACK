from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .errors import RendererError


GBUFFER_LAYOUT: Dict[str, Dict[str, Any]] = {
    "position": {
        "file": "position.npy",
        "dtype": "float32",
        "components": ["x", "y", "z"],
        "semantic": "world-space surface position",
    },
    "normal": {
        "file": "normal.npy",
        "dtype": "float32",
        "components": ["x", "y", "z"],
        "semantic": "normalized world-space geometric normal",
    },
    "tangent": {
        "file": "tangent.npy",
        "dtype": "float32",
        "components": ["x", "y", "z"],
        "semantic": "normalized world-space tangent",
    },
    "bitangent": {
        "file": "bitangent.npy",
        "dtype": "float32",
        "components": ["x", "y", "z"],
        "semantic": "normalized world-space bitangent",
    },
    "uv": {
        "file": "uv.npy",
        "dtype": "float32",
        "components": ["u", "v"],
        "semantic": "interpolated texture coordinate",
    },
    "material_index": {
        "file": "material_index.npy",
        "dtype": "int32",
        "components": ["material_index"],
        "semantic": "index into the scene material table",
    },
    "depth": {
        "file": "depth.npy",
        "dtype": "float32",
        "components": ["window_depth"],
        "semantic": "perspective NDC depth mapped to [0, 1]",
    },
    "coverage": {
        "file": "coverage.npy",
        "dtype": "uint8",
        "components": ["covered"],
        "semantic": "1 for covered pixels, 0 for background",
    },
}


@dataclass
class GBuffer:
    position: np.ndarray
    normal: np.ndarray
    tangent: np.ndarray
    bitangent: np.ndarray
    uv: np.ndarray
    material_index: np.ndarray
    depth: np.ndarray
    coverage: np.ndarray

    @property
    def width(self) -> int:
        return int(self.position.shape[1])

    @property
    def height(self) -> int:
        return int(self.position.shape[0])

    @property
    def covered_pixels(self) -> int:
        return int(np.count_nonzero(self.coverage))

    def arrays(self) -> Dict[str, np.ndarray]:
        return {
            "position": self.position,
            "normal": self.normal,
            "tangent": self.tangent,
            "bitangent": self.bitangent,
            "uv": self.uv,
            "material_index": self.material_index,
            "depth": self.depth,
            "coverage": self.coverage,
        }

    def validate(self) -> None:
        height, width = self.depth.shape
        expected = {
            "position": (height, width, 3),
            "normal": (height, width, 3),
            "tangent": (height, width, 3),
            "bitangent": (height, width, 3),
            "uv": (height, width, 2),
            "material_index": (height, width),
            "coverage": (height, width),
        }
        for name, shape in expected.items():
            array = getattr(self, name)
            if array.shape != shape:
                raise RendererError(
                    "GBuffer channel %s has shape %s, expected %s"
                    % (name, array.shape, shape)
                )
        for name in ("position", "normal", "tangent", "bitangent"):
            if not np.all(np.isfinite(getattr(self, name)[self.coverage.astype(bool)])):
                raise RendererError("GBuffer channel %s contains non-finite values" % name)

    def save(self, directory: Path) -> Dict[str, str]:
        directory.mkdir(parents=True, exist_ok=True)
        files: Dict[str, str] = {}
        for name, array in self.arrays().items():
            filename = GBUFFER_LAYOUT[name]["file"]
            np.save(directory / filename, array, allow_pickle=False)
            files[name] = filename
        metadata = {
            "schema_version": 1,
            "width": self.width,
            "height": self.height,
            "channels": GBUFFER_LAYOUT,
            "files": files,
            "background": {
                "depth": 1.0,
                "coverage": 0,
                "material_index": -1,
            },
        }
        with (directory / "layout.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return files

    @classmethod
    def load(cls, directory: Path) -> "GBuffer":
        layout_path = directory / "layout.json"
        try:
            with layout_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RendererError("cannot read GBuffer layout %s: %s" % (layout_path, exc))
        files = metadata.get("files", {})
        arrays = {}
        for name in GBUFFER_LAYOUT:
            filename = files.get(name, GBUFFER_LAYOUT[name]["file"])
            try:
                arrays[name] = np.load(directory / filename, allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise RendererError(
                    "cannot read GBuffer channel %s from %s: %s"
                    % (name, directory / filename, exc)
                )
        gbuffer = cls(**arrays)
        gbuffer.validate()
        return gbuffer
