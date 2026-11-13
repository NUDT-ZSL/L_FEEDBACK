"""JSON run configuration loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bake import BakeParams
from .layout import LayoutParams
from .scene import SceneGeneratorConfig


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RunConfig:
    generator: SceneGeneratorConfig
    layout: LayoutParams
    bake: BakeParams
    patch: dict[str, Any]

    @classmethod
    def defaults(cls) -> "RunConfig":
        return cls(
            generator=SceneGeneratorConfig(),
            layout=LayoutParams(),
            bake=BakeParams(),
            patch={},
        )


def read_json_object(path: str | Path, label: str) -> dict[str, Any]:
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {label} {config_path}: {exc}") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{label} {config_path} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{label} {config_path} must contain a JSON object")
    return value


def parse_bake_params(value: dict[str, Any] | None) -> BakeParams:
    if value is None:
        return BakeParams()
    if not isinstance(value, dict):
        raise ConfigError("bake config must be an object")
    allowed = {"samples", "ray_epsilon", "workers", "chunk_size"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"unknown bake config keys: {unknown}")
    defaults = BakeParams()
    return BakeParams(
        samples=value.get("samples", defaults.samples),
        ray_epsilon=value.get("ray_epsilon", defaults.ray_epsilon),
        workers=value.get("workers", defaults.workers),
        chunk_size=value.get("chunk_size", defaults.chunk_size),
    )


def load_run_config(path: str | Path | None) -> RunConfig:
    if path is None:
        return RunConfig.defaults()
    value = read_json_object(path, "run config")
    allowed = {"config_version", "scene", "layout", "bake", "patch"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"unknown top-level run config keys: {unknown}")
    if "config_version" in value and value["config_version"] != 1:
        raise ConfigError("unsupported run config version")

    scene_value = value.get("scene")
    if scene_value is not None and not isinstance(scene_value, dict):
        raise ConfigError("scene config must be an object")
    patch_value = value.get("patch", {})
    if patch_value is None:
        patch_value = {}
    if not isinstance(patch_value, dict):
        raise ConfigError("patch config must be an object")
    try:
        generator = SceneGeneratorConfig.from_dict(scene_value)
        layout = LayoutParams.from_dict(value.get("layout"))
        bake = parse_bake_params(value.get("bake"))
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    return RunConfig(generator=generator, layout=layout, bake=bake, patch=dict(patch_value))

