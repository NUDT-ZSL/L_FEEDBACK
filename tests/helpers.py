from __future__ import annotations

import json
from pathlib import Path

from deferred_validation.scene import load_scene


def write_test_scene(
    path: Path,
    *,
    width: int = 128,
    height: int = 96,
    ssaa: int = 1,
    tone_mapper: str = "aces",
    light_scale: float = 1.0,
) -> Path:
    from deferred_validation.default_scene import build_scene

    data = build_scene()
    data["rendering"]["width"] = width
    data["rendering"]["height"] = height
    data["rendering"]["ssaa"] = ssaa
    data["rendering"]["tone_mapper"] = tone_mapper
    data["rendering"]["shadows"]["directional_resolution"] = 128
    data["rendering"]["shadows"]["spot_resolution"] = 128
    data["rendering"]["shadows"]["point_resolution"] = 64
    for light in data["lights"]:
        light["intensity"] *= light_scale
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_test_scene(path: Path, **overrides):
    scene = load_scene(path)
    for key, value in overrides.items():
        setattr(scene.rendering, key, value)
    return scene
