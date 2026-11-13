"""Generate deterministic sample camera/world configurations as JSON.

No external data is involved: every number below is fixed in source so that
regeneration yields byte-identical files.
"""

from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig, write_json


def default_config() -> PipelineConfig:
    return PipelineConfig(
        seed=20261113,
        root_min=(-16.0, -16.0),
        root_size=32.0,
        max_depth=4,
        tile_resolution=9,
        height_scale=6.0,
        noise_octaves=4,
        camera=dict(
            eye=(24.0, 22.0, 28.0),
            look_at=(0.0, 1.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fov_y_deg=55.0,
            near=0.25,
            far=200.0,
        ),
        image_width=256,
        image_height=192,
        sse_threshold=40.0,
        sse_relative_tolerance=1e-6,
        enable_stitching=True,
        light=dict(
            direction=(-0.45, -0.8, -0.4),
            ambient=0.18,
            diffuse_albedo=0.82,
            gamma=2.2,
        ),
        backend="software",
        output_png="artifacts/terrain.png",
        output_contract="artifacts/contract.json",
    )


def generate(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    base = default_config()
    default_path = directory / "default.json"
    write_json(default_path, base.to_dict())
    paths.append(default_path)

    no_stitch = base.with_overrides(
        enable_stitching=False,
        output_png="artifacts/terrain_no_stitch.png",
        output_contract="artifacts/contract_no_stitch.json",
    )
    no_stitch_path = directory / "no_stitch.json"
    write_json(no_stitch_path, no_stitch.to_dict())
    paths.append(no_stitch_path)
    return paths


if __name__ == "__main__":
    created = generate("configs")
    for path in created:
        print(path)
