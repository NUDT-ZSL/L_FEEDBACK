import json
from pathlib import Path

import numpy as np
from PIL import Image

from terrain_gen.config import ErosionConfig, NoiseConfig, TerrainConfig
from terrain_gen.pipeline import run_pipeline


def make_config(prefix, seed=21, iterations=4):
    return TerrainConfig(
        width=256,
        height=256,
        iterations=iterations,
        output_prefix=str(prefix),
        noise=NoiseConfig(seed=seed, octaves=4),
        erosion=ErosionConfig(),
    )


def test_outputs_created_with_stats_and_metadata(tmp_path):
    prefix = tmp_path / "nested" / "deep" / "terrain"
    result = run_pipeline(make_config(prefix))
    for path in result["outputs"].values():
        assert Path(path).exists()

    metadata = json.loads(
        (tmp_path / "nested" / "deep" / "terrain_metadata.json").read_text()
    )
    assert metadata["iterations_executed"] == 4
    assert len(metadata["convergence"]) == 4
    assert "height_stats_final" in metadata
    assert "config" in metadata
    stats = metadata["material_stats"]
    assert abs(sum(stats["fractions"].values()) - 1.0) <= 0.0001


def test_png_modes_and_dimensions(tmp_path):
    run_pipeline(make_config(tmp_path / "terrain"))
    height = Image.open(tmp_path / "terrain_height.png")
    normal = Image.open(tmp_path / "terrain_normal.png")
    material = Image.open(tmp_path / "terrain_material.png")
    assert height.mode == "I;16" and height.size == (256, 256)
    assert normal.mode == "RGB" and normal.size == (256, 256)
    assert material.mode == "I;16" and material.size == (256, 256)


def test_height_png_byte_identical_across_runs(tmp_path):
    prefix_a = tmp_path / "a" / "terrain"
    prefix_b = tmp_path / "b" / "terrain"
    run_pipeline(make_config(prefix_a))
    run_pipeline(make_config(prefix_b))
    bytes_a = (tmp_path / "a" / "terrain_height.png").read_bytes()
    bytes_b = (tmp_path / "b" / "terrain_height.png").read_bytes()
    assert bytes_a == bytes_b

    normal_a = (tmp_path / "a" / "terrain_normal.png").read_bytes()
    normal_b = (tmp_path / "b" / "terrain_normal.png").read_bytes()
    assert normal_a == normal_b


def test_different_seed_changes_height_png(tmp_path):
    run_pipeline(make_config(tmp_path / "a" / "terrain", seed=1))
    run_pipeline(make_config(tmp_path / "b" / "terrain", seed=2))
    bytes_a = (tmp_path / "a" / "terrain_height.png").read_bytes()
    bytes_b = (tmp_path / "b" / "terrain_height.png").read_bytes()
    assert bytes_a != bytes_b


def test_height_png_round_trips_to_unit_range(tmp_path):
    run_pipeline(make_config(tmp_path / "terrain"))
    raw = np.array(Image.open(tmp_path / "terrain_height.png"), dtype=np.float64) / 65535.0
    assert raw.min() >= 0.0
    assert raw.max() <= 1.0
    assert raw.max() > 0.5
