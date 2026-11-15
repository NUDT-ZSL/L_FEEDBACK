import numpy as np

from terrain_gen.config import NoiseConfig
from terrain_gen.materials import (
    MATERIAL_NAMES,
    ROCK,
    SAND,
    SNOW,
    classify_materials,
    material_statistics,
)
from terrain_gen.noise import generate_heightfield


def test_four_material_ids_present_on_terrain():
    field = generate_heightfield(256, 256, NoiseConfig(seed=11, octaves=5))
    material = classify_materials(field)
    assert material.dtype == np.uint16
    unique = set(np.unique(material).tolist())
    assert unique.issubset({0, 1, 2, 3})
    assert len(unique) >= 4
    assert len(MATERIAL_NAMES) == 4


def test_low_terrain_is_sand_and_high_ground_is_snow_or_rock():
    field = generate_heightfield(256, 256, NoiseConfig(seed=12))
    material = classify_materials(field)
    assert (material[field < 0.25] == SAND).all()
    high_gentle = (field >= 0.72)
    assert np.isin(material[high_gentle], [ROCK, SNOW]).all()


def test_statistics_fractions_sum_to_one():
    field = generate_heightfield(256, 256, NoiseConfig(seed=13))
    material = classify_materials(field)
    stats = material_statistics(material)
    fractions = stats["fractions"]
    assert set(fractions) == set(MATERIAL_NAMES)
    assert abs(sum(fractions.values()) - 1.0) <= 0.0001
    assert sum(stats["pixel_counts"].values()) == stats["total_pixels"]
    assert all(0.0 <= value <= 1.0 for value in fractions.values())
