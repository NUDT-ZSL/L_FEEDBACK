import numpy as np
import pytest

from terrain_gen.config import ErosionConfig
from terrain_gen.erosion import ErosionState, hydraulic_step, simulate, thermal_step


def make_height(seed=0):
    rng = np.random.default_rng(seed)
    field = rng.random((64, 64))
    return field


def make_state(seed=0):
    height = make_height(seed)
    return ErosionState(height, np.zeros_like(height), np.zeros_like(height))


def test_hydraulic_keeps_non_negative_and_bounded():
    cfg = ErosionConfig()
    state = make_state()
    for _ in range(20):
        hydraulic_step(state, cfg, cfg.hydraulic_weight)
        assert np.isfinite(state.height).all()
        assert state.height.min() >= 0.0
        assert state.water.min() >= 0.0
        assert state.sediment.min() >= 0.0


def test_hydraulic_water_non_negative_under_heavy_rain():
    cfg = ErosionConfig(rain=1.0, pipe_factor=0.9, erodibility=0.8)
    state = make_state()
    for _ in range(30):
        hydraulic_step(state, cfg, 1.0)
        assert (state.water >= 0.0).all()
        assert (state.height >= 0.0).all()


def test_hydraulic_records_max_erosion_per_iteration():
    cfg = ErosionConfig()
    height = make_height()
    _, diagnostics = simulate(height, cfg, iterations=6)
    assert len(diagnostics) == 6
    assert [row["iteration"] for row in diagnostics] == [1, 2, 3, 4, 5, 6]
    assert all(row["max_erosion"] >= 0.0 for row in diagnostics)
    assert all(row["min_height"] >= 0.0 for row in diagnostics)


def test_thermal_transfer_conserves_total_height():
    cfg = ErosionConfig(talus=0.02, thermal_carry=0.5)
    height = make_height()
    initial_total = float(height.sum())
    max_drift = 0.0
    for _ in range(10):
        _, drift = thermal_step(height, cfg, cfg.thermal_weight)
        max_drift = max(max_drift, abs(drift))
    assert max_drift <= initial_total * 1e-6
    assert height.min() >= 0.0


def test_thermal_relaxes_steep_slopes_only():
    cfg = ErosionConfig(talus=0.05)
    height = np.full((32, 32), 0.5)
    height[:16, :] = 0.9
    initial = height.copy()
    thermal_step(height, cfg, 1.0)
    # Material must move away from the high band towards the low band.
    assert height[:16, :].mean() < initial[:16, :].mean()
    assert height[16:, :].mean() > initial[16:, :].mean()


def test_simulate_deterministic():
    cfg = ErosionConfig()
    source = make_height(5)
    first, diag_first = simulate(source, cfg, 5)
    second, diag_second = simulate(source, cfg, 5)
    np.testing.assert_array_equal(first, second)
    assert diag_first == diag_second


def test_simulate_stays_bounded_over_max_iterations():
    cfg = ErosionConfig(rain=0.5, pipe_factor=0.9, erodibility=0.9, thermal_weight=1.0)
    final, diagnostics = simulate(make_height(2), cfg, iterations=50)
    assert np.isfinite(final).all()
    assert final.min() >= 0.0
    assert final.max() <= 1.0
    assert all(row["min_height"] >= 0.0 for row in diagnostics)
    assert all(np.isfinite(row["max_erosion"]) for row in diagnostics)


def test_thermal_machine_precision_conservation_256():
    cfg = ErosionConfig(talus=0.02, thermal_carry=1.0, thermal_weight=1.0)
    height = np.random.default_rng(4).random((256, 256))
    initial_total = height.sum()
    worst = 0.0
    for _ in range(50):
        _, drift = thermal_step(height, cfg, 1.0)
        worst = max(worst, abs(drift))
    assert worst <= initial_total * 1e-6


def test_fault_injection_raises():
    with pytest.raises(RuntimeError, match="injected failure"):
        simulate(make_height(), ErosionConfig(), 10, fault_at=4)
