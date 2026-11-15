"""Hydraulic and thermal erosion simulation on grid heightfields.

Both passes operate on periodic (wrapping) grids so tiles stay seamless.
All arithmetic is float64. Heights are clipped to be non-negative after every
terrain update, which together with capped erosion prevents the numerical
oscillation and negative-height blow-up common in explicit erosion schemes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ErosionConfig

# Cardinal directions as ``(row shift, col shift)`` consumed by ``np.roll``.
CARDINAL_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))
# Eight-neighbour stencil for thermal talus relaxation.
THERMAL_OFFSETS = (
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
)

EPS = 1.0e-12


@dataclass
class ErosionState:
    height: np.ndarray
    water: np.ndarray
    sediment: np.ndarray


def hydraulic_step(state: ErosionState, cfg: ErosionConfig, weight: float) -> float:
    """Run one hydraulic pass; return the maximum terrain erosion amount."""
    h = state.height
    water = state.water
    sediment = state.sediment

    # 1. Rainfall adds a uniform layer of running water.
    water += cfg.rain

    # 2. Flow accumulation: pipe-model outflow towards the four cardinal cells.
    outflows = []
    total_outflow = np.zeros_like(h)
    surface = h + water
    for shift in CARDINAL_OFFSETS:
        neighbor_surface = np.roll(surface, shift=shift, axis=(0, 1))
        head_delta = surface - neighbor_surface
        outflow = np.maximum(0.0, cfg.pipe_factor * weight * head_delta)
        outflows.append(outflow)
        total_outflow += outflow

    # Limit outflow to available water to keep the scheme stable.
    scale = np.where(total_outflow > water + EPS, water / (total_outflow + EPS), 1.0)
    outflows = [outflow * scale for outflow in outflows]
    total_outflow = np.zeros_like(h)
    total_inflow = np.zeros_like(h)
    for (shift, outflow) in zip(CARDINAL_OFFSETS, outflows):
        total_outflow += outflow
        total_inflow += np.roll(outflow, shift=tuple(-axis for axis in shift), axis=(0, 1))

    # 3. Water mass balance.
    water += total_inflow - total_outflow
    np.maximum(water, 0.0, out=water)

    # 4. Approximate flow speed and local downhill slope.
    flow_volume = np.abs(total_inflow - total_outflow)
    velocity = flow_volume / (water + EPS)
    slope = np.zeros_like(h)
    for shift in CARDINAL_OFFSETS:
        neighbor_h = np.roll(h, shift=shift, axis=(0, 1))
        slope = np.maximum(slope, h - neighbor_h)
    slope = np.maximum(0.0, slope)

    # 5. Sediment transport capacity and erosion / deposition sub-steps.
    capacity = cfg.erodibility * water * velocity * slope
    over_capacity = sediment > capacity

    deposited = np.where(
        over_capacity,
        cfg.deposition * weight * (sediment - capacity),
        0.0,
    )
    sediment -= deposited
    h += deposited

    wanted_erosion = np.where(
        over_capacity,
        0.0,
        cfg.erodibility * weight * (capacity - sediment),
    )
    # Never erode more material than the cell holds: guarantees h >= 0.
    eroded = np.minimum(wanted_erosion, h)
    h -= eroded
    sediment += eroded
    np.maximum(h, 0.0, out=h)

    # 6. Evaporation removes a fraction of the remaining water.
    water *= 1.0 - cfg.evaporation

    return float(eroded.max())


def thermal_step(
    height: np.ndarray, cfg: ErosionConfig, weight: float
) -> tuple[float, float]:
    """Relax material towards neighbours steeper than the talus angle.

    The total downhill excess of each cell is split among its eligible
    neighbours in proportion to their individual excess (the classic
    Musgrave/Olsen thermal-erosion formulation). A single shared delta field
    accumulates every removal and matching deposit, so total height is
    conserved up to floating-point round-off and the scheme cannot blow up.
    """
    delta = np.zeros_like(height)
    before_total = float(height.sum())
    total_eligible = np.zeros_like(height)
    max_eligible = np.zeros_like(height)
    eligible: list[tuple[tuple[int, int], np.ndarray]] = []

    for shift in THERMAL_OFFSETS:
        neighbor = np.roll(height, shift=shift, axis=(0, 1))
        excess = np.maximum(0.0, height - neighbor - cfg.talus)
        total_eligible += excess
        max_eligible = np.maximum(max_eligible, excess)
        eligible.append((shift, excess))

    # A cell only moves material when at least one neighbour is below the talus.
    moveable = np.where(total_eligible > EPS, max_eligible, 0.0)
    transfer = 0.5 * cfg.thermal_carry * weight * moveable

    for shift, excess in eligible:
        share = np.where(total_eligible > EPS, excess / (total_eligible + EPS), 0.0)
        moved = transfer * share
        delta -= moved
        delta += np.roll(moved, shift=tuple(-axis for axis in shift), axis=(0, 1))

    height += delta
    # Round-off can only produce microscopic negatives; clamp them explicitly.
    np.maximum(height, 0.0, out=height)
    max_transfer = float(np.abs(delta).max())
    total_change = float(height.sum()) - before_total
    return max_transfer, total_change


def simulate(
    heightfield: np.ndarray,
    cfg: ErosionConfig,
    iterations: int,
    fault_at: int | None = None,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Run the multi-pass erosion loop and return the field plus diagnostics."""
    state = ErosionState(
        height=heightfield.astype(np.float64, copy=True),
        water=np.zeros_like(heightfield, dtype=np.float64),
        sediment=np.zeros_like(heightfield, dtype=np.float64),
    )
    diagnostics: list[dict[str, float]] = []

    for index in range(1, iterations + 1):
        if fault_at is not None and index == fault_at:
            raise RuntimeError(f"injected failure at erosion iteration {index}")

        if cfg.hydraulic_weight > 0.0:
            max_erosion = hydraulic_step(state, cfg, cfg.hydraulic_weight)
        else:
            max_erosion = 0.0

        if cfg.thermal_weight > 0.0:
            max_thermal, thermal_drift = thermal_step(
                state.height, cfg, cfg.thermal_weight
            )
        else:
            max_thermal, thermal_drift = 0.0, 0.0

        np.maximum(state.height, 0.0, out=state.height)
        diagnostics.append(
            {
                "iteration": index,
                "max_erosion": max_erosion,
                "max_thermal_transfer": max_thermal,
                "thermal_total_drift": thermal_drift,
                "min_height": float(state.height.min()),
                "max_height": float(state.height.max()),
            }
        )

    np.clip(state.height, 0.0, 1.0, out=state.height)
    return state.height, diagnostics
