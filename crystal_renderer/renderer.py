"""Full and incremental render pipelines with dirty-rect tracking."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SceneConfig
from .raster import render_region, to_framebuffer
from .scene import (
    CrystalState,
    dilated_aabb,
    env_background,
    rect_union,
    rects_intersect,
    scene_states,
)


@dataclass
class FrameRecord:
    frame: int
    shaded: int
    dirty_rect: tuple[int, int, int, int] | None
    changed_crystals: list[str] = field(default_factory=list)
    draw_sequence: list[str] = field(default_factory=list)


def draw_sequence(states: list[CrystalState]) -> list[str]:
    ordered = sorted(
        (s for s in states if s.visible),
        key=lambda s: (s.layer, s.crystal_id),
    )
    return [s.crystal_id for s in ordered]


def compute_dirty_rect(
    prev_states: list[CrystalState],
    next_states: list[CrystalState],
    width: int,
    height: int,
):
    prev_by_id = {s.crystal_id: s for s in prev_states}
    changed = []
    dirty = None
    for state in next_states:
        prev = prev_by_id.get(state.crystal_id)
        if prev is None:
            continue
        if prev.state_signature() == state.state_signature():
            continue
        changed.append(state.crystal_id)
        if prev.visible:
            dirty = rect_union(dirty, dilated_aabb(prev, width, height))
        if state.visible:
            dirty = rect_union(dirty, dilated_aabb(state, width, height))
    if dirty is None:
        return None, changed

    stable = False
    while not stable:
        stable = True
        for state in next_states:
            if state.crystal_id in changed or not state.visible:
                continue
            aabb = dilated_aabb(state, width, height)
            if rects_intersect(aabb, dirty) and not _rect_contains(dirty, aabb):
                dirty = rect_union(dirty, aabb)
                stable = False
    return dirty, changed


def _rect_contains(outer, inner) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


class Pipeline:
    def __init__(self, config: SceneConfig, incremental: bool):
        self.config = config
        self.incremental = incremental
        self.env = env_background(config)
        self.acc: np.ndarray | None = None
        self.prev_states: list[CrystalState] | None = None
        self.records: list[FrameRecord] = []

    def render_frame(self, step: int) -> np.ndarray:
        states = scene_states(self.config, step)
        sequence = draw_sequence(states)
        full_region = (0, 0, self.config.width, self.config.height)

        if not self.incremental or self.acc is None or self.prev_states is None:
            self.acc = self.env.copy()
            shaded = render_region(states, self.env, full_region, self.acc)
            record = FrameRecord(step, shaded, full_region, [], sequence)
        else:
            dirty, changed = compute_dirty_rect(
                self.prev_states, states, self.config.width, self.config.height
            )
            if dirty is None:
                shaded = 0
            else:
                x0, y0, x1, y1 = dirty
                self.acc[y0:y1, x0:x1, :] = self.env[y0:y1, x0:x1, :]
                shaded = render_region(states, self.env, dirty, self.acc)
            record = FrameRecord(step, shaded, dirty, changed, sequence)

        self.prev_states = states
        self.records.append(record)
        return to_framebuffer(self.acc)

    def load_accumulator(self, acc: np.ndarray, step: int) -> None:
        self.acc = acc.astype(np.float64).copy()
        self.prev_states = scene_states(self.config, step)
