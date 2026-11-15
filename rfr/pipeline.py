"""Draw-call description and the cached render-state machine.

The renderer first turns the scene graph into an ordered list of
:class:`DrawCall` objects (depth-first).  Both backends consume that exact same
list.  :class:`StateCache` tracks the four pieces of mutable pipeline state -
shader program, blend mode, depth test and face culling - and counts a state
switch only when a value actually changes.  The GPU backend forwards the
changes to OpenGL while the software backend keeps a logically equivalent
state machine and the same counters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


BLEND_MODES = ("opaque", "alpha", "additive")


@dataclass
class DrawCall:
    node_name: str
    positions: np.ndarray          # (V, 3) float32 world-space
    normals: np.ndarray            # (V, 3) float32 world-space
    colors: np.ndarray             # (V, 4) float32 lit linear RGBA
    indices: np.ndarray            # (T*3,) int32
    blend: str
    depth_test: bool
    cull: str                      # none | back | front
    program: str = "default"

    @property
    def triangle_count(self) -> int:
        return int(self.indices.shape[0] // 3)


class StateCache:
    """Mirrors GL pipeline state while counting the real change commands."""

    def __init__(self) -> None:
        self.program: Optional[str] = None
        self.blend: Optional[str] = None
        self.depth_test: Optional[bool] = None
        self.cull: Optional[str] = None
        self.switch_counts: Dict[str, int] = {
            "program": 0,
            "blend": 0,
            "depth_test": 0,
            "cull": 0,
        }
        self.total_switches = 0

    def _record(self, key: str, changed: bool) -> None:
        if changed:
            self.switch_counts[key] += 1
            self.total_switches += 1

    def set_program(self, program: str) -> bool:
        changed = program != self.program
        if changed:
            self.program = program
        self._record("program", changed)
        return changed

    def set_blend(self, blend: str) -> bool:
        if blend not in BLEND_MODES:
            raise ValueError(f"Unknown blend mode: {blend}")
        changed = blend != self.blend
        if changed:
            self.blend = blend
        self._record("blend", changed)
        return changed

    def set_depth_test(self, enabled: bool) -> bool:
        changed = bool(enabled) != self.depth_test
        if changed:
            self.depth_test = bool(enabled)
        self._record("depth_test", changed)
        return changed

    def set_cull(self, cull: str) -> bool:
        if cull not in ("none", "back", "front"):
            raise ValueError(f"Unknown cull mode: {cull}")
        changed = cull != self.cull
        if changed:
            self.cull = cull
        self._record("cull", changed)
        return changed

    def apply(self, draw: DrawCall) -> Dict[str, bool]:
        """Apply one draw call's state, returning which fields changed."""
        return {
            "program": self.set_program(draw.program),
            "blend": self.set_blend(draw.blend),
            "depth_test": self.set_depth_test(draw.depth_test),
            "cull": self.set_cull(draw.cull),
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "current": {
                "program": self.program,
                "blend": self.blend,
                "depth_test": self.depth_test,
                "cull": self.cull,
            },
            "switch_counts": dict(self.switch_counts),
            "total_switches": self.total_switches,
        }
