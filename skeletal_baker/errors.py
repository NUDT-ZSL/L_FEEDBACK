"""Structured business error reporting (JSON Lines).

Business errors (E_* codes) are collected per frame and never abort the
batch; they are flushed to errors.log atomically at the end of the run.
"""

from __future__ import annotations

import json

E_NON_FINITE_MATRIX = "E_NON_FINITE_MATRIX"
E_NON_INVERTIBLE_MATRIX = "E_NON_INVERTIBLE_MATRIX"
E_DEGENERATE_TRIANGLE = "E_DEGENERATE_TRIANGLE"
E_FRAME_FULLY_CULLED = "E_FRAME_FULLY_CULLED"

MAX_DEGENERATE_REPORTS_PER_FRAME = 10


class ErrorLog:
    """Collects structured error entries and writes them as JSON Lines."""

    def __init__(self) -> None:
        self.entries: list[dict] = []
        self._degenerate_counts: dict[int, int] = {}

    def report(self, frame: int, code: str, message: str) -> None:
        self.entries.append(
            {"frame": int(frame), "code": code, "message": str(message)}
        )

    def report_degenerate_triangle(
        self, frame: int, tri_index: int, area: float
    ) -> None:
        """Report a degenerate triangle, capped per frame to limit volume."""
        count = self._degenerate_counts.get(frame, 0)
        if count >= MAX_DEGENERATE_REPORTS_PER_FRAME:
            self._degenerate_counts[frame] = count + 1
            return
        self._degenerate_counts[frame] = count + 1
        self.report(
            frame,
            E_DEGENERATE_TRIANGLE,
            f"triangle {tri_index} has near-zero area ({area:.3e}); skipped",
        )

    def to_jsonl(self) -> bytes:
        lines = [json.dumps(e, ensure_ascii=False, sort_keys=True)
                 for e in self.entries]
        return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
