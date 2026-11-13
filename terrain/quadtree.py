"""Fixed-world quadtree tile keys, bounds and neighbour lookup."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TileKey:
    level: int
    ix: int
    iz: int

    def child(self, cx: int, cz: int) -> "TileKey":
        return TileKey(self.level + 1, self.ix * 2 + cx, self.iz * 2 + cz)

    def children(self) -> tuple["TileKey", "TileKey", "TileKey", "TileKey"]:
        return (
            self.child(0, 0),
            self.child(1, 0),
            self.child(0, 1),
            self.child(1, 1),
        )

    def bounds(self, root_min: tuple[float, float], root_size: float):
        """Return ``(x0, z0, x1, z1)`` world bounds in float64."""
        span = root_size / (1 << self.level)
        x0 = root_min[0] + self.ix * span
        z0 = root_min[1] + self.iz * span
        return x0, z0, x0 + span, z0 + span

    def span(self, root_size: float) -> float:
        return root_size / (1 << self.level)


def root_key() -> TileKey:
    return TileKey(0, 0, 0)


def neighbour(key: TileKey, direction: str) -> TileKey | None:
    """Same-level 4-neighbour, or ``None`` outside the root square.

    Directions: ``"N"`` (negative z), ``"S"`` (positive z),
    ``"W"`` (negative x), ``"E"`` (positive x).
    """
    ix, iz, level = key.ix, key.iz, key.level
    if direction == "N":
        iz -= 1
    elif direction == "S":
        iz += 1
    elif direction == "W":
        ix -= 1
    elif direction == "E":
        ix += 1
    else:
        raise ValueError(f"unknown direction {direction!r}")
    if ix < 0 or iz < 0 or ix >= (1 << level) or iz >= (1 << level):
        return None
    return TileKey(level, ix, iz)


def find_leaf(leaves: frozenset[TileKey], key: TileKey) -> TileKey | None:
    """Locate the leaf tile covering ``key``'s cell.

    ``key`` may be at any depth; the deepest ancestor (including ``key``
    itself) that belongs to ``leaves`` is returned.
    """
    candidate = key
    while candidate.level >= 0:
        if candidate in leaves:
            return candidate
        if candidate.level == 0:
            break
        candidate = TileKey(
            candidate.level - 1, candidate.ix // 2, candidate.iz // 2
        )
    return None


def adjacent_leaf(
    leaves: frozenset[TileKey], key: TileKey, direction: str
) -> TileKey | None:
    """Single leaf sharing ``key``'s edge, or ``None`` at border / split edge.

    When the neighbouring region is subdivided (several finer tiles touch the
    edge), no single leaf covers it and ``None`` is returned; callers needing
    all neighbours should use :func:`adjacent_leaves`.
    """
    same_level = neighbour(key, direction)
    if same_level is None:
        return None
    leaf = find_leaf(leaves, same_level)
    if leaf is not None and leaf.level == key.level:
        return leaf
    return None


def adjacent_leaves(
    leaves: frozenset[TileKey], key: TileKey, direction: str
) -> tuple[TileKey, ...]:
    """All leaf tiles touching ``key``'s edge in ``direction`` (border-safe)."""
    max_level = max((tile.level for tile in leaves), default=key.level)
    scale = 1 << (max_level - key.level)
    if direction in ("N", "S"):
        deep_iz = (
            key.iz * scale - 1 if direction == "N" else (key.iz + 1) * scale
        )
        deep_coords = (
            (key.ix * scale + offset, deep_iz)
            for offset in range(scale)
        )
    else:
        deep_ix = (
            key.ix * scale - 1 if direction == "W" else (key.ix + 1) * scale
        )
        deep_coords = (
            (deep_ix, key.iz * scale + offset)
            for offset in range(scale)
        )
    result: set[TileKey] = set()
    side = deep_iz if direction in ("N", "S") else deep_ix
    border_limit = 1 << max_level
    if side < 0 or side >= border_limit:
        return ()
    for deep_ix, deep_iz in deep_coords:
        leaf = find_leaf(leaves, TileKey(max_level, deep_ix, deep_iz))
        if leaf is not None:
            result.add(leaf)
    return tuple(sorted(result, key=lambda tile: (tile.level, tile.ix, tile.iz)))
