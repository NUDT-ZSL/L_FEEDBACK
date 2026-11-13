"""Horizontal frustum tiling and per-tile draw-coverage manifests."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .errors import BadTileWidthError


def make_tiles(width: int, tile_width: int) -> List[Tuple[int, int]]:
    """Split image width into (x0, x1) column bands.

    The last tile may be narrower when tile_width does not divide width;
    this is a defined result, not an error.
    """
    if tile_width <= 0:
        raise BadTileWidthError(
            "tile width must be a positive integer", tile_width=tile_width
        )
    if width <= 0:
        raise BadTileWidthError("image width must be positive", width=width)
    tiles = []
    x = 0
    while x < width:
        tiles.append((x, min(x + tile_width, width)))
        x += tile_width
    return tiles


def empty_manifest(tiles: List[Tuple[int, int]]) -> List[dict]:
    return [
        {
            "tile": i,
            "x0": x0,
            "x1": x1,
            "segments": {},
            "total_pixels": 0,
        }
        for i, (x0, x1) in enumerate(tiles)
    ]


def recount_tile(ids: np.ndarray, tile_index: int, x0: int, x1: int) -> dict:
    """Recount one tile region of the id buffer into a manifest entry."""
    sub = ids[:, x0:x1]
    valid = sub[sub >= 0]
    segs = {}
    total = 0
    if valid.size:
        vals, counts = np.unique(valid, return_counts=True)
        segs = {str(int(v)): int(c) for v, c in zip(vals, counts)}
        total = int(counts.sum())
    return {
        "tile": tile_index,
        "x0": x0,
        "x1": x1,
        "segments": segs,
        "total_pixels": total,
    }


def recount_from_ids(ids: np.ndarray, tiles: List[Tuple[int, int]]) -> List[dict]:
    """Recount the full id buffer into a manifest."""
    out = empty_manifest(tiles)
    for i, (x0, x1) in enumerate(tiles):
        out[i] = recount_tile(ids, i, x0, x1)
    return out


def cross_check(manifest: List[dict], ids: np.ndarray, tiles) -> List[dict]:
    """Compare a manifest against the actual pixel/id buffer.

    Returns a list of structured diffs; empty list means consistent.
    """
    actual = recount_from_ids(ids, tiles)
    diffs = []
    for exp, act in zip(manifest, actual):
        if exp["segments"] != act["segments"]:
            exp_segs = {int(k): v for k, v in exp["segments"].items()}
            act_segs = {int(k): v for k, v in act["segments"].items()}
            all_ids = sorted(set(exp_segs) | set(act_segs))
            diffs.append(
                {
                    "tile": exp["tile"],
                    "segments": [
                        {
                            "segment": sid,
                            "manifest_pixels": exp_segs.get(sid, 0),
                            "buffer_pixels": act_segs.get(sid, 0),
                            "delta": exp_segs.get(sid, 0) - act_segs.get(sid, 0),
                        }
                        for sid in all_ids
                        if exp_segs.get(sid, 0) != act_segs.get(sid, 0)
                    ],
                }
            )
    return diffs
