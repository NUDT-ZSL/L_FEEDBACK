"""Deterministic sample scene used for local verification and demos."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _rounded_points(points: list[tuple[float, float]]) -> list[list[float]]:
    return [[round(x, 4), round(y, 4)] for x, y in points]


def _self_intersecting_star(
    center_x: float,
    center_y: float,
    radius: float,
    points: int = 5,
) -> list[tuple[float, float]]:
    """Connect every second vertex of a pentagon to form a pentagram."""

    result: list[tuple[float, float]] = []
    for index in range(points):
        angle = -math.pi / 2.0 + index * 4.0 * math.pi / points
        result.append(
            (
                center_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle),
            )
        )
    return result


def _rectangle(
    x0: float, y0: float, x1: float, y1: float
) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def build_sample_scene() -> dict[str, Any]:
    ground_high = [
        (2, 6),
        (42, 3),
        (88, 6),
        (125, 18),
        (126, 70),
        (101, 89),
        (51, 93),
        (14, 82),
        (1, 47),
    ]
    ground_medium = [
        (2, 6),
        (65, 3),
        (125, 18),
        (126, 70),
        (72, 93),
        (1, 47),
    ]
    ground_low = [
        (2, 6),
        (125, 18),
        (126, 70),
        (72, 93),
        (1, 47),
    ]

    river_top = [
        (-28, 29),
        (7, 23),
        (35, 27),
        (61, 35),
        (83, 38),
        (108, 30),
        (132, 26),
        (157, 31),
    ]
    river_bottom = [
        (157, 50),
        (132, 45),
        (108, 49),
        (83, 57),
        (61, 54),
        (35, 46),
        (7, 42),
        (-28, 49),
    ]
    river_high = river_top + river_bottom
    river_medium = river_top[::2] + river_bottom[::2]
    river_low = [
        (-28, 29),
        (35, 27),
        (83, 38),
        (157, 31),
        (157, 50),
        (83, 57),
        (35, 46),
        (-28, 49),
    ]

    road_high = [
        (-25, 78),
        (2, 74),
        (18, 68),
        (31, 63),
        (47, 62),
        (63, 66),
        (79, 72),
        (95, 74),
        (111, 69),
        (126, 60),
        (139, 54),
        (151, 55),
        (160, 60),
    ]
    road_medium = [
        (-25, 78),
        (18, 68),
        (47, 62),
        (79, 72),
        (111, 69),
        (139, 54),
        (160, 60),
    ]
    road_low = [
        (-25, 78),
        (47, 62),
        (95, 74),
        (139, 54),
        (160, 60),
    ]

    buildings = [
        ("house-a", _rectangle(80, 18, 96, 31)),
        ("house-b", _rectangle(100, 20, 119, 34)),
        ("house-c", _rectangle(74, 68, 88, 80)),
    ]
    building_shapes: list[dict[str, Any]] = []
    for building_id, points in buildings:
        levels = _rounded_points(points)
        building_shapes.append(
            {
                "id": building_id,
                "kind": "polygon",
                "geometry": {"high": levels, "medium": levels, "low": levels},
            }
        )

    return {
        "version": 1,
        "name": "deterministic-terrain-128x96",
        "normal_strength": 1.25,
        "layers": [
            {
                "id": "terrain-ground",
                "color": [79, 105, 55],
                "alpha": 0.9,
                "shapes": [
                    {
                        "id": "ground",
                        "kind": "polygon",
                        "geometry": {
                            "high": _rounded_points(ground_high),
                            "medium": _rounded_points(ground_medium),
                            "low": _rounded_points(ground_low),
                        },
                        "simplify": {"merge_distance": 1.5, "tolerance": 1.0},
                    }
                ],
            },
            {
                "id": "river",
                "color": [42, 117, 181],
                "alpha": 0.72,
                "shapes": [
                    {
                        "id": "river-band",
                        "kind": "polygon",
                        "geometry": {
                            "high": _rounded_points(river_high),
                            "medium": _rounded_points(river_medium),
                            "low": _rounded_points(river_low),
                        },
                        "simplify": {"merge_distance": 1.25, "tolerance": 1.0},
                    }
                ],
            },
            {
                "id": "buildings",
                "color": [186, 176, 147],
                "alpha": 0.96,
                "shapes": building_shapes,
            },
            {
                "id": "road",
                "color": [190, 153, 91],
                "alpha": 0.88,
                "shapes": [
                    {
                        "id": "main-road",
                        "kind": "polyline",
                        "width": 4.0,
                        "geometry": {
                            "high": _rounded_points(road_high),
                            "medium": _rounded_points(road_medium),
                            "low": _rounded_points(road_low),
                        },
                        "simplify": {"merge_distance": 2.0, "tolerance": 1.5},
                    }
                ],
            },
            {
                "id": "plaza",
                "color": [224, 181, 75],
                "alpha": 0.68,
                "shapes": [
                    {
                        "id": "self-intersecting-star",
                        "kind": "polygon",
                        "geometry": {
                            "high": _rounded_points(
                                _self_intersecting_star(94, 50, 17)
                            ),
                            "medium": _rounded_points(
                                _self_intersecting_star(94, 50, 17)
                            ),
                            "low": _rounded_points(
                                _self_intersecting_star(94, 50, 17)
                            ),
                        },
                        "simplify": {"merge_distance": 0.5, "tolerance": 0.5},
                    }
                ],
            },
        ],
    }


def write_sample(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(build_sample_scene(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the deterministic sample scene.")
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args(argv)
    path = write_sample(args.output)
    print(
        json.dumps(
            {"ok": True, "input": str(path.resolve())},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
