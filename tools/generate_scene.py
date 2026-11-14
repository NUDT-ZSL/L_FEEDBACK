#!/usr/bin/env python3
"""Generate the deterministic sample scene used by the renderer."""

import json
import math
from pathlib import Path


def color_from_hue(hue: float, saturation: float, value: float) -> list[float]:
    hue = hue % 1.0
    sector = hue * 6.0
    fraction = sector - math.floor(sector)
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))
    index = int(sector) % 6
    colors = [
        (value, t, p),
        (q, value, p),
        (p, value, t),
        (p, q, value),
        (t, p, value),
        (value, p, q),
    ]
    red, green, blue = colors[index]
    return [round(red, 6), round(green, 6), round(blue, 6)]


def main() -> None:
    point_lights = []
    for index in range(64):
        x = (index % 8 - 3.5) * 2.55
        z = (index // 8 - 3.5) * 2.1
        height = 2.2 + 0.35 * math.sin(index * 0.71)
        hue = index / 64.0
        point_lights.append(
            {
                "id": f"p{index:02d}",
                "type": "point",
                "position": [round(x, 5), round(height, 5), round(z, 5)],
                "color": color_from_hue(hue, 0.72, 1.0),
                "intensity": round(2.2 + 0.8 * (1.0 - (index % 7) / 6.0), 5),
                "radius": round(4.2 + 0.22 * (index % 5), 5),
            }
        )

    directional_lights = []
    for index in range(8):
        angle = (index / 8.0) * math.tau
        direction = [round(math.cos(angle) * 0.55, 6), -1.0, round(math.sin(angle) * 0.55, 6)]
        directional_lights.append(
            {
                "id": f"d{index:02d}",
                "type": "directional",
                "direction": direction,
                "color": color_from_hue(index / 8.0 + 0.08, 0.35, 1.0),
                "intensity": round(0.10 + index * 0.012, 6),
            }
        )

    scene = {
        "name": "deterministic-orb-field",
        "scene_id": "deterministic-orb-field-v1",
        "light_version": 1,
        "camera": {
            "position": [0.0, 10.5, 19.0],
            "target": [0.0, 1.4, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov_y": 43.0,
            "near": 0.1,
            "far": 200.0,
        },
        "ambient": [0.018, 0.022, 0.032],
        "exposure": 1.05,
        "meshes": [
            {
                "name": "floor",
                "primitive": "plane",
                "center": [0.0, 0.0, 0.0],
                "size": [32.0, 1.0, 32.0],
                "material": {"albedo": [0.12, 0.145, 0.18], "specular": 0.08},
            },
            {
                "name": "orb-field",
                "primitive": "sphere_grid",
                "center": [0.0, 0.48, 0.0],
                "count_x": 24,
                "count_z": 20,
                "spacing": 0.9,
                "radius": 0.34,
                "slices": 48,
                "stacks": 24,
                "material": {"albedo": [0.52, 0.48, 0.46], "specular": 0.34},
            },
        ],
        "lights": point_lights + directional_lights,
    }

    output = Path(__file__).resolve().parents[1] / "sample_scene.json"
    output.write_text(json.dumps(scene, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
