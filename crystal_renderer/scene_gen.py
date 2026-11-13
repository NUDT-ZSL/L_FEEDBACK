"""Deterministic sample scene and anomaly-injection config generator."""

from __future__ import annotations

import json
import math
import random


def _crystal_polygon(rng, sides, radius):
    angles = sorted(rng.uniform(0.0, 2.0 * math.pi) for _ in range(sides))
    vertices = []
    for angle in angles:
        r = radius * rng.uniform(0.65, 1.0)
        vertices.append([round(r * math.cos(angle), 4), round(r * math.sin(angle), 4)])
    return vertices


def generate_scene(seed, frames=40, width=160, height=120):
    rng = random.Random(seed)
    crystals = []
    count = 6
    anchors = [
        (0.18, 0.25),
        (0.50, 0.18),
        (0.82, 0.28),
        (0.22, 0.78),
        (0.55, 0.62),
        (0.62, 0.72),
    ]
    for index in range(count):
        sides = rng.randint(3, 6)
        radius = rng.uniform(9.0, 15.0)
        anchor_x, anchor_y = anchors[index % len(anchors)]
        crystal = {
            "id": f"crystal_{index}",
            "layer": index,
            "vertices": _crystal_polygon(rng, sides, radius),
            "ior": round(rng.uniform(1.1, 1.6), 3),
            "color": [
                round(rng.uniform(0.2, 0.9), 3),
                round(rng.uniform(0.2, 0.9), 3),
                round(rng.uniform(0.4, 1.0), 3),
                round(rng.uniform(0.35, 0.75), 2),
            ],
            "transform": {
                "translate": [
                    round((anchor_x - 0.5 + rng.uniform(-0.02, 0.02)) * width, 3),
                    round((anchor_y - 0.5 + rng.uniform(-0.02, 0.02)) * height, 3),
                ],
                "rotate": round(rng.uniform(0.0, math.pi), 4),
                "scale": 1.0,
            },
            "depth_gradient": [
                round(rng.uniform(-0.02, 0.02), 5),
                round(rng.uniform(-0.02, 0.02), 5),
            ],
            "refraction_strength": round(rng.uniform(4.0, 9.0), 3),
            "visible": True,
            "keyframes": [],
        }
        crystals.append(crystal)
    mover = crystals[1]
    for step in range(3, frames, 4):
        mover["keyframes"].append({"step": step, "set": {"rotate_by": 0.35}})
    mover["keyframes"].append({"step": frames // 2, "set": {"color": [0.9, 0.3, 0.4, 0.6]}})
    blink = crystals[4]
    blink["keyframes"].append({"step": frames // 3, "set": {"visible": False}})
    blink["keyframes"].append({"step": (2 * frames) // 3, "set": {"visible": True}})
    return {
        "seed": seed,
        "frames": frames,
        "width": width,
        "height": height,
        "snapshot_every": 7,
        "camera": {"scale": 1.0, "rotate": 0.0, "translate": [width / 2.0, height / 2.0]},
        "env": {
            "color_top": [0.03, 0.04, 0.10],
            "color_bottom": [0.12, 0.06, 0.18],
            "stripe_color": [0.20, 0.26, 0.34],
            "stripe_frequency": 0.05,
            "stripe_amplitude": 0.6,
        },
        "crystals": crystals,
    }


def generate_anomaly_scene(seed, frames=12, width=96, height=72):
    scene = generate_scene(seed, frames=frames, width=width, height=height)
    good = scene["crystals"][:3]
    bad_degenerate = {
        "id": "bad_degenerate",
        "layer": 3,
        "vertices": [[0.0, 0.0], [10.0, 0.0]],
        "ior": 1.3,
        "color": [0.5, 0.5, 0.5, 0.5],
    }
    bad_zero_area = {
        "id": "bad_zero_area",
        "layer": 4,
        "vertices": [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]],
        "ior": 1.3,
        "color": [0.5, 0.5, 0.5, 0.5],
    }
    bad_ior = {
        "id": "bad_ior",
        "layer": 5,
        "vertices": [[0.0, 0.0], [8.0, 0.0], [4.0, 8.0]],
        "ior": 0.7,
        "color": [0.5, 0.5, 0.5, 0.5],
    }
    bad_keyframe = {
        "id": "bad_keyframe",
        "layer": 6,
        "vertices": [[0.0, 0.0], [9.0, 1.0], [4.0, 9.0]],
        "ior": 1.25,
        "color": [0.6, 0.4, 0.8, 0.5],
        "transform": {"translate": [10.0, 6.0], "rotate": 0.0, "scale": 1.0},
        "keyframes": [{"step": frames + 5, "set": {"rotate_by": 0.5}}],
    }
    scene["crystals"] = good + [bad_degenerate, bad_zero_area, bad_ior, bad_keyframe]
    return scene


def write_scene(path, scene):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(scene, handle, indent=2, sort_keys=True)
        handle.write("\n")
