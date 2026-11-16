"""Deterministic multi-material, multi-light sample scene."""

from __future__ import annotations

import numpy as np

from .primitives import (
    make_box,
    make_plane,
    make_sphere,
    make_tiling_normal_map,
    mesh_to_json,
)


def build_scene() -> dict:
    normal_map = make_tiling_normal_map(32, 32, 0.08)
    normal_rgba = (normal_map.astype(np.float64) / 255.0).round(4)
    return {
        "version": 1,
        "camera": {
            "position": [4.6, 3.25, 6.8],
            "target": [0.0, 0.85, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov_y_deg": 42.0,
            "near": 0.05,
            "far": 50.0,
        },
        "rendering": {
            "width": 400,
            "height": 300,
            "ssaa": 2,
            "exposure": 1.05,
            "tone_mapper": "aces",
            "shadows": {
                "enabled": True,
                "directional_resolution": 512,
                "spot_resolution": 512,
                "point_resolution": 256,
                "bias": 0.0015,
                "normal_bias": 0.004,
                "filter_radius": 1,
                "filter_taps": 3,
            },
        },
        "indirect": {
            "enabled": True,
            "probe": {
                "type": "hemisphere",
                "sky_color": [0.18, 0.26, 0.38],
                "ground_color": [0.045, 0.035, 0.03],
            },
            "intensity": 0.8,
        },
        "meshes": {
            "floor": mesh_to_json(make_plane(6.0, 2.0)),
            "rounded_box": mesh_to_json(make_box(1.2)),
            "sphere_a": mesh_to_json(make_sphere(0.72, 20, 32)),
            "sphere_b": mesh_to_json(make_sphere(0.47, 16, 28)),
            "sphere_c": mesh_to_json(make_sphere(0.30, 14, 24)),
        },
        "materials": {
            "floor": {
                "albedo": [0.52, 0.56, 0.62],
                "metallic": 0.0,
                "roughness": 0.52,
                "normal_strength": 0.55,
                "normal_map": {
                    "width": 32,
                    "height": 32,
                    "channels": 4,
                    "data": normal_rgba.reshape(-1).tolist(),
                },
            },
            "copper": {
                "albedo": [0.94, 0.40, 0.18],
                "metallic": 1.0,
                "roughness": 0.27,
            },
            "ceramic": {
                "albedo": [0.05, 0.42, 0.52],
                "metallic": 0.15,
                "roughness": 0.14,
            },
            "gold": {
                "albedo": [0.96, 0.72, 0.24],
                "metallic": 1.0,
                "roughness": 0.38,
            },
            "ruby": {
                "albedo": [0.62, 0.045, 0.10],
                "metallic": 0.05,
                "roughness": 0.24,
            },
        },
        "nodes": [
            {"name": "floor_node", "mesh": "floor", "material": "floor"},
            {
                "name": "cube_node",
                "mesh": "rounded_box",
                "material": "copper",
                "translation": [-1.20, 0.62, -0.15],
                "rotation_deg": [0.0, -22.0, 0.0],
                "scale": [1.0, 1.02, 1.0],
            },
            {
                "name": "sphere_a_node",
                "mesh": "sphere_a",
                "material": "ceramic",
                "translation": [1.62, 0.72, 0.72],
            },
            {
                "name": "sphere_b_node",
                "mesh": "sphere_b",
                "material": "gold",
                "translation": [0.28, 0.48, -1.85],
            },
            {
                "name": "sphere_c_node",
                "mesh": "sphere_c",
                "material": "ruby",
                "translation": [-0.42, 1.50, 0.12],
            },
        ],
        "lights": [
            {
                "name": "warm_key",
                "type": "directional",
                "direction": [0.38, -0.80, -0.46],
                "color": [1.0, 0.91, 0.78],
                "intensity": 3.2,
                "casts_shadow": True,
            },
            {
                "name": "cool_fill",
                "type": "point",
                "position": [-2.35, 2.0, 1.55],
                "color": [0.18, 0.48, 1.0],
                "intensity": 10.0,
                "range": 12.0,
                "casts_shadow": True,
            },
            {
                "name": "ember_spot",
                "type": "spot",
                "position": [2.65, 3.5, 1.95],
                "direction": [-0.55, -0.78, -0.38],
                "color": [1.0, 0.30, 0.10],
                "intensity": 30.0,
                "range": 12.0,
                "inner_deg": 21.0,
                "outer_deg": 31.0,
                "casts_shadow": True,
            },
        ],
    }
