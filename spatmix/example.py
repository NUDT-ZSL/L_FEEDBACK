"""Deterministic example scene exercising every pipeline feature."""


def build_example_scene():
    return {
        "version": 1,
        "sample_rate": 48000,
        "duration": 2.0,
        "frame_size": 256,
        "layout": "5.1",
        "listener": {
            "keyframes": [
                {"t": 0.0, "pos": [-40.0, 0.0, 1.6], "heading": 90.0},
                {"t": 1.0, "pos": [0.0, 0.0, 1.6], "heading": 90.0},
                {"t": 2.0, "pos": [40.0, 0.0, 1.6], "heading": 90.0},
            ]
        },
        "sources": [
            {
                "id": "sine_front",
                "waveform": "sine",
                "frequency": 440.0,
                "duration": 2.0,
                "pos": [0.0, 10.0, 1.5],
                "amplitude": 0.8,
            },
            {
                "id": "noise_side",
                "waveform": "noise",
                "frequency": 1000.0,
                "duration": 2.0,
                "pos": [5.0, -8.0, 1.0],
                "amplitude": 0.5,
                "material": "concrete",
            },
            {
                "id": "impulse_box",
                "waveform": "impulse",
                "frequency": 8.0,
                "duration": 2.0,
                "pos": [0.0, 2.0, 1.0],
                "amplitude": 0.9,
                "material": "unobtanium",
            },
        ],
        "obstacles": [
            {
                "id": "wall",
                "min": [-1.0, 4.0, -1.0],
                "max": [1.0, 6.0, 3.0],
                "occlusion": 0.8,
            },
            {
                "id": "impulse_housing",
                "min": [-0.5, 1.5, 0.5],
                "max": [0.5, 2.5, 1.5],
                "occlusion": 0.6,
            },
        ],
        "reflectors": [
            {"id": "floor", "axis": "z", "value": 0.0, "absorption": 0.3},
            {"id": "far_wall", "axis": "x", "value": 500.0, "absorption": 0.5},
        ],
        "materials": {
            "concrete": {"air_absorption": 0.02},
        },
        "config": {
            "speed_of_sound": 343.0,
            "reference_distance": 4.0,
            "max_audible_distance": 30.0,
            "air_absorption": 0.004,
            "doppler_min": 0.9,
            "doppler_max": 1.1,
            "doppler_smoothing": 0.03,
            "min_reflection_gain": 0.001,
            "snapshot_interval": 50,
        },
    }
