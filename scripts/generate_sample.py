#!/usr/bin/env python3
"""Create deterministic clean and degraded planning fixtures."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
LOCALES = ["zh-CN", "en-US", "ja-JP"]


def distributed_durations(weights: Iterable[float], total: int) -> list[int]:
    weight_list = list(weights)
    weight_total = sum(weight_list)
    result: list[int] = []
    previous = 0
    cumulative = 0.0
    for weight in weight_list:
        cumulative += weight
        boundary = round(cumulative * total / weight_total)
        result.append(boundary - previous)
        previous = boundary
    result[-1] += total - sum(result)
    if any(value <= 0 for value in result):
        raise ValueError("generated syllable duration must be positive")
    return result


def line(parts: list[str], weights: Iterable[float], total: int) -> dict:
    durations = distributed_durations(weights, total)
    return {
        "text": "".join(parts),
        "syllables": [
            {"text": part, "duration_samples": duration}
            for part, duration in zip(parts, durations)
        ],
    }


def role(gain_db: float, fundamental_hz: float, amplitude: float) -> dict:
    return {
        "gain_db": gain_db,
        "voice": {
            "fundamental_hz": fundamental_hz,
            "amplitude": amplitude,
            "vibrato_hz": 4.6,
            "vibrato_depth_rad": 0.006,
            "harmonics": [
                {"multiple": 1.0, "gain": 1.0, "phase_deg": 0.0},
                {"multiple": 2.0, "gain": 0.28, "phase_deg": 31.0},
                {"multiple": 3.0, "gain": 0.13, "phase_deg": -17.0},
                {"multiple": 4.5, "gain": 0.05, "phase_deg": 67.0},
            ],
        },
    }


ROLES = {
    "narrator": role(-2.0, 142.0, 0.28),
    "heroine": role(1.5, 219.0, 0.32),
    "rival": role(1.0, 188.0, 0.34),
    "guard": role(0.5, 121.0, 0.36),
}


CLEAN_LINES = {
    "scene_open": {
        "role": "narrator",
        "start_sample": 0,
        "duration_samples": 24000,
        "locales": {
            "zh-CN": (
                ["夜", "幕", "降", "临", "港", "口"],
                [1.0, 0.9, 1.1, 0.8, 1.0, 0.9],
                24000,
            ),
            "en-US": (
                ["Night", "falls", "over", "the", "harbor"],
                [1.0, 0.9, 1.0, 0.6, 1.2],
                21600,
            ),
            "ja-JP": (
                ["夜", "が", "港", "に", "降", "り", "る"],
                [1.0, 0.5, 1.0, 0.6, 1.0, 0.5, 0.8],
                22800,
            ),
        },
    },
    "market_heroine": {
        "role": "heroine",
        "start_sample": 12000,
        "duration_samples": 30000,
        "locales": {
            "zh-CN": (
                ["快", "看", "那", "边", "的", "灯"],
                [0.8, 0.9, 0.6, 0.9, 0.5, 1.2],
                28800,
            ),
            "en-US": (
                ["Look", "at", "those", "lights", "across", "the", "square"],
                [0.8, 0.5, 1.0, 1.0, 1.1, 0.5, 1.1],
                26400,
            ),
            "ja-JP": (
                ["あ", "の", "灯", "り", "を", "見", "て"],
                [0.7, 0.5, 1.0, 0.8, 0.6, 0.9, 0.8],
                27600,
            ),
        },
    },
    "market_rival": {
        "role": "rival",
        "start_sample": 18000,
        "duration_samples": 27000,
        "locales": {
            "zh-CN": (
                ["别", "靠", "近", "那", "扇", "门"],
                [0.8, 0.9, 1.0, 0.6, 0.9, 1.0],
                25200,
            ),
            "en-US": (
                ["Stay", "away", "from", "that", "door"],
                [1.0, 1.0, 0.8, 0.8, 1.1],
                24000,
            ),
            "ja-JP": (
                ["そ", "の", "扉", "に", "近", "づ", "く", "な"],
                [0.6, 0.5, 1.0, 0.6, 0.8, 1.0, 0.5, 0.8],
                25800,
            ),
        },
    },
    "market_guard": {
        "role": "guard",
        "start_sample": 21000,
        "duration_samples": 26000,
        "locales": {
            "zh-CN": (
                ["所", "有", "人", "退", "后"],
                [1.0, 0.9, 0.9, 1.0, 1.1],
                24500,
            ),
            "en-US": (
                ["Everyone", "step", "back", "now"],
                [1.2, 0.9, 0.9, 0.8],
                23400,
            ),
            "ja-JP": (
                ["全", "員", "下", "が", "れ"],
                [1.0, 1.0, 0.8, 0.6, 1.0],
                24000,
            ),
        },
    },
    "aftermath": {
        "role": "narrator",
        "start_sample": 54000,
        "duration_samples": 24000,
        "locales": {
            "zh-CN": (
                ["钟", "声", "仍", "在", "回", "响"],
                [0.8, 0.9, 0.9, 0.7, 0.9, 1.0],
                22800,
            ),
            "en-US": (
                ["The", "bell", "still", "echoes"],
                [0.7, 0.9, 0.9, 1.1],
                21600,
            ),
            "ja-JP": (
                ["鐘", "の", "音", "が", "響", "く"],
                [0.9, 0.5, 0.9, 0.6, 1.0, 0.8],
                22000,
            ),
        },
    },
}


def clean_slots() -> list[dict]:
    slots = []
    for slot_id, spec in CLEAN_LINES.items():
        lines = {}
        for locale, (parts, weights, total) in spec["locales"].items():
            lines[locale] = line(parts, weights, total)
        slots.append(
            {
                "slot_id": slot_id,
                "role": spec["role"],
                "start_sample": spec["start_sample"],
                "duration_samples": spec["duration_samples"],
                "lines": lines,
            }
        )
    return slots


def degraded_slots() -> list[dict]:
    valid_one = line(
        ["灯", "火", "不", "灭"], [1.0, 1.0, 0.8, 1.0], 24000
    )
    valid_two = line(
        ["Go", "while", "the", "road", "is", "clear"],
        [0.7, 1.0, 0.7, 0.9, 0.6, 1.0],
        20000,
    )
    valid_three = line(
        ["静", "か", "に", "進", "め"],
        [0.8, 0.7, 0.7, 1.0, 0.9],
        20000,
    )
    return [
        {
            "slot_id": "valid_first",
            "role": "heroine",
            "start_sample": 0,
            "duration_samples": 24000,
            "lines": {
                "zh-CN": valid_one,
                "en-US": line(
                    ["Keep", "the", "lantern", "burning"],
                    [1.0, 0.8, 1.1, 1.0],
                    24000,
                ),
                "ja-JP": line(
                    ["灯", "を", "消", "す", "な"],
                    [0.8, 0.6, 0.9, 0.6, 0.8],
                    24000,
                ),
            },
        },
        {
            "slot_id": "degraded_middle",
            "role": "rival",
            "start_sample": 24000,
            "duration_samples": 48000,
            "lines": {
                "zh-CN": {
                    "text": "   ",
                    "syllables": [{"text": "空", "duration_samples": 24000}],
                },
                # en-US is deliberately missing.
                "ja-JP": {
                    "text": "短すぎる",
                    "syllables": [
                        {"text": "短", "duration_samples": 6000},
                        {"text": "すぎる", "duration_samples": 6000},
                    ],
                },
            },
        },
        {
            "slot_id": "valid_last",
            "role": "narrator",
            "start_sample": 78000,
            "duration_samples": 20000,
            "lines": {
                "zh-CN": valid_two,
                "en-US": valid_two,
                "ja-JP": valid_three,
            },
        },
    ]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def manifest(project_id: str, table_paths: list[str]) -> dict:
    return {
        "schema_version": 1,
        "project_id": project_id,
        "locales": LOCALES,
        "roles": ROLES,
        "tables": table_paths,
    }


def main() -> None:
    config = {
        "schema_version": 1,
        "sample_rate": 48000,
        "channels": 1,
        "sample_format": "pcm_s16le",
        "stretch": {"min_ratio": 0.75, "max_ratio": 1.25},
        "synthesis": {
            "crossfade_ms": 4.0,
            "attack_ms": 6.0,
            "release_ms": 9.0,
        },
        "mix": {"target_peak_dbfs": -1.0},
        "pack": {"chunk_alignment": 64},
    }
    write_json(EXAMPLES / "config.json", config)

    clean_root = EXAMPLES / "clean"
    write_json(
        clean_root / "project.json",
        manifest(
            "harbor_night_clean",
            ["tables/scene01.json", "tables/scene02.json"],
        ),
    )
    clean = clean_slots()
    write_json(
        clean_root / "tables" / "scene01.json",
        {
            "schema_version": 1,
            "table_id": "scene01",
            "slots": clean[:3],
        },
    )
    write_json(
        clean_root / "tables" / "scene02.json",
        {
            "schema_version": 1,
            "table_id": "scene02",
            "slots": clean[3:],
        },
    )

    degraded_root = EXAMPLES / "degraded"
    write_json(
        degraded_root / "project.json",
        manifest("harbor_night_degraded", ["tables/scene.json"]),
    )
    write_json(
        degraded_root / "tables" / "scene.json",
        {
            "schema_version": 1,
            "table_id": "degraded_scene",
            "slots": degraded_slots(),
        },
    )


if __name__ == "__main__":
    main()
