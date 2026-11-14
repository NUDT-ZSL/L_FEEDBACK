"""Generate a deterministic local demo project for the mixer CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mixer.samples import generate_sample  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=ROOT / "examples" / "demo",
        help="demo project directory",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="run the mixer after generating the fixture",
    )
    args = parser.parse_args()

    destination = args.dest.resolve()
    media_dir = destination / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    media_specs = [
        ("s1_dialogue.wav", 1000, 44_100, 2, 0.16, "dialogue", 0),
        ("s1_ambience.wav", 1000, 48_000, 1, 0.10, "ambience", 1),
        ("s1_music.wav", 1000, 44_100, 2, 0.09, "music", 2),
        ("s2_dialogue.wav", 900, 48_000, 1, 0.18, "dialogue", 3),
        ("s2_ambience.wav", 900, 44_100, 2, 0.11, "ambience", 4),
        ("s2_music.wav", 900, 48_000, 2, 0.10, "music", 5),
        ("s3_dialogue.wav", 800, 44_100, 2, 0.22, "dialogue", 6),
        ("s3_ambience.wav", 800, 48_000, 1, 0.12, "ambience", 7),
        ("s3_music.wav", 800, 44_100, 2, 0.11, "music", 8),
    ]
    for name, duration, rate, channels, amplitude, track, seed in media_specs:
        generate_sample(
            media_dir / name,
            duration_ms=duration,
            sample_rate=rate,
            channels=channels,
            amplitude=amplitude,
            frequency=220.0 + 55.0 * seed,
            track_type=track,
            seed=seed,
        )

    def track(kind: str, shot: str) -> dict:
        return {
            "type": kind,
            "path": f"media/{shot}_{kind}.wav",
            "gain_db": 0.0,
            "fade_in_ms": 30,
            "fade_out_ms": 30,
        }

    shots = {
        "schema_version": 1,
        "time_unit": "ms",
        "target_sample_rate": 48_000,
        "output_channels": 4,
        "output_subtype": "PCM_16",
        "bus_weights": {
            "dialogue": 1.0,
            "ambience": 0.7,
            "music": 0.5,
        },
        "shots": [
            {
                "id": "s1",
                "start_ms": 0,
                "end_ms": 1000,
                "tracks": [track(kind, "s1") for kind in ("dialogue", "ambience", "music")],
            },
            {
                "id": "s2",
                "start_ms": 900,
                "end_ms": 1800,
                "allow_overlap": True,
                "crossfade_ms": 100,
                "tracks": [track(kind, "s2") for kind in ("dialogue", "ambience", "music")],
            },
            {
                "id": "s3",
                "start_ms": 1800,
                "end_ms": 2600,
                "tracks": [track(kind, "s3") for kind in ("dialogue", "ambience", "music")],
            },
        ],
    }
    _write_json(destination / "shots.json", shots)
    _write_json(destination / "choices.json", {"path": ["s1", "s2", "s3"]})

    print(f"generated demo project: {destination}")
    print(
        "run: python -m mixer --config "
        f"{destination / 'shots.json'} --path {destination / 'choices.json'} "
        f"--out {destination / 'mix.wav'} --report {destination / 'report.json'}"
    )

    if args.run:
        return subprocess.call(
            [
                sys.executable,
                "-m",
                "mixer",
                "--config",
                str(destination / "shots.json"),
                "--path",
                str(destination / "choices.json"),
                "--out",
                str(destination / "mix.wav"),
                "--report",
                str(destination / "report.json"),
            ],
            cwd=str(ROOT),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
