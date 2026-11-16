#!/usr/bin/env python3
"""Generate deterministic short WAV sources and a demo skill configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.fixture_factory import make_base_wav, skill_entry, write_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target",
        nargs="?",
        default="demo",
        help="directory for generated assets and config (default: demo)",
    )
    args = parser.parse_args()
    target = Path(args.target).expanduser().resolve()
    assets = target / "assets"
    marker_fire = make_base_wav(
        assets / "fireball_base.wav",
        duration=1.5,
        marker_seconds=0.12,
        seed=197,
    )
    marker_frost = make_base_wav(
        assets / "frost_nova_base.wav",
        duration=1.4,
        marker_seconds=0.085,
        seed=991,
    )
    skills = [
        skill_entry(
            "fireball",
            "assets/fireball_base.wav",
            marker_fire,
            variant_count=4,
            random_seed=1234,
        ),
        skill_entry(
            "frost_nova",
            "assets/frost_nova_base.wav",
            marker_frost,
            variant_count=4,
            random_seed=98765,
        ),
    ]
    config_path = write_config(
        target / "config.json",
        skills,
        output_dir="out",
        temp_dir="tmp",
        concurrency=2,
    )
    print(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
