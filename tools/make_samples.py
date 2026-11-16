#!/usr/bin/env python3
"""Generate deterministic sample and error-path WAV fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpm_grains.samples import generate_fixture_set


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="samples")
    args = parser.parse_args()
    paths = generate_fixture_set(Path(args.directory))
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

