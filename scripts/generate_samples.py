#!/usr/bin/env python3
"""Generate deterministic local WAV samples and project JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dialogue_mixer.samplegen import generate_all


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("samples"),
        help="sample root directory (default: samples)",
    )
    args = parser.parse_args()
    generated = generate_all(args.out)
    for name, path in generated.items():
        print("{0}: {1}".format(name, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

