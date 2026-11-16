#!/usr/bin/env python3
"""Write the deterministic built-in sample scene to JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deferred_validation.default_scene import build_scene


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        default="examples/default_scene.json",
        help="output JSON path (default: examples/default_scene.json)",
    )
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_scene(), indent=2), encoding="utf-8")
    print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
