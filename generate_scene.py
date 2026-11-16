#!/usr/bin/env python3
from __future__ import annotations

import argparse

from soft_renderer.sample_scene import write_indoor_scene


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the deterministic indoor sample scene.")
    parser.add_argument("output", help="destination scene JSON")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    arguments = parser.parse_args()
    if arguments.width <= 0 or arguments.height <= 0:
        parser.error("width and height must be positive")
    write_indoor_scene(arguments.output, arguments.width, arguments.height)
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
