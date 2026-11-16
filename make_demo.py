from __future__ import annotations

import argparse
from pathlib import Path

from strip_renderer.demo import write_demo


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the deterministic demo.")
    parser.add_argument(
        "--out",
        default="examples/demo",
        help="directory for textures and scene JSON files",
    )
    parser.add_argument("--size", type=int, default=128, help="render resolution")
    parser.add_argument(
        "--texture-size",
        type=int,
        default=64,
        help="square texture resolution",
    )
    args = parser.parse_args()
    paths = write_demo(
        Path(args.out),
        size=args.size,
        texture_size=args.texture_size,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
