"""Command line interface for the CPU layered soft-shadow renderer."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .errors import (
    EXIT_CODES,
    RENDER_ERR_DETERMINISM,
    RENDER_ERR_SCHEMA,
    RenderError,
)
from .image_io import write_image
from .renderer import render_postprocessed
from .scene import load_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="softraster",
        description=(
            "Offline NumPy renderer for JSON scenes with layered materials "
            "and per-light soft shadows."
        ),
    )
    parser.add_argument("scene", help="input scene JSON path")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="output path (.npy float32 or .pfm float32)",
    )
    parser.add_argument(
        "--tone-map",
        "--tonemap",
        dest="tone_map",
        nargs="?",
        const="aces",
        choices=("aces", "reinhard"),
        default=None,
        help="enable tone mapping (default method: aces); omitted means off",
    )
    parser.add_argument(
        "--gamma",
        "--srgb-gamma",
        dest="gamma",
        action="store_true",
        help="enable sRGB transfer/gamma independently of tone mapping",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="render twice and require bit-exact float32 output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        scene = load_scene(args.scene)
        first = render_postprocessed(
            scene, tonemap=args.tone_map, gamma=args.gamma
        )
        if args.self_check:
            second = render_postprocessed(
                scene, tonemap=args.tone_map, gamma=args.gamma
            )
            difference = np.abs(
                first.image.astype(np.float64)
                - second.image.astype(np.float64)
            )
            different_pixels = int(
                np.count_nonzero(np.any(difference > 0.0, axis=2))
            )
            max_abs_diff = (
                float(np.max(difference)) if difference.size else 0.0
            )
            exact = bool(np.array_equal(first.image, second.image))
            report = {
                "deterministic": exact,
                "different_pixels": different_pixels,
                "max_abs_diff": max_abs_diff,
            }
            print(json.dumps(report, separators=(",", ":"), sort_keys=True))
            if not exact:
                return EXIT_CODES[RENDER_ERR_DETERMINISM]

        write_image(args.output, first.image)
        return 0
    except RenderError as error:
        print(
            json.dumps(
                error.as_json(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return EXIT_CODES.get(error.code, 1)
    except (OSError, ValueError, FloatingPointError) as error:
        payload = {
            "code": RENDER_ERR_SCHEMA,
            "message": str(error),
        }
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return EXIT_CODES[RENDER_ERR_SCHEMA]


if __name__ == "__main__":
    raise SystemExit(main())
