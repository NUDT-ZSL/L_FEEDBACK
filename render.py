#!/usr/bin/env python3
"""Command-line entry point for the deterministic offline prism path tracer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
import traceback

from prism_tracer.errors import BusinessError, business_error
from prism_tracer.pipeline import (
    RenderConfig,
    automatic_caustic_photons,
    run_pipeline,
)
from prism_tracer.scene import build_scene


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise business_error("E_USAGE", message)


def _parse_resolution(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"([0-9]+)x([0-9]+)", value)
    if not match:
        raise business_error(
            "E_RESOLUTION",
            f"invalid --res {value!r}; expected WIDTHxHEIGHT, for example 320x240",
        )
    width = int(match.group(1))
    height = int(match.group(2))
    if width < 1 or height < 1 or width > 4096 or height > 4096:
        raise business_error(
            "E_RESOLUTION", "--res width and height must each be in [1, 4096]"
        )
    return width, height


def _integer(value: str, code: str, name: str) -> int:
    try:
        return int(value, 10)
    except ValueError as exc:
        raise business_error(code, f"{name} must be an integer, got {value!r}") from exc


def _build_config(args: argparse.Namespace) -> RenderConfig:
    width, height = _parse_resolution(args.res)
    spp = _integer(args.spp, "E_SPP", "--spp")
    seed = _integer(args.seed, "E_SEED", "--seed")
    tile = _integer(args.tile, "E_TILE", "--tile")
    max_depth = _integer(args.max_depth, "E_DEPTH", "--max-depth")
    photon_arg = _integer(args.caustic_photons, "E_CAUSTIC_PHOTONS", "--caustic-photons")
    map_res = _integer(args.caustic_map_resolution, "E_USAGE", "--caustic-map-res")

    if spp <= 0 or spp > 65536:
        raise business_error("E_SPP", "--spp must be in [1, 65536]")
    if seed < 0 or seed > (1 << 63) - 1:
        raise business_error("E_SEED", "--seed must be in [0, 2^63-1]")
    if tile <= 0 or tile > 4096:
        raise business_error("E_TILE", "--tile must be in [1, 4096]")
    if max_depth <= 0 or max_depth > 64:
        raise business_error("E_DEPTH", "--max-depth must be in [1, 64]")
    if photon_arg < -1 or photon_arg > 5_000_000:
        raise business_error(
            "E_CAUSTIC_PHOTONS",
            "--caustic-photons must be -1 (auto), 0 (disabled), or in [1, 5000000]",
        )
    if map_res < 16 or map_res > 2048:
        raise business_error(
            "E_USAGE", "--caustic-map-res must be in [16, 2048]"
        )
    try:
        exposure = float(args.exposure)
    except ValueError as exc:
        raise business_error("E_USAGE", "--exposure must be numeric") from exc
    if not (exposure > 0.0) or exposure > 1024.0:
        raise business_error("E_USAGE", "--exposure must be in (0, 1024]")

    photons = (
        automatic_caustic_photons(width, height)
        if photon_arg == -1
        else photon_arg
    )
    return RenderConfig(
        width=width,
        height=height,
        spp=spp,
        seed=seed,
        tile=tile,
        max_depth=max_depth,
        use_indirect=not args.no_indirect,
        caustic_photons=photons,
        caustic_map_resolution=map_res,
        exposure=exposure,
    )


def _make_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="render.py",
        description=(
            "Deterministic offline path tracer for the built-in glass-prism/water scene."
        ),
    )
    parser.add_argument("--scene", default="prism", help="built-in scene name (prism)")
    parser.add_argument(
        "--res", default="320x240", help="image resolution as WIDTHxHEIGHT"
    )
    parser.add_argument("--spp", default="64", help="samples per pixel")
    parser.add_argument("--seed", default="7", help="non-negative integer RNG seed")
    parser.add_argument(
        "--tile", default="64", help="square progressive-render tile edge in pixels"
    )
    parser.add_argument(
        "--max-depth", default="8", help="maximum camera path surface events"
    )
    parser.add_argument(
        "--no-indirect",
        action="store_true",
        help="disable diffuse indirect bounces while retaining direct light and caustics",
    )
    parser.add_argument(
        "--caustic-photons",
        default="-1",
        help="forward glass photons; -1 is resolution-scaled automatic",
    )
    parser.add_argument(
        "--caustic-map-res",
        default="192",
        dest="caustic_map_resolution",
        help="caustic splat/query map resolution",
    )
    parser.add_argument(
        "--exposure", default="1.35", help="linear exposure before tone mapping"
    )
    parser.add_argument("--out", default="out", help="output directory")
    parser.add_argument("--quiet", action="store_true", help="suppress progress text")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        parser = _make_parser()
        args = parser.parse_args(argv)
        config = _build_config(args)
        scene = build_scene(args.scene)

        def progress(message: str) -> None:
            if not args.quiet:
                print(message, file=sys.stderr, flush=True)

        report = run_pipeline(
            scene,
            config,
            Path(args.out),
            progress=progress,
        )
        if not args.quiet:
            print(
                f"rendered {config.width}x{config.height}, "
                f"{config.spp} spp, seed {config.seed}, "
                f"caustic photons {config.caustic_photons}",
                file=sys.stderr,
            )
            print(f"image: {Path(args.out) / report['outputs']['image']}")
            print(f"aux:   {Path(args.out) / report['outputs']['auxiliary']}")
            print(f"report:{Path(args.out) / report['outputs']['report']}")
            print(f"image_sha256: {report['hashes']['image_sha256']}")
        return 0
    except BusinessError as exc:
        print(f"ERROR [{exc.code}] {exc.message}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print(
            "ERROR [E_INTERRUPTED] render interrupted; completed tiles are checkpointed",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:  # pragma: no cover - last-resort CLI contract
        if os.environ.get("PRISM_TRACER_DEBUG"):
            traceback.print_exc()
        print(f"ERROR [E_INTERNAL] unexpected failure: {exc}", file=sys.stderr)
        return 70


if __name__ == "__main__":
    sys.exit(main())

