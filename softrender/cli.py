"""Command line interface: generate / render / run / verify."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
from PIL import Image

from . import assets, pipeline, tonemap
from .errors import ParameterError, SoftRenderError
from .raster import validate_threads


def _validate_resolution(width: int, height: int) -> None:
    for name, v in (("width", width), ("height", height)):
        if not (16 <= v <= 4096):
            raise ParameterError(f"{name} {v} out of range: must be between 16 and 4096")


def _validate_shadow_res(res: int) -> None:
    if res < 16 or res > 8192 or (res & (res - 1)) != 0:
        raise ParameterError(
            f"shadow map resolution {res} is invalid: must be a power of two "
            f"between 16 and 8192"
        )


def _check_common(args) -> None:
    _validate_resolution(args.width, args.height)
    _validate_shadow_res(args.shadow_res)
    validate_threads(args.threads)


def _write_png_atomic(path: str, rgb8: np.ndarray) -> None:
    """Write PNG via a temp file + rename: never leave a partial image."""
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, f".{os.path.basename(path)}.tmp")
    try:
        Image.fromarray(rgb8, mode="RGB").save(tmp, format="PNG")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _render_to_png(scene_path: str, output: str, width: int, height: int,
                   threads: int, shadow_res: int, quiet: bool = False) -> dict:
    hdr, timings = pipeline.render_scene(scene_path, width, height, threads, shadow_res)
    rgb8 = tonemap.hdr_to_rgb8(hdr)
    _write_png_atomic(output, rgb8)
    if not quiet:
        stats = "  ".join(f"{k}={v:.3f}s" for k, v in timings.items())
        print(f"render stats: {stats}")
        print(f"wrote {output} ({width}x{height}, threads={threads}, "
              f"shadow_res={shadow_res})")
    return timings


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_generate(args) -> int:
    scene_path = assets.generate_assets(args.dir, args.seed)
    print(f"generated assets in {args.dir} (seed={args.seed})")
    print(f"scene: {scene_path}")
    return 0


def cmd_render(args) -> int:
    _check_common(args)
    _render_to_png(args.scene, args.output, args.width, args.height,
                   args.threads, args.shadow_res)
    return 0


def cmd_run(args) -> int:
    _check_common(args)
    assets_dir = args.assets_dir
    if assets_dir is None:
        base = os.path.splitext(os.path.basename(args.output))[0]
        assets_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)),
                                  f"{base}_assets")
    scene_path = assets.generate_assets(assets_dir, args.seed)
    print(f"generated assets in {assets_dir} (seed={args.seed})")
    _render_to_png(scene_path, args.output, args.width, args.height,
                   args.threads, args.shadow_res)
    return 0


def cmd_verify(args) -> int:
    _check_common(args)
    os.makedirs(args.workdir, exist_ok=True)
    scene_path = assets.generate_assets(os.path.join(args.workdir, "assets"), args.seed)
    runs = [
        ("multithreaded run 1", args.threads, os.path.join(args.workdir, "out_mt_1.png")),
        ("multithreaded run 2", args.threads, os.path.join(args.workdir, "out_mt_2.png")),
        ("single-thread reference", 1, os.path.join(args.workdir, "out_ref.png")),
    ]
    hashes = []
    for label, threads, out in runs:
        _render_to_png(scene_path, out, args.width, args.height,
                       threads, args.shadow_res, quiet=True)
        digest = _sha256(out)
        hashes.append(digest)
        print(f"{label}: sha256={digest}  ({out})")
    ok = hashes[0] == hashes[1] == hashes[2]
    if ok:
        print("verify: PASS (all hashes identical)")
        return 0
    print("verify: FAIL (hash mismatch)", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="softrender",
        description="CPU software rasterizer for offline mech preview rendering.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--width", type=int, default=256)
        sp.add_argument("--height", type=int, default=256)
        sp.add_argument("--threads", type=int, default=1)
        sp.add_argument("--shadow-res", type=int, default=256)

    sp = sub.add_parser("generate", help="generate deterministic sample assets")
    sp.add_argument("--dir", required=True, help="output asset directory")
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_generate)

    sp = sub.add_parser("render", help="render an existing scene.json to PNG")
    sp.add_argument("--scene", required=True, help="path to scene.json")
    sp.add_argument("--output", required=True, help="output PNG path")
    common(sp)
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("run", help="generate assets from seed and render")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--output", required=True, help="output PNG path")
    sp.add_argument("--assets-dir", default=None, help="asset directory (optional)")
    common(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("verify", help="reproducibility check (2x MT + 1x ST hash compare)")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--workdir", default="verify_out")
    common(sp)
    sp.set_defaults(func=cmd_verify)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SoftRenderError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
