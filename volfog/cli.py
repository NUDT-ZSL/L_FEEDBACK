"""Command line interface for the volumetric fog renderer."""

from __future__ import annotations

import argparse
import json
import sys

from .config import RenderConfig
from .errors import RenderError
from .pipeline import render_to_files


def build_parser() -> argparse.ArgumentParser:
    parser = _JsonErrorParser(
        prog="volfog",
        description="Offline volumetric fog ray-marching renderer (CPU).",
    )
    parser.add_argument("--mode", choices=["single", "multiple"], default="single")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--voxel-res", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--density-file", default=None,
                        help="Optional .npy 3D array used as external density field")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--output-prefix", default="fog")
    parser.add_argument("--parallel", action="store_true",
                        help="Use the Numba prange parallel rendering path")
    parser.add_argument("--threads", type=int, default=0,
                        help="prange thread count (0 leaves Numba default)")

    parser.add_argument("--camera-pos", type=float, nargs=3, default=[0.0, 0.0, -2.2])
    parser.add_argument("--camera-target", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--focal-length", type=float, default=1.0)

    parser.add_argument("--light-pos", type=float, nargs=3, default=[0.7, 0.8, -0.6])
    parser.add_argument("--light-color", type=float, nargs=3, default=[1.0, 0.95, 0.85])
    parser.add_argument("--light-intensity", type=float, default=6.0)

    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--max-samples", type=int, default=220)
    parser.add_argument("--terminate-threshold", type=float, default=0.005)
    parser.add_argument("--depth-threshold", type=float, default=0.01)
    parser.add_argument("--extinction", type=float, default=2.5)
    parser.add_argument("--albedo", type=float, default=0.9)
    parser.add_argument("--phase-g", type=float, default=0.0)
    parser.add_argument("--light-samples", type=int, default=8)
    parser.add_argument("--multiple-strength", type=float, default=1.0)
    return parser


class _JsonErrorParser(argparse.ArgumentParser):
    """Emit the unified single-line JSON error instead of argparse usage."""

    def error(self, message):
        param = "arguments"
        for token in message.split():
            if token.startswith("--"):
                param = token.lstrip("-").rstrip(":").replace("-", "_")
                break
        payload = {
            "error_code": "E_INVALID_INPUT",
            "param": param,
            "message": message,
        }
        sys.stderr.write(json.dumps(payload, separators=(",", ":")) + "\n")
        raise SystemExit(1)


def config_from_args(argv=None) -> RenderConfig:
    args = build_parser().parse_args(argv)
    return RenderConfig(
        mode=args.mode,
        width=args.width,
        height=args.height,
        voxel_res=args.voxel_res,
        seed=args.seed,
        density_file=args.density_file,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        parallel=args.parallel,
        threads=args.threads,
        camera_pos=tuple(args.camera_pos),
        camera_target=tuple(args.camera_target),
        focal_length=args.focal_length,
        light_pos=tuple(args.light_pos),
        light_color=tuple(args.light_color),
        light_intensity=args.light_intensity,
        step_size=args.step_size,
        max_samples=args.max_samples,
        terminate_threshold=args.terminate_threshold,
        depth_threshold=args.depth_threshold,
        extinction=args.extinction,
        albedo=args.albedo,
        phase_g=args.phase_g,
        light_samples=args.light_samples,
        multiple_strength=args.multiple_strength,
    )


def main(argv=None) -> int:
    try:
        config = config_from_args(argv)
        meta = render_to_files(config)
    except RenderError as exc:
        sys.stderr.write(json.dumps(exc.to_payload(), separators=(",", ":")) + "\n")
        return 1
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
