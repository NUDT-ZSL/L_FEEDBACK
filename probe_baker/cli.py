"""Command line interface for the probe baking pipeline."""
import argparse
import json
import sys

from . import heightfield as hfmod
from . import scheduler
from .errors import BakeError, emit_error

DEFAULT_SPACING = 4
DEFAULT_BLOCK_SIZE = 8
DEFAULT_Z_SCALE = 16.0
DEFAULT_LIGHT_DIR = (0.4, 0.3, 0.8660254)
DEFAULT_MAX_CACHED = 64
DEFAULT_TOLERANCE = 1e-5


def _parse_light_dir(text):
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("light dir must be 'x,y,z'")
    return tuple(parts)


def _parse_rect(text):
    parts = [int(v) for v in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("rect must be 'x,y,w,h'")
    return tuple(parts)


def _add_bake_args(p):
    p.add_argument("--heightfield", required=True)
    p.add_argument("--state-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--spacing", type=int, default=DEFAULT_SPACING)
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    p.add_argument("--z-scale", type=float, default=DEFAULT_Z_SCALE)
    p.add_argument("--light-dir", type=_parse_light_dir,
                   default=DEFAULT_LIGHT_DIR)
    p.add_argument("--threads", type=int, default=4,
                   help="worker threads; env PROBE_BAKE_THREADS overrides")
    p.add_argument("--max-cached-blocks", type=int, default=DEFAULT_MAX_CACHED)


def _params_of(args):
    return {
        "spacing": args.spacing,
        "block_size": args.block_size,
        "z_scale": args.z_scale,
        "light_dir": list(args.light_dir),
    }


def cmd_gen_sample(args):
    heights = hfmod.generate_sample(args.width, args.height, args.seed)
    if args.edit_rect is not None:
        heights = hfmod.apply_edit(heights, args.edit_rect, args.edit_delta)
    version = hfmod.write_heightfield(args.out, heights)
    print(json.dumps({"written": args.out, "version_hash": version,
                      "width": args.width, "height": args.height}))
    return 0


def cmd_bake(args):
    params = _params_of(args)
    threads = scheduler.resolve_threads(args.threads)
    if args.dry_run:
        report = scheduler.dry_run(args.heightfield, params, threads,
                                   args.max_cached_blocks)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    record, interrupted = scheduler.bake(
        args.heightfield, args.state_dir, args.out, args.log, params,
        threads, args.max_cached_blocks,
        verify_determinism=args.verify_determinism,
    )
    print(json.dumps({"status": record["status"], "log": args.log}))
    return 0  # graceful interrupt is still exit code 0


def cmd_reconstruct(args):
    params = _params_of(args)
    threads = scheduler.resolve_threads(args.threads)
    record, interrupted = scheduler.reconstruct(
        args.heightfield, args.state_dir, args.out, args.log, params,
        threads, args.max_cached_blocks, args.edit_rect, args.tolerance,
    )
    print(json.dumps({"status": record["status"], "log": args.log}))
    return 0


def cmd_compare_logs(args):
    report = scheduler.compare_logs(args.log_a, args.log_b)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="probe_baker")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("gen-sample", help="write a deterministic heightfield")
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--edit-rect", type=_parse_rect, default=None,
                   help="x,y,w,h pixel rect to edit before writing")
    p.add_argument("--edit-delta", type=float, default=0.35)
    p.set_defaults(func=cmd_gen_sample)

    p = sub.add_parser("bake", help="full / resumable bake")
    _add_bake_args(p)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verify-determinism", action="store_true")
    p.set_defaults(func=cmd_bake)

    p = sub.add_parser("reconstruct", help="local rebuild after an edit")
    _add_bake_args(p)
    p.add_argument("--edit-rect", type=_parse_rect, required=True)
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    p.set_defaults(func=cmd_reconstruct)

    p = sub.add_parser("compare-logs", help="diff two bake logs")
    p.add_argument("log_a")
    p.add_argument("log_b")
    p.set_defaults(func=cmd_compare_logs)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BakeError as err:
        return emit_error(err)


if __name__ == "__main__":
    sys.exit(main())
