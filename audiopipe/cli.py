"""Command line entry points (no third-party dependencies)."""

import argparse
import json
import sys

from .errors import AudioPipeError, emit_error, EXIT_INTERRUPTED, EXIT_USAGE
from .generate import generate, DEFAULT_LANGUAGES
from .pipeline import run_pipeline, request_shutdown
from .resample import resample, expected_output_len
from . import wavio


def _print(obj):
    sys.stdout.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")


def cmd_generate(args):
    languages = args.languages if args.languages else list(DEFAULT_LANGUAGES)
    summary = generate(
        args.material_dir,
        seed=args.seed,
        languages=languages,
        clip_count=args.clips,
    )
    _print(summary)
    return 0


def cmd_mix(args):
    try:
        results, exit_code = run_pipeline(
            args.material_dir,
            args.out_dir,
            target_rate=args.target_rate,
            headroom_db=args.headroom_db,
            max_workers=args.workers,
            max_open_files=args.max_open_files,
            languages=args.languages,
        )
    except AudioPipeError as exc:
        emit_error(exc)
        return exc.exit_code
    except KeyboardInterrupt:
        request_shutdown()
        emit_error(AudioPipeError("interrupted by user", code="E_INTERRUPTED",
                                  exit_code=EXIT_INTERRUPTED))
        return EXIT_INTERRUPTED

    summary = {
        "completed": len(results.get("completed", [])),
        "fatal": len(results.get("fatal", [])),
        "interrupted": results.get("interrupted", False),
        "manifest": results.get("manifest"),
    }
    _print(summary)
    for item in results.get("fatal", []):
        emit_error(AudioPipeError(
            "fatal clip failure %s" % item["token"],
            code="E_FATAL_CLIP",
            details=item.get("fatal_error"),
        ))
    return exit_code


def cmd_verify_wav(args):
    try:
        wav = wavio.read_wav(args.path)
    except AudioPipeError as exc:
        emit_error(exc)
        return exc.exit_code
    _print({"path": args.path, "sample_rate": wav.sample_rate,
            "samples": len(wav.samples)})
    return 0


def cmd_resample_demo(args):
    wav = wavio.read_wav(args.path)
    out, meta = resample(wav.samples, wav.sample_rate, args.target_rate)
    meta["expected"] = expected_output_len(len(wav.samples), wav.sample_rate, args.target_rate)
    _print(meta)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="audiopipe")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate deterministic material")
    g.add_argument("material_dir")
    g.add_argument("--seed", type=int, default=20261114)
    g.add_argument("--clips", type=int, default=4)
    g.add_argument("--languages", nargs="*")
    g.set_defaults(func=cmd_generate)

    m = sub.add_parser("mix", help="mix and package per language")
    m.add_argument("material_dir")
    m.add_argument("out_dir")
    m.add_argument("--target-rate", type=int, default=48000)
    m.add_argument("--headroom-db", type=float, default=-1.0)
    m.add_argument("--workers", type=int, default=4)
    m.add_argument("--max-open-files", type=int, default=8)
    m.add_argument("--languages", nargs="*")
    m.set_defaults(func=cmd_mix)

    v = sub.add_parser("verify-wav", help="strictly validate one wav")
    v.add_argument("path")
    v.set_defaults(func=cmd_verify_wav)

    r = sub.add_parser("resample", help="report deterministic resample length")
    r.add_argument("path")
    r.add_argument("--target-rate", type=int, default=48000)
    r.set_defaults(func=cmd_resample_demo)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except AudioPipeError as exc:
        emit_error(exc)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
