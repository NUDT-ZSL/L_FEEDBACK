"""Command line interface for dubmux."""

from __future__ import annotations

import argparse
import sys

from .audio import DetectConfig
from .errors import EXIT_FATAL
from .pipeline import JobConfig, run


def _parse_named(values: list[str], kind: str) -> dict[str, str]:
    out = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"invalid {kind} mapping {item!r}; expected LANG=PATH")
        lang, path = item.split("=", 1)
        out[lang.strip()] = path.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dubmux",
        description="Deterministic multilingual dub align / subtitle retime / mux.",
    )
    p.add_argument("--video", required=True, help="reference video file")
    p.add_argument("--lang", action="append", default=[],
                   help="LANG=PATH dub WAV (repeatable)")
    p.add_argument("--subs", action="append", default=[],
                   help="LANG=PATH subtitle timeline JSON (repeatable)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--report", required=True, help="alignment report JSON path")
    p.add_argument("--jobs", type=int, default=4, help="max parallel languages")
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--tolerance", type=float, default=0.05,
                   help="max boundary deviation from target interval (s)")
    p.add_argument("--max-sub-duration", type=float, default=7.0)
    p.add_argument("--min-sub-duration", type=float, default=1.0)
    p.add_argument("--silence-threshold", type=float, default=0.01,
                   help="fixed RMS threshold for dialogue detection")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dubs = _parse_named(args.lang, "--lang")
    subs = _parse_named(args.subs, "--subs")
    if not dubs:
        print("[fatal] at least one --lang LANG=PATH mapping is required")
        return EXIT_FATAL
    missing = sorted(set(dubs) - set(subs))
    if missing:
        print(f"[fatal] missing --subs for languages: {', '.join(missing)}")
        return EXIT_FATAL
    languages = {lang: (dubs[lang], subs[lang]) for lang in sorted(dubs)}
    cfg = JobConfig(
        sample_rate=args.sample_rate,
        tolerance=args.tolerance,
        max_sub_duration=args.max_sub_duration,
        min_sub_duration=args.min_sub_duration,
        jobs=max(1, args.jobs),
        detect=DetectConfig(threshold=args.silence_threshold),
    )
    return run(args.video, languages, args.out, args.report, cfg)


if __name__ == "__main__":
    sys.exit(main())
