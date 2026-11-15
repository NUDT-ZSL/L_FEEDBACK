"""Command line entry point.

Usage:
    python -m audiopipe MANIFEST --output-dir DIR [--jobs N] [--crossfade-ms MS]

Exit codes: 0 = all segments succeeded, 1 = partial failure (valid output),
2 = fatal error (no output).
"""

from __future__ import annotations

import argparse
import sys

from .errors import PipelineError
from .manifest import load_manifest
from .pipeline import run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="audiopipe",
        description="Broadcast audio transcode / loudness-normalise / concat")
    parser.add_argument("manifest", help="path to the JSON manifest")
    parser.add_argument("--output-dir", required=True,
                        help="directory for output.wav and report.json")
    parser.add_argument("--jobs", type=int, default=None,
                        help="max concurrent segment workers "
                             "(default: manifest 'jobs' or 4)")
    parser.add_argument("--crossfade-ms", type=float, default=None,
                        help="junction crossfade in ms (default: manifest or 20)")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
    except PipelineError as exc:
        print(f"致命错误 [{exc.code}]: {exc.message}", file=sys.stderr)
        return 2
    jobs = args.jobs if args.jobs is not None else (manifest.jobs or 4)
    if jobs < 1:
        print("错误: --jobs 必须 >= 1", file=sys.stderr)
        return 2
    if not manifest.segments:
        print("致命错误: 清单中没有片段", file=sys.stderr)
        return 2
    try:
        return run(manifest, args.output_dir, jobs=jobs,
                   crossfade_ms=args.crossfade_ms)
    except Exception as exc:
        print(f"致命错误: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
