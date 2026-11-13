"""Generate the deterministic local MXF acceptance set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mxf_r128.probe import resolve_tool
from mxf_r128.samples import generate_samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--long-duration", type=float, default=400.0)
    parser.add_argument("--include-long", action="store_true")
    parser.add_argument("--include-subtitle", action="store_true")
    args = parser.parse_args()

    ffmpeg = resolve_tool("FFMPEG_BIN", "ffmpeg")
    ffprobe = resolve_tool("FFPROBE_BIN", "ffprobe")
    if ffmpeg is None or ffprobe is None:
        print("E_FFMPEG_NOT_FOUND", file=sys.stderr)
        return 2
    try:
        sample_set = generate_samples(
            args.output_dir,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            duration=args.duration,
            include_long=args.include_long,
            long_duration=args.long_duration,
            include_subtitle=args.include_subtitle,
        )
    except (OSError, RuntimeError):
        print("E_SAMPLE_GENERATION", file=sys.stderr)
        return 3
    for path in sample_set.files:
        print(path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
