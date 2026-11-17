#!/usr/bin/env python3
"""CLI entry point for the offline rhythm-game audio engine.

Usage:
    python3 main.py --config assets/config.json \
        --output out/mix.wav --report out/report.json

Business errors are printed to stderr with a stable code
(ERR_MISSING_FILE / ERR_BAD_SAMPLE_RATE / ERR_INVALID_BPM) and exit
with a non-zero status without leaving partial output files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from audio_engine.engine import run
from audio_engine.errors import BusinessError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline rhythm-game audio engine")
    parser.add_argument("--config", required=True, help="path to engine config JSON")
    parser.add_argument("--output", required=True, help="output stereo WAV path")
    parser.add_argument("--report", required=True, help="output quality report JSON path")
    args = parser.parse_args(argv)

    try:
        if not os.path.isfile(args.config):
            raise BusinessError("ERR_MISSING_FILE", f"config not found: {args.config}")
        with open(args.config, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        config["_config_dir"] = os.path.dirname(os.path.abspath(args.config))
        run(config, args.output, args.report)
    except BusinessError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
