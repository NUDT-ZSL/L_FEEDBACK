from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .config import ConfigError, load_config
from .pipeline import run_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m audio_variants",
        description=(
            "Generate deterministic, loudness-normalized WAV variants for game audio."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="path to the JSON configuration file",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="directory containing source .wav files (not recursive)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        return run_batch(config, Path(args.input).expanduser())
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: filesystem operation failed: {exc}", file=sys.stderr)
        return 1
