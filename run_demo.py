#!/usr/bin/env python3
"""Convenience entry point: build deterministic samples and run all languages."""

from __future__ import annotations

import argparse
from pathlib import Path

from lipsync_tool.cli import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build samples and run the lip-sync demo")
    parser.add_argument("--workdir", default="demo_workdir", help="sample and output root directory")
    parser.add_argument("--config", default=None, help="optional JSON config file")
    args = parser.parse_args()

    root = Path(args.workdir)
    main(["make-samples", "--output-dir", str(root / "samples")])
    process_args = [
        "process",
        "--metadata", str(root / "samples" / "animation_metadata.json"),
        "--audio-dir", str(root / "samples" / "audio"),
        "--video", str(root / "samples" / "video" / "cutscene_silent.mp4"),
        "--output-dir", str(root / "output"),
        "--languages", "en,ja,zh",
    ]
    if args.config:
        process_args += ["--config", args.config]
    raise SystemExit(main(process_args))
