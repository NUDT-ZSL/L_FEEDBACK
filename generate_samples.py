#!/usr/bin/env python3
"""Convenience entry point for deterministic sample generation."""

import argparse

from audio_event_mixer.generator import generate_demo_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="demo_data")
    args = parser.parse_args()
    manifest = generate_demo_dataset(args.out_dir)
    print(manifest["events_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
