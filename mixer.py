#!/usr/bin/env python3
"""Compatibility wrapper for running the mixer without installation."""

import sys

from game_audio_mixer.cli import main


if __name__ == "__main__":
    sys.exit(main())
