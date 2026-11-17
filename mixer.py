#!/usr/bin/env python3
"""Windows-friendly command-line entry point."""

from __future__ import annotations

import sys

from bgm_mixer.cli import main


if __name__ == "__main__":
    sys.exit(main())
