#!/usr/bin/env python3
"""Thin wrapper for the bundled deterministic sample generator."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cinema_master.samplegen import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

