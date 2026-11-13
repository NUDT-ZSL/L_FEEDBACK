#!/usr/bin/env python3
"""Entry point: python3 main.py <command> ..."""
import sys

from probe_baker.cli import main

if __name__ == "__main__":
    sys.exit(main())
