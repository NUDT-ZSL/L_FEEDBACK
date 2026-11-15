#!/usr/bin/env python3
"""Portable script entry point for Windows and POSIX."""

from podcast_post.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

