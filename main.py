"""Windows-friendly command-line entry point."""

from __future__ import annotations

import sys


def _entry() -> int:
    try:
        from cinemix.cli import main
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required dependency"
        print(
            f"error: missing Python dependency '{missing}'. "
            "Run: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1
    return main()


if __name__ == "__main__":
    raise SystemExit(_entry())
