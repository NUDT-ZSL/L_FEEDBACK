"""Compatibility launcher for environments expecting main.py."""

from material_baker.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

