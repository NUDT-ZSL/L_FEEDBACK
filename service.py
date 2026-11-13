"""Compatibility entry point: python service.py --port 5000."""

from depconfusion.service import main


if __name__ == "__main__":
    raise SystemExit(main())
