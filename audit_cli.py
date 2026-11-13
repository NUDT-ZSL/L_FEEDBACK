"""Windows-friendly wrapper for ``python -m pe_audit.cli``."""

from pe_audit.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
