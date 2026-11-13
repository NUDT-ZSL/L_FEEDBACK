"""Windows-friendly wrapper for ``python -m pe_audit.generator``."""

from pe_audit.generator import main


if __name__ == "__main__":
    raise SystemExit(main())
