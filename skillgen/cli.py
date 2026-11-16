from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import ConfigError, load_config
from .pipeline import (
    Generator,
    OutputBusyError,
    OutputLock,
    write_json_atomic,
)


def _error_report(
    code: str,
    message: str,
    issues: list[dict[str, Any]] | None = None,
    exit_code: int = 2,
) -> dict[str, Any]:
    return {
        "version": __version__,
        "status": "error",
        "error_code": code,
        "message": message,
        "issues": issues or [],
        "exit_code": exit_code,
    }


def _fallback_output_dir(config_path: Path) -> Path:
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if isinstance(raw, dict):
            value = raw.get(
                "output_dir",
                raw.get("outputDir", raw.get("output_directory")),
            )
            if isinstance(value, str) and value.strip():
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = config_path.parent / path
                return path.resolve()
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    return config_path.parent


def _try_write_error_report(output_dir: Path, report: dict[str, Any]) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output_dir / "error_report.json", report)
    except OSError:
        pass


def _print_json(value: dict[str, Any]) -> None:
    sys.stdout.write(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    sys.stdout.write("\n")
    sys.stdout.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skillgen",
        description=(
            "Deterministically generate transient-aligned, loudness-calibrated "
            "16-bit PCM skill-audio variants."
        ),
    )
    parser.add_argument("config", nargs="?", help="path to skill configuration JSON")
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore successful JSONL state entries and regenerate every variant",
    )
    parser.add_argument(
        "--fail-on-variant-error",
        action="store_true",
        help="exit with status 1 when processing finished with variant errors",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config is None:
        parser.error("config is required")
    config_path = Path(args.config).expanduser().resolve()
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        report = _error_report(
            exc.overall_code,
            str(exc),
            [issue.as_dict() for issue in exc.issues],
            exit_code=2,
        )
        _try_write_error_report(_fallback_output_dir(config_path), report)
        _print_json(report)
        return 2
    except Exception as exc:
        report = _error_report(
            "E_INTERNAL",
            f"unexpected configuration failure: {type(exc).__name__}: {exc}",
            exit_code=3,
        )
        _try_write_error_report(_fallback_output_dir(config_path), report)
        _print_json(report)
        return 3

    try:
        with OutputLock(config.output_dir):
            manifest = Generator(config).run(force=args.force)
    except OutputBusyError as exc:
        report = _error_report("E_BUSY", str(exc), exit_code=4)
        _print_json(report)
        return 4
    except Exception as exc:
        report = _error_report(
            "E_INTERNAL",
            f"unexpected processing failure: {type(exc).__name__}: {exc}",
            exit_code=3,
        )
        _try_write_error_report(config.output_dir, report)
        _print_json(report)
        return 3

    _print_json(manifest)
    if args.fail_on_variant_error and manifest["summary"]["error_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
