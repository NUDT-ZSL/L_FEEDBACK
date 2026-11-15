"""Command-line entry point: ``python -m terrain_gen <config.yaml>``."""

from __future__ import annotations

import json
import sys

from .config import ConfigError, parse_config
from .pipeline import run_pipeline, run_sweep


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(
            json.dumps(
                {"error_code": "ERR_CFG", "message": "usage: terrain_gen <config.yaml>"}
            ),
            file=sys.stderr,
        )
        return 2

    try:
        config = parse_config(argv[0])
    except ConfigError as exc:
        print(
            json.dumps({"error_code": "ERR_CFG", "message": exc.message}),
            file=sys.stderr,
        )
        return 2

    try:
        if config.sweep_iterations:
            summary = run_sweep(config)
            for entry in summary["results"]:
                status = entry["status"]
                detail = entry.get("error_type", entry.get("outputs", ""))
                print(f"iterations={entry['iterations']} {status} {detail}")
            print(f"summary: {summary['summary_path']}")
            return int(summary["exit_code"])

        result = run_pipeline(config)
        for kind, path in result["outputs"].items():
            print(f"{kind}: {path}")
        return 0
    except Exception as exc:  # noqa: BLE001 - single-run failure must be reported
        print(
            json.dumps(
                {"error_code": "ERR_RUN", "error_type": type(exc).__name__, "message": str(exc)}
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
