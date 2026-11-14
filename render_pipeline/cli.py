from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .batch import render_batch_with_error_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="render_cards.py",
        description="Render deterministic offline material preview cards.",
    )
    parser.add_argument("--scene", required=True, type=Path, help="scene descriptor JSON")
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="output directory; an existing directory is replaced as a whole",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = render_batch_with_error_output(args.scene, args.out)
    except (OSError, ValueError) as exc:
        print(json.dumps({"code": "IO_OUTPUT_FAILED", "message": str(exc)}), file=sys.stderr)
        return 1
    summary = {
        "output": str(result.output_directory),
        "status": result.manifest.get("status"),
        "input_sha256": result.manifest.get("input_sha256"),
        "presets": [
            {
                "id": item["id"],
                "status": item["status"],
                "diagnostic_codes": item.get("diagnostic_codes", []),
            }
            for item in result.manifest.get("preset_results", [])
        ],
        "diagnostics": len(result.diagnostics),
    }
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return result.exit_code
