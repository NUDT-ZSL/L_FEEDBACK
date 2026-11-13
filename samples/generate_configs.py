#!/usr/bin/env python3
"""Write the deterministic JSON taint-source configurations used by tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONFIGS = {
    "config.minimal.json": {
        "sources": [{"function": "source_a", "param_index": 1, "type": "string"}],
        "dangerous_apis": [],
        "sanitizers": [],
    },
    "config.sample.json": {
        "sources": [{"function": "source_a", "param_index": 1, "type": "string"}],
        "dangerous_apis": [],
        "sanitizers": ["sanitize", "safe_copy"],
        "module_hints": ["sample"],
    },
    "config.sample.param0.json": {
        "sources": [{"function": "source_a", "param_index": 0, "type": "int"}],
        "dangerous_apis": [],
        "sanitizers": ["sanitize", "safe_copy"],
        "module_hints": ["sample"],
    },
    "config.summary.json": {
        "sources": [
            {"function": "source_a", "param_index": 0, "type": "string"},
            {"function": "source_b", "param_index": 0, "type": "string"},
        ],
        "dangerous_apis": [],
        "sanitizers": [],
        "module_hints": ["summary"],
    },
    "config.warnings.json": {
        "sources": [
            {"function": "source_a", "param_index": 0, "type": "string"},
            {"function": "missing_func", "param_index": 0, "type": "ptr"},
        ],
        "dangerous_apis": [],
        "sanitizers": [],
        "module_hints": ["warnings"],
    },
    "config.regress.json": {
        "sources": [
            {"function": "source_a", "param_index": 0, "type": "string"},
            {"function": "source_b", "param_index": 0, "type": "string"},
        ],
        "dangerous_apis": [],
        "sanitizers": [],
        "module_hints": ["regress"],
    },
    "config.large.json": {
        "sources": [{"function": "large_function_00000", "param_index": 0, "type": "int"}],
        "dangerous_apis": [],
        "sanitizers": [],
        "module_hints": ["large"],
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, value in CONFIGS.items():
        (outdir / name).write_text(
            json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
