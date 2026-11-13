"""Shared test helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from pe_audit.binary import ImportSymbol
from pe_audit.callgraph import CallAnalysis, ParsedFunction, parse_function
from pe_audit.generator import generate_samples
from pe_audit.manifests import FunctionRegion, Manifest


ROOT = Path(__file__).resolve().parents[1]


def generate(directory: str | Path, seed: int = 0) -> Path:
    target = Path(directory)
    generate_samples(seed, target)
    return target


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "audit_cli.py", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def run_generator(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "generate_samples.py", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def make_analysis(
    bodies: dict[str, bytes],
    *,
    local_names: list[str] | None = None,
    imports: list[ImportSymbol] | None = None,
) -> CallAnalysis:
    names = list(local_names) if local_names is not None else sorted(bodies)
    import_symbols = imports or [
        ImportSymbol("msvcrt.dll", "strcpy"),
        ImportSymbol("msvcrt.dll", "strncpy"),
    ]
    regions = {
        name: FunctionRegion(name, 0x1000 + index * 0x100, len(body))
        for index, (name, body) in enumerate(sorted(bodies.items()))
    }
    manifest = Manifest(
        version="test",
        regions=regions,
        entry_points=[],
        raw={},
    )
    parsed = {
        name: parse_function(
            body=body,
            function=name,
            local_names=names,
            imports=import_symbols,
            known_functions=set(bodies),
        )
        for name, body in sorted(bodies.items())
    }
    return CallAnalysis(manifest, import_symbols, parsed)
