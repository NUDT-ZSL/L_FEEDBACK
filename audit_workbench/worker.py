"""One isolated source-root/configuration combination worker."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .analysis import analyze
from .config import AuditConfig, config_from_dict
from .cparse import ParsedUnit, parse_unit
from .errors import WorkbenchError
from .preprocessor import Preprocessor


SOURCE_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".c++"}


def _dedupe_unit(unit: ParsedUnit) -> None:
    call_map: dict[str, Any] = {}
    for call in sorted(
        unit.calls,
        key=lambda item: (
            item.callsite_id,
            item.fingerprint,
            item.relative_file,
            item.line,
            item.column,
        ),
    ):
        call_map.setdefault(call.callsite_id, call)
    selected = set(id(call) for call in call_map.values())
    unit.calls = [
        call for call in unit.calls if id(call) in selected
    ]
    for function in unit.functions.values():
        function.calls = [
            call
            for call in function.calls
            if id(call) in selected or call.callsite_id == ""
        ]
        # Non-dangerous calls also have IDs and are de-duplicated in the same
        # deterministic manner. Calls not present in unit.calls (for example a
        # header call retained from another TU object) are filtered by ID.
        seen_calls: set[str] = set()
        deduped = []
        for call in function.calls:
            if call.callsite_id and call.callsite_id in seen_calls:
                continue
            seen_calls.add(call.callsite_id)
            deduped.append(call)
        function.calls = deduped
    failure_map: dict[tuple[str, int, str, str], Any] = {}
    for failure in unit.failures:
        failure_map.setdefault(
            (failure.file, failure.line, failure.reason, failure.snippet),
            failure,
        )
    unit.failures = sorted(
        failure_map.values(),
        key=lambda item: (item.file, item.line, item.reason, item.snippet),
    )
    unit.included_files = sorted(set(unit.included_files))


def run_combination(
    root: Path,
    config: AuditConfig,
    *,
    max_hops: int = 8,
) -> dict[str, Any]:
    root = Path(root).resolve()
    sources = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS
    )
    merged = ParsedUnit()
    for source in sources:
        preprocessing = Preprocessor(root, config)
        preprocessed = preprocessing.process(source)
        parsed = parse_unit(preprocessed, str(source))
        merged.merge(parsed)
    _dedupe_unit(merged)
    findings, summary = analyze(merged, config, max_hops=max_hops)
    return {
        "status": "ok",
        "calls": findings,
        "summary": summary,
        "parse_failures": [failure.to_dict() for failure in merged.failures],
        "parse_failure_count": len(merged.failures),
        "included_files": merged.included_files,
        "macro_expansions": merged.macro_expansions,
        "translation_units": [str(source.relative_to(root).as_posix()) for source in sources],
        "errors": [],
    }


def execute_payload(payload: dict[str, Any]) -> dict[str, Any]:
    root = Path(payload["root"])
    config = config_from_dict(payload["config"], source="<worker-config>")
    try:
        return run_combination(
            root,
            config,
            max_hops=int(payload.get("max_hops", 8)),
        )
    except WorkbenchError as exc:
        error = exc.to_dict()
        error["context"].setdefault("root", payload.get("root_label") or root.name)
        error["context"].setdefault("config", config.name)
        return {
            "status": "error",
            "calls": [],
            "summary": {
                "calls_total": 0,
                "reachability": {"reachable": 0, "unreachable": 0, "unknown": 0},
                "taint": {"tainted": 0, "untainted": 0, "unknown": 0},
                "high_risk": 0,
                "entry_missing": [],
            },
            "parse_failures": [],
            "parse_failure_count": 0,
            "included_files": [],
            "macro_expansions": 0,
            "translation_units": [],
            "errors": [error],
        }
    except Exception as exc:  # pragma: no cover - defensive isolation boundary
        return {
            "status": "error",
            "calls": [],
            "summary": {
                "calls_total": 0,
                "reachability": {"reachable": 0, "unreachable": 0, "unknown": 0},
                "taint": {"tainted": 0, "untainted": 0, "unknown": 0},
                "high_risk": 0,
                "entry_missing": [],
            },
            "parse_failures": [],
            "parse_failure_count": 0,
            "included_files": [],
            "macro_expansions": 0,
            "translation_units": [],
            "errors": [
                {
                    "code": "E_WORKER",
                    "message": f"worker failed: {type(exc).__name__}: {exc}",
                    "context": {
                        "root": payload.get("root_label") or root.name,
                        "config": config.name,
                    },
                }
            ],
        }


def worker_main(payload_path: Path, result_path: Path) -> int:
    try:
        payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
        result = execute_payload(payload)
    except Exception as exc:  # pragma: no cover - process boundary
        result = {
            "status": "error",
            "calls": [],
            "summary": {
                "calls_total": 0,
                "reachability": {"reachable": 0, "unreachable": 0, "unknown": 0},
                "taint": {"tainted": 0, "untainted": 0, "unknown": 0},
                "high_risk": 0,
                "entry_missing": [],
            },
            "parse_failures": [],
            "parse_failure_count": 0,
            "included_files": [],
            "macro_expansions": 0,
            "translation_units": [],
            "errors": [
                {
                    "code": "E_WORKER",
                    "message": f"worker bootstrap failed: {type(exc).__name__}: {exc}",
                    "context": {"file": str(payload_path), "line": 1},
                }
            ],
        }
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_name(result_path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, result_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(worker_main(Path(sys.argv[1]), Path(sys.argv[2])))

