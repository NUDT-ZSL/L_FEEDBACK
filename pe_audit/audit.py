"""End-to-end identity-aligned patch audit and deterministic JSON report."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from .binary import PeImage, parse_image
from .callgraph import CallAnalysis, analyze_calls
from .diffing import diff_aligned_functions
from .errors import AuditError, EXIT_IO
from .manifests import (
    Alignment,
    Manifest,
    align_manifests,
    load_json,
    parse_manifest,
)
from .reachability import path_for_function, reachability
from .rules import (
    DefectDocument,
    RuleSpec,
    evaluate_function,
    load_defects,
)


CONCLUSION_SEALED = "已封堵"
CONCLUSION_REACHABLE = "仍可达"
CONCLUSION_INSUFFICIENT = "证据不足"


def _read_bytes(path: str | Path, description: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise AuditError(
            "INPUT_READ_ERROR", f"{description}: {exc}", EXIT_IO
        ) from exc


def _resolve_entries(
    *,
    entries_path: str | Path | None,
    defects: DefectDocument,
    before: Manifest,
    after: Manifest,
    before_image: PeImage,
    after_image: PeImage,
) -> list[str]:
    if entries_path is not None:
        document = load_json(entries_path, "entries")
        raw = document.get("entries", document.get("entry_points"))
        if not isinstance(raw, list) or not all(
            isinstance(entry, str) and entry for entry in raw
        ):
            raise AuditError(
                "ENTRIES_INVALID",
                "entries must be a list of non-empty strings",
                3,
            )
        return sorted(set(raw))
    if defects.entry_points:
        return list(defects.entry_points)
    combined = sorted(set(before.entry_points) | set(after.entry_points))
    if combined:
        return combined
    for manifest, image in ((before, before_image), (after, after_image)):
        for region in manifest.regions.values():
            if region.rva == image.entry_rva:
                return [region.name]
    return []


def _reachable_enough(
    graph: dict[str, Any], function: str
) -> tuple[bool, list[str], list[dict[str, Any]]]:
    item = path_for_function(graph, function)
    if item is None:
        return False, [], []
    return True, list(item["path"]), [dict(edge) for edge in item["edges"]]


def _conclusion(
    *,
    rule: RuleSpec,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    before_graph: dict[str, Any],
    after_graph: dict[str, Any],
) -> tuple[str, str]:
    target = rule.target
    before_reachable, before_path, before_edges = _reachable_enough(
        before_graph, target
    )
    after_reachable, after_path, after_edges = _reachable_enough(
        after_graph, target
    )

    if after_state["deleted"]:
        return CONCLUSION_SEALED, "target_deleted"

    if not after_state["evidence_complete"]:
        return CONCLUSION_INSUFFICIENT, "evidence_incomplete"

    if after_state["verdict"] == "CLEAN":
        if before_state["verdict"] == "VULNERABLE":
            return CONCLUSION_SEALED, "patched_clean"
        return CONCLUSION_SEALED, "already_clean"

    if after_reachable:
        return CONCLUSION_REACHABLE, "vulnerable_and_reachable"

    if after_graph["graph_incomplete"]:
        return CONCLUSION_INSUFFICIENT, "graph_incomplete"
    return CONCLUSION_INSUFFICIENT, "vulnerable_unreachable"


def _target_report(
    *,
    rule: RuleSpec,
    before_analysis: CallAnalysis,
    after_analysis: CallAnalysis,
    before_graph: dict[str, Any],
    after_graph: dict[str, Any],
) -> dict[str, Any]:
    before_state = evaluate_function(
        rule, before_analysis.functions.get(rule.target)
    )
    after_state = evaluate_function(
        rule, after_analysis.functions.get(rule.target)
    )
    conclusion, basis = _conclusion(
        rule=rule,
        before_state=before_state,
        after_state=after_state,
        before_graph=before_graph,
        after_graph=after_graph,
    )
    before_reachable, before_path, before_edges = _reachable_enough(
        before_graph, rule.target
    )
    after_reachable, after_path, after_edges = _reachable_enough(
        after_graph, rule.target
    )
    if after_reachable:
        path, path_edges = after_path, after_edges
    elif before_reachable:
        path, path_edges = before_path, before_edges
    else:
        path, path_edges = [], []

    after_evidence = after_state["evidence"]
    before_evidence = before_state["evidence"]
    call_evidence = (
        [dict(item) for item in after_evidence]
        if after_evidence
        else [dict(item) for item in before_evidence]
    )
    return {
        "id": rule.id,
        "function": rule.target,
        "dangerous_imports": sorted(
            pattern.label() for pattern in rule.dangerous_imports
        ),
        "rule": rule.to_dict()["rule"],
        "before": before_state,
        "after": after_state,
        "before_reachable": before_reachable,
        "after_reachable": after_reachable,
        "call_evidence": call_evidence,
        "path": path,
        "path_edges": path_edges,
        "basis": basis,
        "conclusion": conclusion,
    }


def build_audit_report(
    *,
    before_path: str | Path,
    after_path: str | Path,
    before_manifest_path: str | Path,
    after_manifest_path: str | Path,
    defects_path: str | Path,
    entries_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate all inputs and return a path-free, deterministic report."""

    before_data = _read_bytes(before_path, "before binary")
    after_data = _read_bytes(after_path, "after binary")
    before_image = parse_image(before_data)
    after_image = parse_image(after_data)

    before_document = load_json(before_manifest_path, "before manifest")
    after_document = load_json(after_manifest_path, "after manifest")
    before_manifest = parse_manifest(
        before_document, image=before_image, version="before"
    )
    after_manifest = parse_manifest(
        after_document, image=after_image, version="after"
    )
    alignment = align_manifests(before_manifest, after_manifest)

    known = set(before_manifest.regions) | set(after_manifest.regions)
    defects = load_defects(defects_path, known)
    entries = _resolve_entries(
        entries_path=entries_path,
        defects=defects,
        before=before_manifest,
        after=after_manifest,
        before_image=before_image,
        after_image=after_image,
    )

    before_analysis = analyze_calls(before_image, before_manifest)
    after_analysis = analyze_calls(after_image, after_manifest)
    before_graph = reachability(before_analysis, entries)
    after_graph = reachability(after_analysis, entries)
    diff = diff_aligned_functions(
        before=before_analysis,
        after=after_analysis,
        alignment=alignment,
    )

    targets = [
        _target_report(
            rule=rule,
            before_analysis=before_analysis,
            after_analysis=after_analysis,
            before_graph=before_graph,
            after_graph=after_graph,
        )
        for rule in defects.rules
    ]
    targets.sort(key=lambda item: item["id"])
    by_conclusion = {
        CONCLUSION_SEALED: sum(
            item["conclusion"] == CONCLUSION_SEALED for item in targets
        ),
        CONCLUSION_REACHABLE: sum(
            item["conclusion"] == CONCLUSION_REACHABLE for item in targets
        ),
        CONCLUSION_INSUFFICIENT: sum(
            item["conclusion"] == CONCLUSION_INSUFFICIENT for item in targets
        ),
    }

    return {
        "schema_version": 1,
        "input_sha256": {
            "before": hashlib.sha256(before_data).hexdigest(),
            "after": hashlib.sha256(after_data).hexdigest(),
        },
        "functions": {
            "before": before_manifest.names,
            "after": after_manifest.names,
            "alignment": alignment.to_dict(),
        },
        "imports": {
            "before": [
                symbol.to_dict()
                for symbol in sorted(
                    before_image.imports,
                    key=lambda item: (item.module, item.name),
                )
            ],
            "after": [
                symbol.to_dict()
                for symbol in sorted(
                    after_image.imports,
                    key=lambda item: (item.module, item.name),
                )
            ],
        },
        "calls": {
            "before": before_analysis.to_dict(),
            "after": after_analysis.to_dict(),
        },
        "diff": diff,
        "reachability": {"before": before_graph, "after": after_graph},
        "targets": targets,
        "summary": {
            "target_count": len(targets),
            "sealed": by_conclusion[CONCLUSION_SEALED],
            "still_reachable": by_conclusion[CONCLUSION_REACHABLE],
            "insufficient_evidence": by_conclusion[CONCLUSION_INSUFFICIENT],
            "by_conclusion": by_conclusion,
        },
    }


def report_json_bytes(report: Mapping[str, Any]) -> bytes:
    text = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    return text.encode("utf-8")


def write_audit_report(path: str | Path, report: Mapping[str, Any]) -> None:
    """Atomically write a report so an input error never leaves partial output."""

    destination = Path(path)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    data = report_json_bytes(report)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(parent)
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def summary_json_bytes(report: Mapping[str, Any]) -> bytes:
    text = json.dumps(
        report["summary"], ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    return text.encode("utf-8")
