"""Audit orchestration, report assembly, baseline diffs, and gate decisions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import RULES_VERSION, __version__
from .config import AuditConfig, resolve_config
from .errors import AuditError, WorkbenchError
from .hashing import canonical_json, sha256_text, source_root_hash
from .scheduler import active_child_count, run_combinations
from .snapshot import (
    diff_payloads,
    ensure_state_dir,
    load_snapshot,
    snapshot_payload,
    write_snapshot,
)


REPORT_FORMAT = "audit-report-v1"


@dataclass(frozen=True)
class RootSpec:
    name: str
    path: Path
    root_hash: str


def _empty_summary() -> dict[str, Any]:
    return {
        "calls_total": 0,
        "reachability": {"reachable": 0, "unreachable": 0, "unknown": 0},
        "taint": {"tainted": 0, "untainted": 0, "unknown": 0},
        "high_risk": 0,
        "entry_missing": [],
    }


def _split_csv(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item and item not in result:
                result.append(item)
    return result


def discover_roots(root_values: Iterable[str]) -> tuple[list[RootSpec], list[Path]]:
    roots: list[RootSpec] = []
    config_dirs: list[Path] = []
    seen_labels: set[str] = set()
    for value in _split_csv(root_values):
        root_arg = Path(value).expanduser()
        if not root_arg.is_dir():
            raise WorkbenchError(
                "E_IO",
                f"source root does not exist or is not a directory: {value}",
                {"file": str(root_arg.resolve()), "line": 1},
            )
        manifest_path = root_arg / "roots.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise WorkbenchError(
                    "E_CONFIG",
                    f"cannot read generated roots manifest: {exc}",
                    {"file": str(manifest_path), "line": 1},
                ) from exc
            config_dirs.append(root_arg / "configs")
            entries = manifest.get("roots") if isinstance(manifest, dict) else None
            if not isinstance(entries, list) or not entries:
                raise WorkbenchError(
                    "E_CONFIG",
                    "roots.json does not contain a non-empty roots array",
                    {"file": str(manifest_path), "line": 1},
                )
            for entry in entries:
                name = str(entry.get("name") or Path(str(entry.get("path", ""))).name)
                path = (root_arg / str(entry.get("path", ""))).resolve()
                if not path.is_dir():
                    raise WorkbenchError(
                        "E_IO",
                        f"manifest source root is missing: {entry.get('path')}",
                        {"file": str(manifest_path), "line": 1},
                    )
                if name in seen_labels:
                    raise WorkbenchError(
                        "E_ARGS",
                        f"duplicate source root label: {name}",
                        {"file": str(manifest_path), "line": 1, "root": name},
                    )
                seen_labels.add(name)
                roots.append(
                    RootSpec(name, path, source_root_hash(path))
                )
        else:
            path = root_arg.resolve()
            name = path.name
            if name in seen_labels:
                raise WorkbenchError(
                    "E_ARGS",
                    f"duplicate source root label: {name}",
                    {"file": str(path), "line": 1, "root": name},
                )
            seen_labels.add(name)
            roots.append(RootSpec(name, path, source_root_hash(path)))
            parent_configs = path.parent / "configs"
            if parent_configs.is_dir():
                config_dirs.append(parent_configs)
    roots.sort(key=lambda item: (item.name, str(item.path)))
    return roots, config_dirs


def load_scan_configs(
    config_tokens: Iterable[str],
    config_files: Iterable[str],
    search_dirs: Iterable[Path],
    state_dir: Path,
) -> list[AuditConfig]:
    dirs = list(search_dirs) + [Path(state_dir) / "configs"]
    configs: list[AuditConfig] = []
    seen_names: dict[str, str] = {}
    tokens = _split_csv(config_tokens)
    if not tokens and not list(config_files):
        tokens = ["cfgA"]
    for token in tokens:
        config = resolve_config(token, dirs)
        if config.name in seen_names:
            raise WorkbenchError(
                "E_ARGS",
                f"duplicate configuration name: {config.name}",
                {"config": config.name, "line": 1},
            )
        seen_names[config.name] = token
        configs.append(config)
    for file_value in config_files:
        config = resolve_config(file_value, dirs)
        if config.name in seen_names:
            raise WorkbenchError(
                "E_ARGS",
                f"duplicate configuration name: {config.name}",
                {"config": config.name, "line": 1},
            )
        seen_names[config.name] = file_value
        configs.append(config)
    configs.sort(key=lambda item: item.name)
    return configs


def _combine_call(
    call: dict[str, Any],
    root_name: str,
    config_name: str,
) -> dict[str, Any]:
    combined = dict(call)
    combined["root"] = root_name
    combined["config"] = config_name
    return combined


def _summarize(
    calls: list[dict[str, Any]],
    parse_failures: list[dict[str, Any]],
    combinations: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = _empty_summary()
    summary["calls_total"] = len(calls)
    for state in ("reachable", "unreachable", "unknown"):
        summary["reachability"][state] = sum(
            1 for call in calls if call["reachability"] == state
        )
    for state in ("tainted", "untainted", "unknown"):
        summary["taint"][state] = sum(1 for call in calls if call["taint"] == state)
    summary["high_risk"] = sum(1 for call in calls if call["high_risk"])
    summary["entry_missing"] = sorted(
        {
            entry
            for combination in combinations
            for entry in combination.get("entry_missing", [])
        }
    )
    summary["parse_failures"] = len(parse_failures)
    summary["macro_expansions"] = sum(
        int(combination.get("macro_expansions", 0))
        for combination in combinations
    )
    summary["combinations_total"] = len(combinations)
    summary["combinations_ok"] = sum(
        1 for combination in combinations if combination["status"] == "ok"
    )
    summary["timed_out"] = sum(
        1 for combination in combinations if combination["status"] == "timed_out"
    )
    summary["errors"] = len(errors)
    summary["active_children"] = active_child_count()
    return summary


def _sort_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        calls,
        key=lambda call: (
            call["root"],
            call["config"],
            call["location"]["file"],
            call["location"]["line"],
            call["location"]["column"],
            call["id"],
            call["fingerprint"],
        ),
    )


def _sort_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for error in errors:
        key = canonical_json(error)
        unique[key] = error
    return sorted(
        unique.values(),
        key=lambda error: (
            error["code"],
            error["message"],
            canonical_json(error["context"]),
        ),
    )


def run_audit(
    roots: list[RootSpec],
    configs: list[AuditConfig],
    *,
    state_dir: Path,
    jobs: int,
    timeout: float,
    max_hops: int,
) -> tuple[dict[str, Any], str]:
    ensure_state_dir(state_dir)
    combinations = [
        (root.name, root.path, config)
        for root in roots
        for config in configs
    ]
    combinations.sort(key=lambda item: (item[0], item[2].name))
    results = run_combinations(
        combinations,
        jobs=jobs,
        timeout=timeout,
        state_dir=state_dir,
        max_hops=max_hops,
    )

    calls: list[dict[str, Any]] = []
    parse_failures: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    combination_records: list[dict[str, Any]] = []
    root_by_name = {root.name: root for root in roots}
    config_by_name = {config.name: config for config in configs}
    for root_name, root_path, config, result in results:
        for call in result.get("calls", []):
            calls.append(_combine_call(call, root_name, config.name))
        for failure in result.get("parse_failures", []):
            item = dict(failure)
            item["root"] = root_name
            item["config"] = config.name
            parse_failures.append(item)
        combination_errors = []
        for error in result.get("errors", []):
            item = {
                "code": str(error.get("code") or "E_WORKER"),
                "message": str(error.get("message") or "worker error"),
                "context": dict(error.get("context") or {}),
            }
            item["context"].setdefault("root", root_name)
            item["context"].setdefault("config", config.name)
            item["context"].setdefault("combination", f"{root_name}/{config.name}")
            errors.append(item)
            combination_errors.append(item)
        combination_records.append(
            {
                "root": root_name,
                "config": config.name,
                "root_hash": root_by_name[root_name].root_hash,
                "config_hash": config_by_name[config.name].content_hash(),
                "status": result.get("status", "ok"),
                "call_count": len(result.get("calls", [])),
                "high_risk": result.get("summary", {}).get("high_risk", 0),
                "parse_failure_count": result.get("parse_failure_count", 0),
                "entry_missing": result.get("summary", {}).get("entry_missing", []),
                "included_files": result.get("included_files", []),
                "macro_expansions": result.get("macro_expansions", 0),
                "translation_units": result.get("translation_units", []),
                "errors": combination_errors,
            }
        )

    calls = _sort_calls(calls)
    parse_failures.sort(
        key=lambda item: (
            item["root"],
            item["config"],
            item["file"],
            item["line"],
            item["reason"],
            item["snippet"],
        )
    )
    errors = _sort_errors(errors)
    summary = _summarize(calls, parse_failures, combination_records, errors)
    roots_payload = [
        {
            "name": root.name,
            "path": str(root.path),
            "root_hash": root.root_hash,
            "file_count": len(
                [
                    path
                    for path in root.path.rglob("*")
                    if path.is_file()
                    and path.suffix.lower()
                    in {".c", ".h", ".cc", ".cpp", ".c++", ".hpp", ".hh", ".hxx"}
                ]
            ),
            "max_directory_depth": max(
                (
                    len(path.relative_to(root.path).parts) - 1
                    for path in root.path.rglob("*")
                    if path.is_file()
                    and path.suffix.lower()
                    in {".c", ".h", ".cc", ".cpp", ".c++", ".hpp", ".hh", ".hxx"}
                ),
                default=0,
            ),
        }
        for root in roots
    ]
    configs_payload = [
        {
            "name": config.name,
            "config_hash": config.content_hash(),
            "config": config.to_dict(),
        }
        for config in configs
    ]
    payload = snapshot_payload(
        roots=roots_payload,
        configs=configs_payload,
        calls=calls,
        parse_failures=parse_failures,
        combinations=combination_records,
        errors=errors,
        summary=summary,
    )
    snapshot_id = write_snapshot(state_dir, payload)
    return payload, snapshot_id


def _gate_spec(spec: str) -> dict[str, Any]:
    spec = spec.strip()
    match = re.fullmatch(r"new-high\s*(>=|>)\s*([0-9]+)", spec)
    if match:
        return {
            "spec": spec,
            "type": "new-high",
            "operator": match.group(1),
            "threshold": int(match.group(2)),
        }
    if spec.startswith("transition=") or spec.startswith("transition:"):
        target = spec.split("=", 1)[1].strip() if "=" in spec else spec.split(":", 1)[1].strip()
        if not target:
            raise WorkbenchError(
                "E_ARGS",
                "empty transition gate target",
                {"spec": spec, "line": 1},
            )
        return {"spec": spec, "type": "transition", "target": target}
    raise WorkbenchError(
        "E_ARGS",
        f"unsupported --fail-on expression: {spec}",
        {"spec": spec, "line": 1},
    )


def _transition_matches(transition: dict[str, Any], target: str) -> bool:
    target = target.strip()
    if target.lower() in {"any", "*"}:
        return True
    if target.lower() in {"unreachable_to_reachable", "unreachable->reachable"}:
        target = "reachability:unreachable->reachable"
    elif target.lower() in {"untainted_to_tainted", "untainted->tainted"}:
        target = "taint:untainted->tainted"
    elif target.lower() in {"high_false_to_true", "high:false->true", "high-risk:false->true"}:
        target = "high_risk:false->true"
    if target in {"reachable", "unreachable", "unknown"}:
        return (
            "reachability" in transition["changed_fields"]
            and transition["to"].get("reachability") == target
        )
    if target in {"high", "high-risk", "tainted"}:
        field = "high_risk" if target in {"high", "high-risk"} else "taint"
        return field in transition["changed_fields"]
    match = re.fullmatch(
        r"(?:(reachability|taint|high_risk)\s*:\s*)?([^>-]+)\s*->\s*(.+)",
        target,
    )
    if match:
        field, old, new = match.group(1), match.group(2).strip(), match.group(3).strip()
        if field is None:
            if old in {"reachable", "unreachable", "unknown"} or new in {
                "reachable",
                "unreachable",
                "unknown",
            }:
                field = "reachability"
            elif old in {"tainted", "untainted", "unknown"} or new in {
                "tainted",
                "untainted",
            }:
                field = "taint"
            else:
                field = "high_risk"
        return (
            field in transition["changed_fields"]
            and str(transition["from"].get(field)).lower() == old.lower()
            and str(transition["to"].get(field)).lower() == new.lower()
        )
    return False


def _common_call_count(baseline: dict[str, Any], current: dict[str, Any]) -> int:
    def keys(payload: dict[str, Any]) -> set[tuple[str, str, str, str]]:
        return {
            (call["root"], call["config"], call["id"], call.get("fingerprint", ""))
            for call in payload.get("calls", [])
        }

    return len(keys(baseline) & keys(current))


def apply_baseline_and_gate(
    payload: dict[str, Any],
    *,
    state_dir: Path,
    baseline_id: str | None,
    baseline_payload: dict[str, Any] | None,
    gate_specs: list[str],
    strict_content_lock: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    errors: list[dict[str, Any]] = list(payload.get("errors", []))
    diff: dict[str, Any] | None = None
    gate: dict[str, Any] | None = None
    if baseline_id is None:
        if gate_specs:
            raise WorkbenchError(
                "E_ARGS",
                "--fail-on requires --baseline",
                {"option": "--fail-on", "line": 1},
            )
        return diff, errors, gate
    if baseline_payload is None:
        try:
            baseline_payload, _ = load_snapshot(state_dir, baseline_id)
        except WorkbenchError as exc:
            errors.append(exc.to_dict())
            return None, _sort_errors(errors), None
    baseline = baseline_payload
    diff = diff_payloads(baseline, payload, baseline_id)
    if not gate_specs:
        return diff, errors, gate

    parsed_specs = [_gate_spec(spec) for spec in gate_specs]
    baseline_roots = {item["name"] for item in baseline.get("roots", [])}
    current_roots = {item["name"] for item in payload.get("roots", [])}
    root_set_match = baseline_roots == current_roots
    compatible = diff["config_content_match"] and root_set_match
    if strict_content_lock and not diff["content_match"]:
        compatible = False
    if not diff["root_content_match"] and compatible:
        # Incremental source evolution is guardable when stable semantic
        # identities overlap.  A wholesale replacement with no shared
        # fingerprint is rejected as a baseline mismatch instead of silently
        # reusing old entries.
        if _common_call_count(baseline, payload) == 0:
            compatible = False
    if not compatible:
        context = {
            "snapshot": baseline_id,
            "line": 1,
            "root_content_match": diff["root_content_match"],
            "config_content_match": diff["config_content_match"],
            "baseline_root_hashes": {
                item["name"]: item["root_hash"] for item in baseline.get("roots", [])
            },
            "current_root_hashes": {
                item["name"]: item["root_hash"] for item in payload.get("roots", [])
            },
            "baseline_config_hashes": {
                item["name"]: item["config_hash"] for item in baseline.get("configs", [])
            },
            "current_config_hashes": {
                item["name"]: item["config_hash"] for item in payload.get("configs", [])
            },
        }
        errors.append(
            AuditError(
                "E_BASELINE_MISMATCH",
                "baseline source/configuration content does not match the current audit",
                context,
            ).to_dict()
        )
        gate = {
            "status": "rejected",
            "specs": [spec["spec"] for spec in parsed_specs],
            "reason": "E_BASELINE_MISMATCH",
        }
        return diff, _sort_errors(errors), gate

    failed_specs: list[dict[str, Any]] = []
    for spec in parsed_specs:
        if spec["type"] == "new-high":
            count = diff["new_high_count"]
            failed = (
                count > spec["threshold"]
                if spec["operator"] == ">"
                else count >= spec["threshold"]
            )
            if failed:
                failed_specs.append(
                    {
                        "spec": spec["spec"],
                        "new_high_count": count,
                        "threshold": spec["threshold"],
                        "operator": spec["operator"],
                    }
                )
        else:
            matched = [
                transition
                for transition in diff["transitions"]
                if _transition_matches(transition, spec["target"])
            ]
            if matched:
                failed_specs.append(
                    {
                        "spec": spec["spec"],
                        "matched_transitions": matched,
                    }
                )
    if failed_specs:
        gate = {
            "status": "failed",
            "specs": [spec["spec"] for spec in parsed_specs],
            "failed": failed_specs,
            "new_high_count": diff["new_high_count"],
        }
        errors.append(
            AuditError(
                "E_GUARD_FAILED",
                "baseline guard threshold or transition rule was triggered",
                {
                    "snapshot": baseline_id,
                    "failed_specs": failed_specs,
                    "new_high_count": diff["new_high_count"],
                },
            ).to_dict()
        )
    else:
        gate = {
            "status": "passed",
            "specs": [spec["spec"] for spec in parsed_specs],
            "new_high_count": diff["new_high_count"],
        }
    return diff, _sort_errors(errors), gate


def build_report(
    payload: dict[str, Any],
    snapshot_id: str,
    *,
    diff: dict[str, Any] | None,
    errors: list[dict[str, Any]],
    gate: dict[str, Any] | None,
) -> dict[str, Any]:
    errors = _sort_errors(errors)
    calls = _sort_calls(payload.get("calls", []))
    parse_failures = payload.get("parse_failures", [])
    combinations = payload.get("combinations", [])
    summary = _summarize(calls, parse_failures, combinations, errors)
    code = "OK"
    if errors:
        if gate and gate.get("status") == "failed":
            code = "E_GUARD_FAILED"
        else:
            code = errors[0]["code"]
    report = {
        "format": REPORT_FORMAT,
        "code": code,
        "tool_version": __version__,
        "rules_version": RULES_VERSION,
        "snapshot_id": snapshot_id,
        "roots": payload.get("roots", []),
        "configs": payload.get("configs", []),
        "combinations": combinations,
        "calls": calls,
        "parse_failures": parse_failures,
        "errors": errors,
        "summary": summary,
        "baseline": diff,
        "gate": gate,
    }
    if errors:
        report["message"] = errors[0]["message"]
        report["context"] = errors[0]["context"]
    return report


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def report_text(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "AUDIT WORKBENCH REPORT",
        f"code={report.get('code')}",
        f"snapshot_id={report.get('snapshot_id')}",
        f"rules_version={report.get('rules_version')}",
        f"calls_total={summary.get('calls_total', 0)}",
        f"reachable={summary.get('reachability', {}).get('reachable', 0)}",
        f"unreachable={summary.get('reachability', {}).get('unreachable', 0)}",
        f"unknown={summary.get('reachability', {}).get('unknown', 0)}",
        f"tainted={summary.get('taint', {}).get('tainted', 0)}",
        f"untainted={summary.get('taint', {}).get('untainted', 0)}",
        f"taint_unknown={summary.get('taint', {}).get('unknown', 0)}",
        f"high_risk={summary.get('high_risk', 0)}",
        f"parse_failures={summary.get('parse_failures', 0)}",
        f"macro_expansions={summary.get('macro_expansions', 0)}",
        f"timed_out={summary.get('timed_out', 0)}",
        f"errors={summary.get('errors', 0)}",
    ]
    for call in report.get("calls", []):
        location = call["location"]
        lines.append(
            "CALL"
            f"|id={call['id']}"
            f"|root={call['root']}"
            f"|config={call['config']}"
            f"|file={location['file']}"
            f"|line={location['line']}"
            f"|column={location['column']}"
            f"|reachability={call['reachability']}"
            f"|taint={call['taint']}"
            f"|high_risk={str(call['high_risk']).lower()}"
            f"|rule={call['rule']}"
            f"|path_hops={len(call['path'])}"
            f"|truncated={str(call['path_truncated']).lower()}"
        )
    for failure in report.get("parse_failures", []):
        lines.append(
            "PARSE_FAILURE"
            f"|root={failure['root']}"
            f"|config={failure['config']}"
            f"|file={failure['file']}"
            f"|line={failure['line']}"
            f"|reason={failure['reason']}"
        )
    for error in report.get("errors", []):
        lines.append(
            "ERROR"
            f"|code={error['code']}"
            f"|message={error['message']}"
            f"|context={canonical_json(error['context'])}"
        )
    if report.get("gate"):
        lines.append(
            f"GATE|status={report['gate'].get('status')}"
            f"|specs={canonical_json(report['gate'].get('specs', []))}"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def minimal_error_report(
    error: dict[str, Any],
    *,
    snapshot_id: str = "",
) -> dict[str, Any]:
    summary = _empty_summary()
    summary.update(
        {
            "parse_failures": 0,
            "combinations_total": 0,
            "combinations_ok": 0,
            "timed_out": 0,
            "errors": 1,
            "active_children": active_child_count(),
        }
    )
    report = {
        "format": REPORT_FORMAT,
        "code": error["code"],
        "tool_version": __version__,
        "rules_version": RULES_VERSION,
        "snapshot_id": snapshot_id,
        "roots": [],
        "configs": [],
        "combinations": [],
        "calls": [],
        "parse_failures": [],
        "errors": [error],
        "summary": summary,
        "baseline": None,
        "gate": None,
    }
    report["message"] = error["message"]
    report["context"] = dict(error["context"])
    return report

