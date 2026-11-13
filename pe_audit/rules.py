"""Defect-rule loading, strict validation and per-function evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .binary import ImportSymbol
from .callgraph import CallSite, ParsedFunction
from .errors import AuditError, rule_error


@dataclass(frozen=True)
class ImportPattern:
    module: str | None
    name: str

    def matches(self, symbol: ImportSymbol) -> bool:
        return self.name == symbol.name and (
            self.module is None or self.module == symbol.module
        )

    def label(self) -> str:
        if self.module is None:
            return self.name
        return f"{self.module}!{self.name}"


@dataclass(frozen=True)
class RuleSpec:
    id: str
    target: str
    dangerous_imports: tuple[ImportPattern, ...]
    rule_type: str
    branch_number: int | None
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        rule: dict[str, Any] = {"type": self.rule_type}
        if self.branch_number is not None:
            rule["branch_number"] = self.branch_number
        return {
            "id": self.id,
            "target": self.target,
            "dangerous_imports": sorted(
                item.label() for item in self.dangerous_imports
            ),
            "rule": rule,
        }


@dataclass(frozen=True)
class DefectDocument:
    rules: tuple[RuleSpec, ...]
    entry_points: tuple[str, ...]
    raw: dict[str, Any]


def _parse_import_pattern(value: Any, rule_id: str) -> ImportPattern:
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise rule_error(
                "dangerous import name must not be empty", rule=rule_id
            )
        if "!" in value:
            module, name = value.split("!", 1)
            if not module or not name:
                raise rule_error(
                    "dangerous import module/name must not be empty", rule=rule_id
                )
            return ImportPattern(module, name)
        return ImportPattern(None, value)
    if not isinstance(value, Mapping):
        raise rule_error(
            "dangerous_imports entries must be objects or strings", rule=rule_id
        )
    if "module" not in value:
        raise rule_error("dangerous import is missing module", rule=rule_id)
    if "name" not in value:
        raise rule_error("dangerous import is missing name", rule=rule_id)
    module = value["module"]
    name = value["name"]
    if not isinstance(module, str) or not module:
        raise rule_error(
            "dangerous import module must be a non-empty string", rule=rule_id
        )
    if not isinstance(name, str) or not name:
        raise rule_error(
            "dangerous import name must be a non-empty string", rule=rule_id
        )
    return ImportPattern(module, name)


def validate_defects(
    document: Mapping[str, Any],
    known_functions: Iterable[str] | None = None,
) -> DefectDocument:
    if not isinstance(document, Mapping):
        raise rule_error("defect document root must be an object")
    raw_rules = document.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise rule_error("rules must be a non-empty list")
    known = set(known_functions) if known_functions is not None else None
    rules: list[RuleSpec] = []
    identifiers: set[str] = set()
    for index, item in enumerate(raw_rules):
        if not isinstance(item, Mapping):
            raise rule_error(f"rules[{index}] must be an object")
        for field in ("id", "target", "dangerous_imports", "rule"):
            if field not in item:
                raise rule_error(
                    f"rules[{index}] is missing {field}", index=index
                )
        rule_id = item["id"]
        target = item["target"]
        if not isinstance(rule_id, str) or not rule_id:
            raise rule_error("rule id must be a non-empty string", index=index)
        if rule_id in identifiers:
            raise rule_error("duplicate rule id", id=rule_id)
        identifiers.add(rule_id)
        if not isinstance(target, str) or not target:
            raise rule_error("rule target must be a non-empty string", id=rule_id)
        if known is not None and target not in known:
            raise rule_error(
                "rule target is absent from both function manifests",
                id=rule_id,
                target=target,
            )
        raw_imports = item["dangerous_imports"]
        if not isinstance(raw_imports, list) or not raw_imports:
            raise rule_error(
                "dangerous_imports must be a non-empty list", id=rule_id
            )
        patterns = tuple(
            sorted(
                (
                    _parse_import_pattern(pattern, rule_id)
                    for pattern in raw_imports
                ),
                key=lambda pattern: (pattern.module or "", pattern.name),
            )
        )
        raw_rule = item["rule"]
        if not isinstance(raw_rule, Mapping):
            raise rule_error("rule must be an object", id=rule_id)
        rule_type = raw_rule.get("type")
        if rule_type not in {"forbid_call", "guarded_call"}:
            raise rule_error(
                "rule.type must be forbid_call or guarded_call", id=rule_id
            )
        branch_number: int | None = None
        if rule_type == "guarded_call":
            if "branch_number" not in raw_rule:
                raise rule_error(
                    "guarded_call requires branch_number", id=rule_id
                )
            branch_number = raw_rule["branch_number"]
            if (
                isinstance(branch_number, bool)
                or not isinstance(branch_number, int)
                or branch_number < 0
            ):
                raise rule_error(
                    "guarded_call branch_number must be a non-negative integer",
                    id=rule_id,
                )
        rules.append(
            RuleSpec(
                id=rule_id,
                target=target,
                dangerous_imports=patterns,
                rule_type=str(rule_type),
                branch_number=branch_number,
                raw=dict(item),
            )
        )

    raw_entries = document.get("entry_points")
    if raw_entries is None:
        raw_entries = []
    if not isinstance(raw_entries, list) or not all(
        isinstance(entry, str) and entry for entry in raw_entries
    ):
        raise rule_error("entry_points must be a list of non-empty strings")
    rules.sort(key=lambda rule: rule.id)
    return DefectDocument(
        rules=tuple(rules),
        entry_points=tuple(sorted(set(raw_entries))),
        raw=dict(document),
    )


def load_defects(
    path: str | Path,
    known_functions: Iterable[str] | None = None,
) -> DefectDocument:
    try:
        with open(path, "r", encoding="utf-8") as stream:
            document = json.load(stream)
    except OSError as exc:
        raise AuditError("INPUT_READ_ERROR", f"defects: {exc}", 2) from exc
    except json.JSONDecodeError as exc:
        raise rule_error(
            "defect document is not valid JSON",
            line=exc.lineno,
            column=exc.colno,
        ) from exc
    return validate_defects(document, known_functions)


def _evidence_for_call(
    call: CallSite, expected_branch: int | None
) -> dict[str, Any]:
    assert call.kind == "import" and call.import_symbol is not None
    guarded = (
        expected_branch is not None
        and call.branch_number is not None
        and call.branch_number == expected_branch
    )
    return {
        "call_offset": call.offset,
        "branch_offset": call.branch_offset,
        "branch_number": call.branch_number,
        "expected_branch_number": expected_branch,
        "guarded": guarded,
        "import": call.import_symbol.to_dict(),
    }


def evaluate_function(
    rule: RuleSpec,
    parsed: ParsedFunction | None,
) -> dict[str, Any]:
    if parsed is None:
        return {
            "present": False,
            "deleted": True,
            "verdict": "CLEAN",
            "evidence": [],
            "evidence_complete": True,
            "parse_issues": [],
        }

    matches: list[CallSite] = []
    for call in parsed.calls:
        if call.kind != "import" or call.import_symbol is None:
            continue
        if any(
            pattern.matches(call.import_symbol)
            for pattern in rule.dangerous_imports
        ):
            matches.append(call)

    evidence = [
        _evidence_for_call(call, rule.branch_number) for call in matches
    ]
    if rule.rule_type == "forbid_call":
        vulnerable = bool(matches)
    else:
        vulnerable = any(not item["guarded"] for item in evidence)

    return {
        "present": True,
        "deleted": False,
        "verdict": "VULNERABLE" if vulnerable else "CLEAN",
        "evidence": evidence,
        "evidence_complete": parsed.complete,
        "parse_issues": list(parsed.parse_issues),
    }
