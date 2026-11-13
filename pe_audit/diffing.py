"""Function-name-aligned effective-instruction diffing."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from .callgraph import CallAnalysis, EffectiveInstruction
from .manifests import Alignment


@dataclass
class FunctionDiff:
    function: str
    status: str
    before_invalid_bytes: int = 0
    after_invalid_bytes: int = 0
    added: list[dict[str, Any]] = field(default_factory=list)
    removed: list[dict[str, Any]] = field(default_factory=list)
    modified: list[dict[str, Any]] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "added": len(self.added),
            "removed": len(self.removed),
            "modified": len(self.modified),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "status": self.status,
            "before_invalid_bytes": self.before_invalid_bytes,
            "after_invalid_bytes": self.after_invalid_bytes,
            "changes": {
                "added": self.added,
                "removed": self.removed,
                "modified": self.modified,
            },
            "counts": self.counts,
        }


def _modified_pair(
    before_item: EffectiveInstruction | None,
    after_item: EffectiveInstruction | None,
    before_index: int,
    after_index: int,
) -> dict[str, Any]:
    before_kind = before_item.kind if before_item else None
    after_kind = after_item.kind if after_item else None
    return {
        "kind": before_kind if before_kind == after_kind else "mixed",
        "before_index": before_index,
        "after_index": after_index,
        "before": before_item.change_record(before_index) if before_item else None,
        "after": after_item.change_record(after_index) if after_item else None,
    }


def diff_function(
    *,
    before: CallAnalysis,
    after: CallAnalysis,
    function: str,
) -> FunctionDiff:
    before_fn = before.functions[function]
    after_fn = after.functions[function]
    left = [instruction.signature for instruction in before_fn.effective]
    right = [instruction.signature for instruction in after_fn.effective]
    matcher = SequenceMatcher(a=left, b=right, autojunk=False)
    added: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    modified: list[dict[str, Any]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            for index in range(i1, i2):
                removed.append(before_fn.effective[index].change_record(index))
            continue
        if tag == "insert":
            for index in range(j1, j2):
                added.append(after_fn.effective[index].change_record(index))
            continue
        paired = min(i2 - i1, j2 - j1)
        for delta in range(paired):
            before_index = i1 + delta
            after_index = j1 + delta
            modified.append(
                _modified_pair(
                    before_fn.effective[before_index],
                    after_fn.effective[after_index],
                    before_index,
                    after_index,
                )
            )
        for index in range(i1 + paired, i2):
            removed.append(before_fn.effective[index].change_record(index))
        for index in range(j1 + paired, j2):
            added.append(after_fn.effective[index].change_record(index))

    status = "modified" if (added or removed or modified) else "unchanged"
    return FunctionDiff(
        function=function,
        status=status,
        before_invalid_bytes=before_fn.invalid_bytes,
        after_invalid_bytes=after_fn.invalid_bytes,
        added=added,
        removed=removed,
        modified=modified,
    )


def diff_aligned_functions(
    *,
    before: CallAnalysis,
    after: CallAnalysis,
    alignment: Alignment,
) -> dict[str, Any]:
    details: list[FunctionDiff] = []
    unchanged: list[str] = []
    changed: list[str] = []
    total = {"added": 0, "removed": 0, "modified": 0}
    per_function: dict[str, dict[str, int]] = {}
    for name in alignment.common:
        result = diff_function(before=before, after=after, function=name)
        details.append(result)
        per_function[name] = result.counts
        for key, value in result.counts.items():
            total[key] += value
        if result.status == "unchanged":
            unchanged.append(name)
        else:
            changed.append(name)

    for name in alignment.removed:
        parsed = before.functions[name]
        removed_records = [
            instruction.change_record(index)
            for index, instruction in enumerate(parsed.effective)
        ]
        result = FunctionDiff(
            function=name,
            status="removed",
            before_invalid_bytes=parsed.invalid_bytes,
            removed=removed_records,
        )
        details.append(result)
        per_function[name] = result.counts
        total["removed"] += len(removed_records)
    for name in alignment.added:
        parsed = after.functions[name]
        added_records = [
            instruction.change_record(index)
            for index, instruction in enumerate(parsed.effective)
        ]
        result = FunctionDiff(
            function=name,
            status="added",
            after_invalid_bytes=parsed.invalid_bytes,
            added=added_records,
        )
        details.append(result)
        per_function[name] = result.counts
        total["added"] += len(added_records)

    details.sort(key=lambda item: item.function)
    return {
        "aligned_functions": list(alignment.common),
        "added_functions": list(alignment.added),
        "removed_functions": list(alignment.removed),
        "unchanged_functions": sorted(unchanged),
        "modified_functions": sorted(changed),
        "changed_instruction_counts": total,
        "per_function_instruction_counts": {
            name: per_function[name] for name in sorted(per_function)
        },
        "details": [item.to_dict() for item in details],
    }
