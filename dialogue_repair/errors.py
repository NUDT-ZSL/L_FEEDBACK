"""Structured business errors used by the command-line interface."""

from __future__ import annotations

from typing import Any, Iterable


def make_error(
    category: str,
    message: str,
    *,
    operation_index: int | None = None,
    field: str | None = None,
    related_operation_indices: Iterable[int] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "category": category,
        "operation_index": operation_index,
        "message": message,
    }
    if field is not None:
        item["field"] = field
    if related_operation_indices is not None:
        item["related_operation_indices"] = sorted(set(related_operation_indices))
    if details:
        item["details"] = dict(sorted(details.items()))
    return item


def error_sort_key(error: dict[str, Any]) -> tuple[Any, ...]:
    operation_index = error.get("operation_index")
    return (
        0 if operation_index is None else 1,
        -1 if operation_index is None else operation_index,
        str(error.get("category", "")),
        str(error.get("field", "")),
        str(error.get("message", "")),
    )


def sort_errors(errors: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for error in errors:
        key = repr(
            (
                error.get("category"),
                error.get("operation_index"),
                error.get("field"),
                error.get("message"),
                error.get("related_operation_indices"),
                error.get("details"),
            )
        )
        unique.setdefault(key, error)
    return sorted(unique.values(), key=error_sort_key)


class RepairFailure(Exception):
    """A failure that must be emitted as structured JSON."""

    def __init__(self, errors: Iterable[dict[str, Any]], *, exit_code: int = 2):
        self.errors = sort_errors(errors)
        self.exit_code = exit_code
        super().__init__("; ".join(error["message"] for error in self.errors))

    def payload(self) -> dict[str, Any]:
        return {
            "ok": False,
            "exit_code": self.exit_code,
            "error_count": len(self.errors),
            "errors": self.errors,
        }

