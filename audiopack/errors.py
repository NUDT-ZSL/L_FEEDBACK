"""Structured business error codes and issue collection.

Validation and asset loading never raise on the first problem.  Every problem
is converted into an :class:`Issue` carrying a stable business code so that the
caller can report all of them in one structured report.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


class ErrorCode:
    """Stable business codes surfaced in reports and CLI exit paths."""

    E_DUP_ID = "E_DUP_ID"
    E_BAD_TIME = "E_BAD_TIME"
    E_BAD_FIELD = "E_BAD_FIELD"
    E_BAD_LANG = "E_BAD_LANG"
    E_MISSING_ASSET = "E_MISSING_ASSET"
    E_CORRUPT_ASSET = "E_CORRUPT_ASSET"
    E_LOAD_FAILED = "E_LOAD_FAILED"
    E_SNAPSHOT = "E_SNAPSHOT"


@dataclass
class Issue:
    """A single collected business error.

    Context fields are optional so the same shape works for manifest rows,
    asset files and snapshot blocks.
    """

    code: str
    message: str
    event_id: str | None = None
    lang: str | None = None
    path: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IssueCollector:
    """Accumulates issues without short-circuiting."""

    def __init__(self) -> None:
        self._issues: list[Issue] = []

    def add(
        self,
        code: str,
        message: str,
        *,
        event_id: str | None = None,
        lang: str | None = None,
        path: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> Issue:
        issue = Issue(
            code=code,
            message=message,
            event_id=event_id,
            lang=lang,
            path=path,
            detail=detail or {},
        )
        self._issues.append(issue)
        return issue

    def extend(self, issues: list[Issue]) -> None:
        self._issues.extend(issues)

    @property
    def issues(self) -> list[Issue]:
        return self._issues

    @property
    def codes(self) -> list[str]:
        return [issue.code for issue in self._issues]

    def __bool__(self) -> bool:
        return bool(self._issues)

    def to_list(self) -> list[dict[str, Any]]:
        return [issue.to_dict() for issue in self._issues]
