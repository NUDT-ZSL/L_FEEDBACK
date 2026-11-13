"""A deliberately small PEP 440-compatible version implementation."""

from __future__ import annotations

import re
from functools import total_ordering


_VERSION_RE = re.compile(
    r"^\s*v?"
    r"(?P<release>\d+(?:\.\d+)*)"
    r"(?:(?P<pre>a|alpha|b|beta|c|rc|pre)(?P<pre_num>\d+)?)?"
    r"(?:[-_.]?post(?P<post>\d+)?)?"
    r"(?:[-_.]?dev(?P<dev>\d+)?)?"
    r"\s*$",
    re.IGNORECASE,
)

_PRE_RANK = {
    "a": 0,
    "alpha": 0,
    "b": 1,
    "beta": 1,
    "c": 2,
    "rc": 2,
    "pre": 2,
}


def _trim_release(parts: tuple[int, ...]) -> tuple[int, ...]:
    trimmed = list(parts)
    while len(trimmed) > 1 and trimmed[-1] == 0:
        trimmed.pop()
    return tuple(trimmed)


@total_ordering
class Version:
    """Comparable version object for the simplified PEP 440 dialect.

    Supported forms include ``1``, ``1.2.3``, ``1.2rc1``, ``1.2.post1``,
    ``1.2.dev1`` and an optional leading ``v``.  Local version labels and
    arbitrary epochs are intentionally rejected by this lab.
    """

    __slots__ = ("text", "_key")

    def __init__(self, value: str):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("版本不能为空")
        match = _VERSION_RE.match(value)
        if not match:
            raise ValueError(f"无法解析版本: {value!r}")

        self.text = value.strip()
        if self.text[:1] in ("v", "V"):
            self.text = self.text[1:]

        release = _trim_release(tuple(int(part) for part in match.group("release").split(".")))
        pre = match.group("pre")
        post = match.group("post")
        dev = match.group("dev")

        if dev is not None:
            # A development release sorts before any pre-release/final release.
            phase_rank = -1
            phase_number = int(dev or 0)
        elif pre is not None:
            phase_rank = _PRE_RANK[pre.lower()]
            phase_number = int(match.group("pre_num") or 0)
        elif post is not None:
            phase_rank = 4
            phase_number = int(post or 0)
        else:
            phase_rank = 3
            phase_number = 0

        # The final slot keeps "1.0" before "1.0.post1" while normal releases
        # have no post suffix.
        self._key = (
            release,
            phase_rank,
            phase_number,
            1 if post is not None else 0,
            int(post or 0),
        )

    @property
    def key(self) -> tuple:
        return self._key

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Version({self.text!r})"

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            try:
                other = Version(other)
            except ValueError:
                return False
        if not isinstance(other, Version):
            return NotImplemented
        return self._key == other._key

    def __lt__(self, other: object) -> bool:
        if isinstance(other, str):
            other = Version(other)
        if not isinstance(other, Version):
            return NotImplemented
        return self._key < other._key


def parse_version(value: str) -> Version:
    return Version(value)


def is_valid_version(value: str) -> bool:
    try:
        Version(value)
    except (TypeError, ValueError):
        return False
    return True


def highest(versions: list[str] | tuple[str, ...] | set[str] | None) -> str | None:
    """Return the highest parseable version without changing its spelling."""

    if not versions:
        return None
    try:
        return max((Version(value) for value in versions), key=lambda item: item.key).text
    except ValueError as exc:
        raise ValueError(f"版本集合包含非法版本: {exc}") from exc


def satisfies(version: str, operator: str, wanted: str) -> bool:
    left = Version(version)
    right = Version(wanted)
    if operator == "==":
        return left == right
    if operator == ">=":
        return left >= right
    raise ValueError(f"不支持的版本运算符: {operator}")
