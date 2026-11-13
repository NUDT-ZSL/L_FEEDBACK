"""Manifest parsing and source-order policy helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .versions import is_valid_version


_REQ_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:\[[^\]]*\])?"
    r"\s*(?P<op>==|>=)\s*"
    r"(?P<version>[^\s#]+)"
    r"(?P<rest>.*)$"
)
_HASH_RE = re.compile(r"--hash(?:=|\s+)sha256:([0-9a-fA-F]{64})(?:\s|$)")
_POLICY_COMMENT_RE = re.compile(r"^\s*#\s*depconfusion-policy\s*:\s*private-first\s*$", re.I)
_POLICY_FILENAME = ".depconfusion-policy.json"


@dataclass(frozen=True)
class Requirement:
    package: str
    operator: str
    version: str
    hashes: tuple[str, ...] = ()
    line_number: int = 0
    source_line: str = ""
    comment: str = ""

    @property
    def text(self) -> str:
        return f"{self.package}{self.operator}{self.version}"


@dataclass
class ParseResult:
    requirements: list[Requirement] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_order: tuple[str, str] = ("public", "private")


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def parse_line(raw_line: str, line_number: int = 0) -> Requirement | None:
    """Parse one manifest line, or return None for comments/unsupported lines."""

    stripped = raw_line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    content, marker, comment = stripped.partition("#")
    content = content.strip()
    if not content:
        return None

    match = _REQ_RE.match(content)
    if not match:
        raise ValueError("应为 name==version 或 name>=version")

    package = normalize_name(match.group("name"))
    operator = match.group("op")
    version_text = match.group("version")
    if not is_valid_version(version_text):
        raise ValueError(f"非法版本 {version_text!r}")

    rest = match.group("rest")
    hashes = tuple(hash_value.lower() for hash_value in _HASH_RE.findall(rest))
    leftover = _HASH_RE.sub("", rest).strip()
    # Keep comments and unrelated pip options out of the strict parser's way.
    if leftover and not leftover.startswith("-"):
        raise ValueError(f"无法识别的内容: {leftover!r}")

    return Requirement(
        package=package,
        operator=operator,
        version=version_text,
        hashes=hashes,
        line_number=line_number,
        source_line=raw_line.rstrip("\r\n"),
        comment=comment.strip() if marker else "",
    )


def parse_manifest_text(text: str) -> ParseResult:
    parsed = ParseResult()
    order_set = False
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if _POLICY_COMMENT_RE.match(raw_line):
            parsed.source_order = ("private", "public")
            order_set = True
            continue
        try:
            requirement = parse_line(raw_line, line_number)
        except ValueError as exc:
            if raw_line.strip() and not raw_line.lstrip().startswith("#"):
                parsed.warnings.append(f"第 {line_number} 行已跳过: {exc}")
            continue
        if requirement is not None:
            parsed.requirements.append(requirement)
    if order_set:
        parsed.source_order = ("private", "public")
    return parsed


def parse_manifest(path: str | Path) -> ParseResult:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(str(manifest_path))
    return parse_manifest_text(manifest_path.read_text(encoding="utf-8"))


def policy_file_for(manifest_path: str | Path) -> Path:
    return Path(manifest_path).resolve().parent / _POLICY_FILENAME


def detect_external_policy(manifest_path: str | Path) -> tuple[str, str] | None:
    policy_path = policy_file_for(manifest_path)
    if not policy_path.is_file():
        return None
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    order = payload.get("source_order")
    if order == ["private", "public"]:
        return ("private", "public")
    return None


def manifest_source_order(manifest_path: str | Path) -> tuple[str, str]:
    parsed = parse_manifest(manifest_path)
    if parsed.source_order == ("private", "public"):
        return parsed.source_order
    external = detect_external_policy(manifest_path)
    if external:
        return external
    return parsed.source_order


def discover_manifests(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(str(root))
    candidates = [
        path
        for path in root.rglob("*.txt")
        if path.is_file() and "_snapshot" not in path.name
    ]
    return sorted(candidates, key=lambda item: str(item).lower())


def write_manifest(path: str | Path, requirements: Iterable[Requirement]) -> None:
    """Atomically replace a manifest while preserving a private-first marker."""

    manifest_path = Path(path)
    existing_order = manifest_source_order(manifest_path) if manifest_path.exists() else ("public", "private")
    lines: list[str] = []
    if existing_order == ("private", "public"):
        lines.append("# depconfusion-policy: private-first")
    for requirement in requirements:
        suffix = ""
        for hash_value in requirement.hashes:
            suffix += f" --hash=sha256:{hash_value}"
        if requirement.comment:
            suffix += f"  # {requirement.comment}"
        lines.append(f"{requirement.package}{requirement.operator}{requirement.version}{suffix}")
    text = "\n".join(lines) + "\n"
    atomic_write_text(manifest_path, text)


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(target)
