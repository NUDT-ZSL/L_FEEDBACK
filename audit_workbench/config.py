"""Compile-configuration model with canonical JSON and stable validation errors."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .errors import AuditError, WorkbenchError
from .hashing import canonical_json, sha256_text


@dataclass(frozen=True)
class MacroSetting:
    name: str
    value: str | None = None
    defined: bool = True

    def normalized(self) -> tuple[str, bool, str | None]:
        if not self.defined:
            return (self.name, False, None)
        return (self.name, True, self.value if self.value is not None else "1")

    def to_dict(self) -> dict[str, Any]:
        if not self.defined:
            return {"name": self.name, "defined": False}
        return {"name": self.name, "defined": True, "value": self.value or ""}


@dataclass
class AuditConfig:
    name: str
    macros: dict[str, MacroSetting] = field(default_factory=dict)
    include_paths: list[str] = field(default_factory=list)
    language_standard: str = "c11"
    entries: list[str] = field(default_factory=list)

    def normalized_macros(self) -> dict[str, tuple[bool, str | None]]:
        return {
            name: setting.normalized()[1:]
            for name, setting in sorted(self.macros.items())
        }

    def equivalent(self, other: "AuditConfig") -> bool:
        """Compare semantic config content; the display name is intentionally ignored."""

        return (
            self.normalized_macros() == other.normalized_macros()
            and [p.replace("\\", "/") for p in self.include_paths]
            == [p.replace("\\", "/") for p in other.include_paths]
            and self.language_standard.strip().lower()
            == other.language_standard.strip().lower()
            and sorted(self.entries) == sorted(other.entries)
        )

    def to_dict(self, include_name: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "include_paths": list(self.include_paths),
            "language_standard": self.language_standard,
            "macros": [
                self.macros[name].to_dict()
                for name in sorted(self.macros)
            ],
            "entries": sorted(self.entries),
        }
        if include_name:
            result["name"] = self.name
        return result

    def content_hash(self) -> str:
        return sha256_text(canonical_json(self.to_dict(include_name=True)))

    def serialized(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


BUILTIN_CONFIGS: dict[str, dict[str, Any]] = {
    "cfgA": {
        "name": "cfgA",
        "macros": [
            {"name": "ENABLE_UNSAFE", "defined": True, "value": "1"},
            {"name": "FEATURE_ALT", "defined": False},
            {"name": "FEATURE_B_ONLY", "defined": False},
            {"name": "HANG_MODE", "defined": False},
        ],
        "include_paths": [],
        "language_standard": "c11",
        "entries": ["entry_unit_a", "entry_unit_b", "entry_unit_c"],
    },
    "cfgB": {
        "name": "cfgB",
        "macros": [
            {"name": "ENABLE_UNSAFE", "defined": False},
            {"name": "FEATURE_ALT", "defined": False},
            {"name": "FEATURE_B_ONLY", "defined": True, "value": "1"},
            {"name": "HANG_MODE", "defined": False},
        ],
        "include_paths": [],
        "language_standard": "c11",
        "entries": ["entry_unit_a", "entry_unit_b", "entry_unit_c"],
    },
    "cfgC": {
        "name": "cfgC",
        "macros": [
            {"name": "ENABLE_UNSAFE", "defined": False},
            {"name": "FEATURE_ALT", "defined": True, "value": "1"},
            {"name": "FEATURE_B_ONLY", "defined": False},
            {"name": "HANG_MODE", "defined": True, "value": "1"},
        ],
        "include_paths": [],
        "language_standard": "c11",
        "entries": ["entry_unit_a", "entry_unit_b", "entry_unit_c"],
    },
}


_MACRO_NAME_RE = re.compile(r'^\s*[A-Za-z_][A-Za-z0-9_]*$')


def _line_for_macro_occurrence(raw: str, occurrence: int, name: str) -> int:
    """Best-effort JSON source line lookup for a duplicate macro key."""

    pattern = re.compile(r'"name"\s*:\s*' + re.escape(json.dumps(name)))
    matches = list(pattern.finditer(raw))
    if occurrence < len(matches):
        return raw.count("\n", 0, matches[occurrence].start()) + 1
    # Object-style "MODE": repeated keys are also searched without quotes differences.
    pattern2 = re.compile(r'"' + re.escape(name) + r'"\s*:')
    matches2 = list(pattern2.finditer(raw))
    if occurrence < len(matches2):
        return raw.count("\n", 0, matches2[occurrence].start()) + 1
    return 1


def _macros_object_text(raw: str) -> tuple[str, int] | None:
    match = re.search(r'"macros"\s*:\s*', raw)
    if not match:
        return None
    start = match.end()
    while start < len(raw) and raw[start].isspace():
        start += 1
    if start >= len(raw) or raw[start] != "{":
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start : index + 1], start
    return None


def _object_macro_value(value: Any) -> tuple[bool, str | None]:
    if value is None or value is False:
        return (False, None)
    if value is True:
        return (True, "1")
    return (True, str(value))


def _check_object_macro_duplicates(raw_json: str, source: str) -> None:
    located = _macros_object_text(raw_json)
    if located is None:
        return
    object_text, _ = located
    try:
        pairs = json.loads(object_text, object_pairs_hook=lambda value: value)
    except json.JSONDecodeError:
        return
    if not isinstance(pairs, list):
        return
    seen: dict[str, tuple[bool, str | None]] = {}
    occurrences: dict[str, int] = {}
    for key, value in pairs:
        if not isinstance(key, str):
            continue
        occurrences[key] = occurrences.get(key, 0) + 1
        normalized = _object_macro_value(value)
        if key in seen and seen[key] != normalized:
            line = _line_for_macro_occurrence(raw_json, occurrences[key] - 1, key)
            raise WorkbenchError(
                "E_MACRO_CONFLICT",
                f"macro {key!r} is defined with conflicting values",
                {"file": source, "line": line, "macro": key},
            )
        seen[key] = normalized


def _parse_macro_list(
    raw_json: str,
    data: Any,
    source: str,
) -> dict[str, MacroSetting]:
    settings: dict[str, MacroSetting] = {}
    occurrences: dict[str, int] = {}

    def add(setting: MacroSetting) -> None:
        if not _MACRO_NAME_RE.match(setting.name):
            raise WorkbenchError(
                "E_CONFIG",
                f"invalid macro name {setting.name!r}",
                {"file": source, "line": 1, "macro": setting.name},
            )
        occurrences[setting.name] = occurrences.get(setting.name, 0) + 1
        total = occurrences[setting.name]
        previous = settings.get(setting.name)
        if previous is not None and previous.normalized() != setting.normalized():
            line = _line_for_macro_occurrence(raw_json, total - 1, setting.name)
            raise WorkbenchError(
                "E_MACRO_CONFLICT",
                f"macro {setting.name!r} is defined with conflicting values",
                {"file": source, "line": line, "macro": setting.name},
            )
        settings[setting.name] = setting

    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                if "=" in item:
                    name, value = item.split("=", 1)
                    add(MacroSetting(name.strip(), value))
                else:
                    add(MacroSetting(item.strip(), "1"))
                continue
            if not isinstance(item, dict) or "name" not in item:
                raise WorkbenchError(
                    "E_CONFIG",
                    "macro entries must contain a name",
                    {"file": source, "line": 1},
                )
            name = str(item["name"])
            if item.get("defined", True) is False:
                add(MacroSetting(name, None, False))
            else:
                value = item.get("value", "1")
                add(MacroSetting(name, "" if value is None else str(value), True))
        return settings

    if isinstance(data, dict):
        _check_object_macro_duplicates(raw_json, source)
        for name, value in data.items():
            if value is None:
                add(MacroSetting(str(name), None, False))
            elif isinstance(value, bool):
                add(MacroSetting(str(name), "1" if value else None, value))
            else:
                add(MacroSetting(str(name), str(value), True))
        return settings

    if data is None:
        return settings
    raise WorkbenchError(
        "E_CONFIG",
        "macros must be a JSON object or array",
        {"file": source, "line": 1},
    )


def config_from_dict(
    data: dict[str, Any],
    *,
    source: str = "<memory>",
    raw_json: str | None = None,
) -> AuditConfig:
    raw_json = raw_json if raw_json is not None else json.dumps(data)
    if not isinstance(data, dict):
        raise WorkbenchError(
            "E_CONFIG",
            "configuration must be a JSON object",
            {"file": source, "line": 1},
        )
    name = str(data.get("name") or Path(source).stem or "config")
    macros = _parse_macro_list(raw_json, data.get("macros", []), source)
    include_paths_raw = data.get("include_paths", [])
    if not isinstance(include_paths_raw, list):
        raise WorkbenchError(
            "E_CONFIG",
            "include_paths must be an ordered JSON array",
            {"file": source, "line": 1},
        )
    include_paths = [str(item) for item in include_paths_raw]
    entries_raw = data.get("entries", data.get("entry_functions", []))
    if not isinstance(entries_raw, list):
        raise WorkbenchError(
            "E_CONFIG",
            "entries must be a JSON array",
            {"file": source, "line": 1},
        )
    entries = [str(item) for item in entries_raw]
    language_standard = str(data.get("language_standard", "c11"))
    return AuditConfig(
        name=name,
        macros=macros,
        include_paths=include_paths,
        language_standard=language_standard,
        entries=entries,
    )


def load_config_file(path: Path) -> AuditConfig:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkbenchError(
            "E_IO",
            f"cannot read configuration: {exc}",
            {"file": str(path), "line": 1},
        ) from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorkbenchError(
            "E_CONFIG",
            f"invalid configuration JSON: {exc.msg}",
            {"file": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    return config_from_dict(data, source=str(path), raw_json=raw)


def load_config_from_json(raw: str, source: str = "<memory>") -> AuditConfig:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorkbenchError(
            "E_CONFIG",
            f"invalid configuration JSON: {exc.msg}",
            {"file": source, "line": exc.lineno, "column": exc.colno},
        ) from exc
    return config_from_dict(data, source=source, raw_json=raw)


def resolve_config(
    token: str,
    search_dirs: Iterable[Path] = (),
) -> AuditConfig:
    """Resolve a config path, JSON file stem, bundle config, or builtin ID."""

    candidate = Path(token)
    if candidate.is_file():
        return load_config_file(candidate)
    for directory in search_dirs:
        file_candidate = directory / (token if token.endswith(".json") else token + ".json")
        if file_candidate.is_file():
            return load_config_file(file_candidate)
    if token in BUILTIN_CONFIGS:
        return config_from_dict(dict(BUILTIN_CONFIGS[token]), source=f"<builtin:{token}>")
    # Permit a bare path with a .json suffix to produce a useful IO error.
    if token.endswith(".json"):
        raise WorkbenchError(
            "E_IO",
            f"configuration file not found: {token}",
            {"file": token, "line": 1},
        )
    raise WorkbenchError(
        "E_ARGS",
        f"unknown configuration identifier: {token}",
        {"config": token, "line": 1},
    )

