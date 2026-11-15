"""Load and validate planning manifest, tables, voices, and locale lines."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import BakeConfig, parse_config
from .constants import BAKER_VERSION
from .errors import InputFatal, Issue
from .util import loads_json_strict, object_hash, sha256_bytes


@dataclass(frozen=True)
class Harmonic:
    multiple: float
    gain: float
    phase_deg: float


@dataclass(frozen=True)
class Voice:
    role_name: str
    gain_db: float
    fundamental_hz: float
    amplitude: float
    vibrato_hz: float
    vibrato_depth_rad: float
    harmonics: tuple[Harmonic, ...]
    raw: dict[str, Any]

    @property
    def linear_gain(self) -> float:
        return 10.0 ** (self.gain_db / 20.0)


@dataclass(frozen=True)
class Syllable:
    text: str
    duration_samples: int


@dataclass(frozen=True)
class Line:
    text: str
    syllables: tuple[Syllable, ...]
    raw: dict[str, Any]

    @property
    def natural_samples(self) -> int:
        return sum(item.duration_samples for item in self.syllables)


@dataclass
class EntryPlan:
    entry_id: str
    slot_id: str
    locale: str
    role_name: str
    start_sample: int
    duration_samples: int
    order: int
    table_path: str
    line: Line | None = None
    voice: Voice | None = None
    issue: Issue | None = None
    input_hash: str = ""


@dataclass
class LoadedInput:
    config: BakeConfig
    config_raw: dict[str, Any]
    config_hash: str
    manifest_path: Path
    manifest_raw: dict[str, Any]
    locales: tuple[str, ...]
    roles: dict[str, Voice | None]
    roles_raw: dict[str, Any]
    entries: list[EntryPlan]
    issues: list[Issue] = field(default_factory=list)
    table_files: list[dict[str, Any]] = field(default_factory=list)
    project_hash: str = ""
    input_version: str = ""

    @property
    def timeline_samples(self) -> int:
        return max(
            (
                entry.start_sample + entry.duration_samples
                for entry in self.entries
                if entry.issue is None
            ),
            default=0,
        )

    @property
    def valid_entries(self) -> list[EntryPlan]:
        return [entry for entry in self.entries if entry.issue is None]

    def locale_signature(self, locale: str) -> str:
        parts: list[dict[str, Any]] = []
        for entry in self.entries:
            if entry.locale != locale:
                continue
            if entry.issue is None:
                parts.append(
                    {
                        "slot_id": entry.slot_id,
                        "state": "valid",
                        "input_hash": entry.input_hash,
                    }
                )
            else:
                parts.append(
                    {
                        "slot_id": entry.slot_id,
                        "state": "error",
                        "code": entry.issue.code,
                    }
                )
        return object_hash(
            {
                "baker_version": BAKER_VERSION,
                "config_hash": self.config_hash,
                "locale": locale,
                "entries": parts,
            }
        )


def _read_json(path: Path) -> tuple[Any, str]:
    try:
        raw_bytes = path.read_bytes()
    except FileNotFoundError as exc:
        raise InputFatal(
            "INPUT_FILE_MISSING",
            "input file does not exist",
            {"path": str(path)},
        ) from exc
    except OSError as exc:
        raise InputFatal(
            "INPUT_FILE_UNREADABLE",
            "input file could not be read",
            {"path": str(path), "error": str(exc)},
        ) from exc
    try:
        value = loads_json_strict(raw_bytes)
    except (UnicodeDecodeError, ValueError) as exc:
        raise InputFatal(
            "INPUT_JSON_INVALID",
            "input is not valid UTF-8 JSON",
            {"path": str(path), "error": str(exc)},
        ) from exc
    return value, sha256_bytes(raw_bytes)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _parse_voice(role_name: str, raw: Any) -> tuple[Voice | None, Issue | None]:
    context = {"role": role_name}
    if not isinstance(raw, dict):
        return None, Issue("VOICE_INVALID", "role definition must be an object", context)
    gain_db = _finite_number(raw.get("gain_db"))
    voice = raw.get("voice")
    if gain_db is None:
        return None, Issue("VOICE_INVALID", "role gain_db must be finite", context)
    if not isinstance(voice, dict):
        return None, Issue("VOICE_INVALID", "role voice must be an object", context)

    fundamental_hz = _finite_number(voice.get("fundamental_hz"))
    amplitude = _finite_number(voice.get("amplitude"))
    vibrato_hz = _finite_number(voice.get("vibrato_hz", 0.0))
    vibrato_depth = _finite_number(voice.get("vibrato_depth_rad", 0.0))
    if fundamental_hz is None or fundamental_hz <= 0:
        return None, Issue(
            "VOICE_INVALID", "fundamental_hz must be finite and > 0", context
        )
    if amplitude is None or amplitude <= 0:
        return None, Issue(
            "VOICE_INVALID", "amplitude must be finite and > 0", context
        )
    if vibrato_hz is None or vibrato_hz < 0:
        return None, Issue(
            "VOICE_INVALID", "vibrato_hz must be finite and >= 0", context
        )
    if vibrato_depth is None or vibrato_depth < 0:
        return None, Issue(
            "VOICE_INVALID",
            "vibrato_depth_rad must be finite and >= 0",
            context,
        )

    harmonics_raw = voice.get("harmonics")
    if not isinstance(harmonics_raw, list) or not harmonics_raw:
        return None, Issue(
            "VOICE_INVALID", "harmonics must be a non-empty list", context
        )
    harmonics: list[Harmonic] = []
    for index, item in enumerate(harmonics_raw):
        if not isinstance(item, dict):
            return None, Issue(
                "VOICE_INVALID",
                "harmonic must be an object",
                {**context, "harmonic_index": index},
            )
        multiple = _finite_number(item.get("multiple"))
        gain = _finite_number(item.get("gain"))
        phase = _finite_number(item.get("phase_deg", 0.0))
        if multiple is None or multiple <= 0 or gain is None or phase is None:
            return None, Issue(
                "VOICE_INVALID",
                "harmonic multiple/gain/phase_deg are invalid",
                {**context, "harmonic_index": index},
            )
        harmonics.append(Harmonic(multiple, gain, phase))
    if not any(abs(item.gain) > 0 for item in harmonics):
        return None, Issue(
            "VOICE_INVALID", "at least one harmonic gain must be non-zero", context
        )

    return (
        Voice(
            role_name=role_name,
            gain_db=gain_db,
            fundamental_hz=fundamental_hz,
            amplitude=amplitude,
            vibrato_hz=vibrato_hz,
            vibrato_depth_rad=vibrato_depth,
            harmonics=tuple(harmonics),
            raw=raw,
        ),
        None,
    )


def _parse_line(raw: Any) -> tuple[Line | None, Issue | None]:
    if not isinstance(raw, dict):
        return None, Issue("INVALID_LINE", "line must be an object")
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, Issue("EMPTY_TEXT", "line text is empty")
    syllables_raw = raw.get("syllables")
    if not isinstance(syllables_raw, list) or not syllables_raw:
        return None, Issue("INVALID_SYLLABLES", "syllables must be a non-empty list")
    syllables: list[Syllable] = []
    for index, item in enumerate(syllables_raw):
        if not isinstance(item, dict):
            return None, Issue(
                "INVALID_SYLLABLES",
                "syllable must be an object",
                {"syllable_index": index},
            )
        duration = item.get("duration_samples")
        if isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0:
            return None, Issue(
                "INVALID_SYLLABLES",
                "syllable duration_samples must be a positive integer",
                {"syllable_index": index},
            )
        syllable_text = item.get("text", "")
        if not isinstance(syllable_text, str):
            return None, Issue(
                "INVALID_SYLLABLES",
                "syllable text must be a string",
                {"syllable_index": index},
            )
        syllables.append(Syllable(syllable_text, duration))
    return Line(text=text, syllables=tuple(syllables), raw=raw), None


def _slot_error_entries(
    slot_id: str,
    role_name: str,
    start_sample: int,
    duration_samples: int,
    locales: tuple[str, ...],
    order: int,
    table_path: str,
    issue: Issue,
) -> list[EntryPlan]:
    return [
        EntryPlan(
            entry_id=f"{slot_id}/{locale}",
            slot_id=slot_id,
            locale=locale,
            role_name=role_name,
            start_sample=start_sample,
            duration_samples=duration_samples,
            order=order,
            table_path=table_path,
            issue=Issue(
                code=issue.code,
                message=issue.message,
                context={**issue.context, "locale": locale},
            ),
        )
        for locale in locales
    ]


def _load_table(
    *,
    root: Path,
    table_ref: str,
    locales: tuple[str, ...],
    roles: dict[str, Voice | None],
    config_hash: str,
    order_start: int,
    seen_slot_ids: set[str],
) -> tuple[list[EntryPlan], list[Issue], dict[str, Any]]:
    path = (root / table_ref).resolve()
    context = {"path": table_ref}
    try:
        raw_bytes = path.read_bytes()
    except FileNotFoundError:
        issue = Issue("INPUT_FILE_MISSING", "planning table does not exist", context)
        return [], [issue], {"path": table_ref, "state": "missing"}
    except OSError as exc:
        issue = Issue(
            "INPUT_FILE_UNREADABLE",
            "planning table could not be read",
            {**context, "error": str(exc)},
        )
        return [], [issue], {"path": table_ref, "state": "unreadable"}

    raw_hash = sha256_bytes(raw_bytes)
    try:
        table = loads_json_strict(raw_bytes)
    except (UnicodeDecodeError, ValueError) as exc:
        issue = Issue(
            "INPUT_JSON_INVALID",
            "planning table is not valid UTF-8 JSON",
            {**context, "error": str(exc)},
        )
        return [], [issue], {
            "path": table_ref,
            "state": "invalid_json",
            "raw_hash": raw_hash,
        }

    if isinstance(table, list):
        slots = table
        table_id = path.stem
    elif isinstance(table, dict):
        table_id = table.get("table_id", path.stem)
        slots = table.get("slots")
        if table.get("schema_version") != 1:
            issue = Issue(
                "TABLE_INVALID",
                "unsupported table schema_version",
                {**context, "schema_version": table.get("schema_version")},
            )
            return [], [issue], {
                "path": table_ref,
                "state": "invalid",
                "raw_hash": raw_hash,
            }
    else:
        issue = Issue(
            "TABLE_INVALID",
            "planning table root must be an object or array",
            context,
        )
        return [], [issue], {
            "path": table_ref,
            "state": "invalid",
            "raw_hash": raw_hash,
        }

    if not isinstance(slots, list):
        issue = Issue(
            "TABLE_INVALID", "planning table slots must be an array", context
        )
        return [], [issue], {
            "path": table_ref,
            "state": "invalid",
            "raw_hash": raw_hash,
        }

    entries: list[EntryPlan] = []
    issues: list[Issue] = []
    order = order_start
    for index, slot in enumerate(slots):
        slot_context = {
            "path": table_ref,
            "table_id": table_id,
            "slot_index": index,
        }
        if not isinstance(slot, dict):
            issue = Issue(
                "SLOT_INVALID", "slot must be an object", slot_context
            )
            issues.append(issue)
            order += 1
            continue
        slot_id = slot.get("slot_id")
        role_name = slot.get("role")
        start_sample = slot.get("start_sample")
        duration_samples = slot.get("duration_samples")
        if not isinstance(slot_id, str) or not slot_id.strip():
            issue = Issue("SLOT_INVALID", "slot_id must be a non-empty string", slot_context)
            issues.append(issue)
            order += 1
            continue
        if not isinstance(role_name, str) or not role_name:
            issue = Issue(
                "SLOT_INVALID",
                "role must be a non-empty string",
                {**slot_context, "slot_id": slot_id},
            )
            issues.append(issue)
            order += 1
            continue
        if (
            isinstance(start_sample, bool)
            or not isinstance(start_sample, int)
            or start_sample < 0
        ):
            issue = Issue(
                "SLOT_INVALID",
                "start_sample must be a non-negative integer",
                {**slot_context, "slot_id": slot_id},
            )
            issues.append(issue)
            order += 1
            continue
        if (
            isinstance(duration_samples, bool)
            or not isinstance(duration_samples, int)
            or duration_samples <= 0
        ):
            issue = Issue(
                "SLOT_INVALID",
                "duration_samples must be a positive integer",
                {**slot_context, "slot_id": slot_id},
            )
            issues.append(issue)
            order += 1
            continue
        if slot_id in seen_slot_ids:
            issue = Issue(
                "DUPLICATE_SLOT",
                "slot_id already exists",
                {**slot_context, "slot_id": slot_id},
            )
            entries.extend(
                _slot_error_entries(
                    slot_id,
                    role_name,
                    start_sample,
                    duration_samples,
                    locales,
                    order,
                    table_ref,
                    issue,
                )
            )
            issues.append(issue)
            order += 1
            continue
        seen_slot_ids.add(slot_id)

        lines = slot.get("lines")
        if not isinstance(lines, dict):
            issue = Issue(
                "SLOT_INVALID",
                "lines must be an object keyed by locale",
                {**slot_context, "slot_id": slot_id},
            )
            entries.extend(
                _slot_error_entries(
                    slot_id,
                    role_name,
                    start_sample,
                    duration_samples,
                    locales,
                    order,
                    table_ref,
                    issue,
                )
            )
            issues.append(issue)
            order += 1
            continue

        voice = roles.get(role_name)
        role_issue: Issue | None = None
        if role_name not in roles:
            role_issue = Issue(
                "VOICE_NOT_FOUND",
                "role is not defined in the project manifest",
                {**slot_context, "slot_id": slot_id, "role": role_name},
            )
        elif voice is None:
            role_issue = Issue(
                "VOICE_INVALID",
                "role definition failed validation",
                {**slot_context, "slot_id": slot_id, "role": role_name},
            )

        for locale in locales:
            line_raw = lines.get(locale)
            line: Line | None = None
            issue: Issue | None = role_issue
            if issue is None:
                if locale not in lines:
                    issue = Issue(
                        "MISSING_LINE",
                        "locale line is missing for this slot",
                        {
                            **slot_context,
                            "slot_id": slot_id,
                            "locale": locale,
                        },
                    )
                else:
                    line, line_issue = _parse_line(line_raw)
                    if line_issue is not None:
                        issue = Issue(
                            code=line_issue.code,
                            message=line_issue.message,
                            context={
                                **slot_context,
                                "slot_id": slot_id,
                                "locale": locale,
                                **line_issue.context,
                            },
                        )
            if issue is not None:
                issues.append(issue)

            entry_id = f"{slot_id}/{locale}"
            entry = EntryPlan(
                entry_id=entry_id,
                slot_id=slot_id,
                locale=locale,
                role_name=role_name,
                start_sample=start_sample,
                duration_samples=duration_samples,
                order=order,
                table_path=table_ref,
                line=line,
                voice=voice if issue is None else None,
                issue=issue,
            )
            if issue is None and line is not None and voice is not None:
                entry.input_hash = object_hash(
                    {
                        "baker_version": BAKER_VERSION,
                        "config_hash": config_hash,
                        "locale": locale,
                        "role_name": role_name,
                        "role": voice.raw,
                        "slot_id": slot_id,
                        "start_sample": start_sample,
                        "duration_samples": duration_samples,
                        "line": line.raw,
                    }
                )
            entries.append(entry)
        order += 1

    return entries, issues, {
        "path": table_ref,
        "state": "loaded",
        "raw_hash": raw_hash,
        "table_id": table_id,
        "slot_count": len(slots),
    }


def load_input(project_path: Path, config_path: Path) -> LoadedInput:
    config_raw, _ = _read_json(config_path)
    config = parse_config(config_raw)
    config_hash = object_hash(config_raw)

    manifest, manifest_raw_hash = _read_json(project_path)
    if not isinstance(manifest, dict):
        raise InputFatal("MANIFEST_INVALID", "project manifest root must be an object")
    if manifest.get("schema_version") != 1:
        raise InputFatal(
            "MANIFEST_INVALID",
            "unsupported project schema_version",
            {"schema_version": manifest.get("schema_version")},
        )
    locales_raw = manifest.get("locales")
    if (
        not isinstance(locales_raw, list)
        or not locales_raw
        or any(not isinstance(item, str) or not item for item in locales_raw)
        or len(set(locales_raw)) != len(locales_raw)
    ):
        raise InputFatal(
            "MANIFEST_INVALID",
            "locales must be a non-empty array of unique strings",
        )
    locales = tuple(locales_raw)
    roles_raw = manifest.get("roles")
    if not isinstance(roles_raw, dict) or not roles_raw:
        raise InputFatal(
            "MANIFEST_INVALID", "roles must be a non-empty object"
        )
    tables_raw = manifest.get("tables")
    if (
        not isinstance(tables_raw, list)
        or not tables_raw
        or any(not isinstance(item, str) or not item for item in tables_raw)
    ):
        raise InputFatal(
            "MANIFEST_INVALID",
            "tables must be a non-empty array of relative paths",
        )

    roles: dict[str, Voice | None] = {}
    issues: list[Issue] = []
    for role_name, role_raw in roles_raw.items():
        if not isinstance(role_name, str) or not role_name:
            issues.append(Issue("VOICE_INVALID", "role names must be strings"))
            continue
        voice, issue = _parse_voice(role_name, role_raw)
        roles[role_name] = voice
        if issue is not None:
            issues.append(issue)

    entries: list[EntryPlan] = []
    table_files: list[dict[str, Any]] = []
    seen_slot_ids: set[str] = set()
    order = 0
    for table_ref in tables_raw:
        table_entries, table_issues, table_meta = _load_table(
            root=project_path.parent,
            table_ref=table_ref,
            locales=locales,
            roles=roles,
            config_hash=config_hash,
            order_start=order,
            seen_slot_ids=seen_slot_ids,
        )
        entries.extend(table_entries)
        issues.extend(table_issues)
        table_files.append(table_meta)
        order += max(1, len(table_entries) // max(1, len(locales)))

    project_hash = object_hash(
        {
            "manifest_raw_hash": manifest_raw_hash,
            "manifest": manifest,
            "tables": table_files,
        }
    )
    input_version = object_hash(
        {
            "baker_version": BAKER_VERSION,
            "config_hash": config_hash,
            "project_hash": project_hash,
        }
    )
    result = LoadedInput(
        config=config,
        config_raw=config_raw,
        config_hash=config_hash,
        manifest_path=project_path,
        manifest_raw=manifest,
        locales=locales,
        roles=roles,
        roles_raw=roles_raw,
        entries=entries,
        issues=issues,
        table_files=table_files,
        project_hash=project_hash,
        input_version=input_version,
    )

    # Assign stable per-slot order after table loading; table paths can otherwise
    # make order increments depend on error counts.
    for index, entry in enumerate(sorted(result.entries, key=lambda item: (item.order, item.slot_id, item.locale))):
        entry.order = index
    return result
