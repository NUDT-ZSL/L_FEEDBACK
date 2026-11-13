"""Function-boundary manifest validation and name-based alignment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .binary import PeImage
from .errors import AuditError, bounds_error, manifest_error


@dataclass(frozen=True)
class FunctionRegion:
    name: str
    rva: int
    length: int

    @property
    def end_rva(self) -> int:
        return self.rva + self.length


@dataclass
class Manifest:
    version: str
    regions: dict[str, FunctionRegion]
    entry_points: list[str]
    raw: dict[str, Any]

    @property
    def names(self) -> list[str]:
        return sorted(self.regions)


@dataclass(frozen=True)
class Alignment:
    before: Manifest
    after: Manifest
    common: tuple[str, ...]
    removed: tuple[str, ...]
    added: tuple[str, ...]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "common": list(self.common),
            "removed": list(self.removed),
            "added": list(self.added),
        }


def load_json(path: str | Path, description: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as stream:
            value = json.load(stream)
    except OSError as exc:
        raise AuditError("INPUT_READ_ERROR", f"{description}: {exc}", 2) from exc
    except json.JSONDecodeError as exc:
        raise manifest_error(
            f"{description} is not valid JSON",
            line=exc.lineno,
            column=exc.colno,
        ) from exc
    if not isinstance(value, dict):
        raise manifest_error(f"{description} root must be a JSON object")
    return value


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise manifest_error(f"{field} must be an integer")
    return value


def parse_manifest(
    document: Mapping[str, Any],
    *,
    image: PeImage,
    version: str,
) -> Manifest:
    if not isinstance(document, Mapping):
        raise manifest_error("manifest root must be an object")
    raw_functions = document.get("functions")
    if not isinstance(raw_functions, list) or not raw_functions:
        raise manifest_error("functions must be a non-empty list")

    regions: dict[str, FunctionRegion] = {}
    for index, item in enumerate(raw_functions):
        if not isinstance(item, Mapping):
            raise manifest_error(f"functions[{index}] must be an object")
        if "name" not in item:
            raise manifest_error("function entry is missing name", index=index)
        if "rva" not in item:
            raise manifest_error("function entry is missing rva", index=index)
        if "length" not in item:
            raise manifest_error("function entry is missing length", index=index)
        name = item["name"]
        if not isinstance(name, str) or not name:
            raise manifest_error("function name must be a non-empty string", index=index)
        if len(name.encode("utf-8")) > 255:
            raise manifest_error("function name is unreasonably long", name=name)
        rva = _require_int(item["rva"], f"functions[{index}].rva")
        length = _require_int(item["length"], f"functions[{index}].length")
        if rva < 0 or length <= 0:
            raise bounds_error(
                "function interval has a negative RVA or non-positive length",
                function=name,
                rva=rva,
                length=length,
            )
        if name in regions:
            raise manifest_error("duplicate function name", function=name)
        if image.interval_bytes(rva, length) is None:
            raise bounds_error(
                "function interval is outside mapped section data",
                function=name,
                rva=rva,
                length=length,
            )
        regions[name] = FunctionRegion(name, rva, length)

    ordered = sorted(regions.values(), key=lambda region: (region.rva, region.name))
    previous: FunctionRegion | None = None
    for region in ordered:
        if previous is not None and region.rva < previous.end_rva:
            raise manifest_error(
                "function intervals overlap",
                first=previous.name,
                second=region.name,
                first_rva=previous.rva,
                second_rva=region.rva,
            )
        previous = region

    raw_entries = document.get("entry_points", [])
    if raw_entries is None:
        raw_entries = []
    if not isinstance(raw_entries, list) or not all(
        isinstance(entry, str) and entry for entry in raw_entries
    ):
        raise manifest_error("entry_points must be a list of non-empty strings")

    return Manifest(
        version=version,
        regions=regions,
        entry_points=sorted(set(raw_entries)),
        raw=dict(document),
    )


def align_manifests(before: Manifest, after: Manifest) -> Alignment:
    before_names = set(before.regions)
    after_names = set(after.regions)
    return Alignment(
        before=before,
        after=after,
        common=tuple(sorted(before_names & after_names)),
        removed=tuple(sorted(before_names - after_names)),
        added=tuple(sorted(after_names - before_names)),
    )


def region_bytes(image: PeImage, region: FunctionRegion) -> bytes:
    data = image.interval_bytes(region.rva, region.length)
    if data is None:
        raise bounds_error(
            "function interval is outside mapped section data",
            function=region.name,
            rva=region.rva,
            length=region.length,
        )
    return data
