"""Repair manifest parsing and complete semantic validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import make_error
from .wavio import WavData, WavFormatError, read_wav


SAMPLE_RATE = 48000
OPERATION_TYPES = ("replace", "splice", "gain")
_MISSING = object()
_INVALID = object()


@dataclass
class OperationSpec:
    position: int
    declared_index: int | None
    operation_type: str | None
    start: int | None = None
    end: int | None = None
    fade_ms: float = 0.0
    fade_samples: int = 0
    frequency_hz: float | None = None
    phase_radians: float | None = None
    amplitude: float | None = None
    reference_file: str | None = None
    reference_start: int | None = None
    reference_end: int | None = None
    reference_path: Path | None = None
    reference_audio: WavData | None = None
    gain: float | None = None

    @property
    def interval_valid(self) -> bool:
        return self.start is not None and self.end is not None and self.start < self.end


@dataclass
class Manifest:
    quantization_bits: int | None
    sample_rate: int | None
    operations: list[OperationSpec] = field(default_factory=list)


class _DuplicateKeyError(ValueError):
    pass


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _add(
    errors: list[dict[str, Any]],
    category: str,
    message: str,
    *,
    operation_index: int | None = None,
    field_name: str | None = None,
    related: list[int] | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    errors.append(
        make_error(
            category,
            message,
            operation_index=operation_index,
            field=field_name,
            related_operation_indices=related,
            details=details,
        )
    )


def _alias_value(
    obj: dict[str, Any],
    canonical: str,
    aliases: tuple[str, ...],
    *,
    errors: list[dict[str, Any]],
    operation_index: int,
) -> Any:
    present = [(key, obj[key]) for key in (canonical, *aliases) if key in obj]
    if not present:
        return _MISSING
    first_value = present[0][1]
    for _, value in present[1:]:
        if value != first_value:
            _add(
                errors,
                "invalid_field",
                f"aliases for '{canonical}' contain conflicting values",
                operation_index=operation_index,
                field_name=canonical,
            )
            return _INVALID
    return first_value


def _require_number(
    value: Any,
    *,
    errors: list[dict[str, Any]],
    operation_index: int,
    field_name: str,
    missing_category: str = "missing_field",
) -> float | None:
    if value is _MISSING:
        _add(
            errors,
            missing_category,
            f"missing required field '{field_name}'",
            operation_index=operation_index,
            field_name=field_name,
        )
        return None
    if value is _INVALID:
        return None
    if value is None:
        _add(
            errors,
            "invalid_field",
            f"field '{field_name}' must be a finite number",
            operation_index=operation_index,
            field_name=field_name,
        )
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _add(
            errors,
            "invalid_field",
            f"field '{field_name}' must be a finite number",
            operation_index=operation_index,
            field_name=field_name,
        )
        return None
    number = float(value)
    if not math.isfinite(number):
        _add(
            errors,
            "invalid_field",
            f"field '{field_name}' must be finite",
            operation_index=operation_index,
            field_name=field_name,
        )
        return None
    return number


def _require_int(
    value: Any,
    *,
    errors: list[dict[str, Any]],
    operation_index: int,
    field_name: str,
) -> int | None:
    if value is _MISSING:
        _add(
            errors,
            "missing_field",
            f"missing required field '{field_name}'",
            operation_index=operation_index,
            field_name=field_name,
        )
        return None
    if value is _INVALID:
        return None
    if value is None:
        _add(
            errors,
            "invalid_field",
            f"field '{field_name}' must be an integer sample position",
            operation_index=operation_index,
            field_name=field_name,
        )
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        _add(
            errors,
            "invalid_field",
            f"field '{field_name}' must be an integer sample position",
            operation_index=operation_index,
            field_name=field_name,
        )
        return None
    return value


def _parse_quantization(
    obj: dict[str, Any], errors: list[dict[str, Any]]
) -> int | None:
    keys = (
        "quantization_bits",
        "bit_depth",
        "output_bit_depth",
        "quantization",
        "output_quantization",
    )
    present = [(key, obj[key]) for key in keys if key in obj]
    if not present:
        _add(
            errors,
            "missing_field",
            "missing required field 'quantization_bits'",
            field_name="quantization_bits",
        )
        return None

    normalized: list[int] = []
    for _, value in present:
        bits: int | None = None
        if isinstance(value, bool):
            bits = None
        elif isinstance(value, int):
            bits = value
        elif isinstance(value, str):
            token = value.strip().lower().replace("_", "-")
            if token in ("16", "16-bit", "16bit", "pcm-s16le"):
                bits = 16
            elif token in ("24", "24-bit", "24bit", "pcm-s24le"):
                bits = 24
        if bits not in (16, 24):
            _add(
                errors,
                "invalid_quantization",
                "quantization_bits must be 16 or 24",
                field_name="quantization_bits",
                details={"value": str(value)},
            )
            return None
        normalized.append(bits)

    if len(set(normalized)) != 1:
        _add(
            errors,
            "invalid_quantization",
            "conflicting quantization fields were supplied",
            field_name="quantization_bits",
        )
        return None
    return normalized[0]


def _parse_operation(
    raw: Any,
    position: int,
    errors: list[dict[str, Any]],
) -> OperationSpec:
    if not isinstance(raw, dict):
        _add(
            errors,
            "invalid_operation",
            "operation must be a JSON object",
            operation_index=position,
        )
        return OperationSpec(
            position=position,
            declared_index=None,
            operation_type=None,
        )

    index_value = _alias_value(
        raw,
        "index",
        ("operation_index", "sequence"),
        errors=errors,
        operation_index=position,
    )
    declared_index = _require_int(
        index_value,
        errors=errors,
        operation_index=position,
        field_name="index",
    )
    if declared_index is not None and declared_index != position:
        _add(
            errors,
            "operation_index_mismatch",
            f"declared index {declared_index} does not match position {position}",
            operation_index=position,
            field_name="index",
            details={"declared_index": declared_index},
        )

    operation_type: str | None = None
    type_value = _alias_value(
        raw,
        "type",
        ("operation",),
        errors=errors,
        operation_index=position,
    )
    if type_value is _MISSING:
        _add(
            errors,
            "missing_field",
            "operation is missing field 'type'",
            operation_index=position,
            field_name="type",
        )
    elif type_value is _INVALID:
        pass
    elif not isinstance(type_value, str):
        _add(
            errors,
            "invalid_field",
            "operation type must be a string",
            operation_index=position,
            field_name="type",
        )
    else:
        operation_type = type_value
        if operation_type not in OPERATION_TYPES:
            _add(
                errors,
                "unknown_operation_type",
                f"unknown operation type '{operation_type}'",
                operation_index=position,
                field_name="type",
                details={"type": operation_type},
            )

    spec = OperationSpec(
        position=position,
        declared_index=declared_index,
        operation_type=operation_type,
    )

    start_value = _alias_value(
        raw,
        "start_sample",
        ("start", "target_start", "start_index"),
        errors=errors,
        operation_index=position,
    )
    end_value = _alias_value(
        raw,
        "end_sample",
        ("end", "target_end", "end_index"),
        errors=errors,
        operation_index=position,
    )
    start = _require_int(
        start_value,
        errors=errors,
        operation_index=position,
        field_name="start_sample",
    )
    end = _require_int(
        end_value,
        errors=errors,
        operation_index=position,
        field_name="end_sample",
    )
    if start is not None and end is not None:
        if start < 0 or end <= start:
            _add(
                errors,
                "invalid_interval",
                "target interval must satisfy 0 <= start_sample < end_sample",
                operation_index=position,
                field_name="end_sample",
                details={"start_sample": start, "end_sample": end},
            )
        else:
            spec.start = start
            spec.end = end

    if operation_type in ("replace", "splice"):
        fade_value = _alias_value(
            raw,
            "fade_ms",
            ("crossfade_ms", "fade_duration_ms"),
            errors=errors,
            operation_index=position,
        )
        fade_ms = _require_number(
            fade_value,
            errors=errors,
            operation_index=position,
            field_name="fade_ms",
        )
    elif operation_type == "gain":
        fade_value = _alias_value(
            raw,
            "fade_ms",
            ("crossfade_ms", "fade_duration_ms"),
            errors=errors,
            operation_index=position,
        )
        if fade_value is _MISSING:
            fade_value = 0.0
        fade_ms = _require_number(
            fade_value,
            errors=errors,
            operation_index=position,
            field_name="fade_ms",
        )
    else:
        fade_value = raw.get("fade_ms", raw.get("crossfade_ms"))
        fade_ms = (
            _require_number(
                fade_value,
                errors=errors,
                operation_index=position,
                field_name="fade_ms",
            )
            if fade_value is not None
            else 0.0
        )

    if fade_ms is not None:
        if fade_ms < 0.0:
            _add(
                errors,
                "invalid_fade_length",
                "fade_ms must be non-negative",
                operation_index=position,
                field_name="fade_ms",
                details={"fade_ms": fade_ms},
            )
        elif operation_type == "gain" and fade_ms > 0.0:
            _add(
                errors,
                "invalid_fade_length",
                "gain operations do not support crossfade",
                operation_index=position,
                field_name="fade_ms",
                details={"fade_ms": fade_ms},
            )
        else:
            fade_samples = int(math.floor(fade_ms * SAMPLE_RATE / 1000.0 + 0.5))
            if fade_ms > 0.0 and fade_samples < 2:
                _add(
                    errors,
                    "invalid_fade_length",
                    "positive fade length rounds to fewer than two samples",
                    operation_index=position,
                    field_name="fade_ms",
                    details={"fade_samples": fade_samples},
                )
            elif spec.interval_valid and fade_samples * 2 > (spec.end - spec.start):
                _add(
                    errors,
                    "invalid_fade_length",
                    "left and right crossfades do not fit inside the target interval",
                    operation_index=position,
                    field_name="fade_ms",
                    details={
                        "fade_samples": fade_samples,
                        "interval_samples": spec.end - spec.start,
                    },
                )
            else:
                spec.fade_ms = fade_ms
                spec.fade_samples = fade_samples

    if operation_type == "replace":
        frequency_value = _alias_value(
            raw,
            "frequency_hz",
            ("frequency", "freq_hz"),
            errors=errors,
            operation_index=position,
        )
        phase_value = _alias_value(
            raw,
            "phase_radians",
            ("phase", "start_phase_radians", "phase_rad"),
            errors=errors,
            operation_index=position,
        )
        amplitude_value = raw.get("amplitude", 1.0)
        frequency = _require_number(
            frequency_value,
            errors=errors,
            operation_index=position,
            field_name="frequency_hz",
        )
        phase = _require_number(
            phase_value,
            errors=errors,
            operation_index=position,
            field_name="phase_radians",
        )
        amplitude = _require_number(
            amplitude_value,
            errors=errors,
            operation_index=position,
            field_name="amplitude",
        )
        if frequency is not None and frequency < 0.0:
            _add(
                errors,
                "invalid_field",
                "frequency_hz must be non-negative",
                operation_index=position,
                field_name="frequency_hz",
            )
        if amplitude is not None and amplitude < 0.0:
            _add(
                errors,
                "invalid_field",
                "amplitude must be non-negative",
                operation_index=position,
                field_name="amplitude",
            )
        spec.frequency_hz = frequency
        spec.phase_radians = phase
        spec.amplitude = amplitude

    elif operation_type == "splice":
        reference_value = _alias_value(
            raw,
            "reference_file",
            ("reference", "reference_path", "ref_file"),
            errors=errors,
            operation_index=position,
        )
        reference_start_value = _alias_value(
            raw,
            "reference_start",
            ("reference_start_sample", "source_start", "ref_start"),
            errors=errors,
            operation_index=position,
        )
        reference_end_value = _alias_value(
            raw,
            "reference_end",
            ("reference_end_sample", "source_end", "ref_end"),
            errors=errors,
            operation_index=position,
        )
        reference_file: str | None = None
        if reference_value is _MISSING:
            _add(
                errors,
                "missing_field",
                "missing required field 'reference_file'",
                operation_index=position,
                field_name="reference_file",
            )
            reference_file = None
        elif reference_value is _INVALID:
            reference_file = None
        elif not isinstance(reference_value, str) or not reference_value.strip():
            _add(
                errors,
                "invalid_field",
                "reference_file must be a non-empty string",
                operation_index=position,
                field_name="reference_file",
            )
        else:
            reference_file = reference_value
        reference_start = _require_int(
            reference_start_value,
            errors=errors,
            operation_index=position,
            field_name="reference_start",
        )
        reference_end = _require_int(
            reference_end_value,
            errors=errors,
            operation_index=position,
            field_name="reference_end",
        )
        if reference_start is not None and reference_end is not None:
            if reference_start < 0 or reference_end <= reference_start:
                _add(
                    errors,
                    "reference_out_of_bounds",
                    "reference interval must satisfy 0 <= start < end",
                    operation_index=position,
                    field_name="reference_end",
                    details={
                        "reference_start": reference_start,
                        "reference_end": reference_end,
                    },
                )
            else:
                spec.reference_start = reference_start
                spec.reference_end = reference_end
                if (
                    spec.interval_valid
                    and (spec.end - spec.start) != (reference_end - reference_start)
                ):
                    _add(
                        errors,
                        "reference_length_mismatch",
                        "reference interval length must equal target interval length",
                        operation_index=position,
                        details={
                            "target_samples": spec.end - spec.start,
                            "reference_samples": reference_end - reference_start,
                        },
                    )
        spec.reference_file = reference_file

    elif operation_type == "gain":
        gain_value = _alias_value(
            raw,
            "gain",
            ("linear_gain", "factor"),
            errors=errors,
            operation_index=position,
        )
        gain = _require_number(
            gain_value,
            errors=errors,
            operation_index=position,
            field_name="gain",
        )
        if gain is not None and gain < 0.0:
            _add(
                errors,
                "negative_gain",
                "gain must be non-negative",
                operation_index=position,
                field_name="gain",
                details={"gain": gain},
            )
        spec.gain = gain

    return spec


def _check_overlaps(
    operations: list[OperationSpec], errors: list[dict[str, Any]]
) -> None:
    valid = [op for op in operations if op.interval_valid]
    relations: dict[int, set[int]] = {}
    for left_index, left in enumerate(valid):
        for right in valid[left_index + 1 :]:
            assert left.start is not None and left.end is not None
            assert right.start is not None and right.end is not None
            if left.start < right.end and right.start < left.end:
                relations.setdefault(left.position, set()).add(right.position)
                relations.setdefault(right.position, set()).add(left.position)
    for position in sorted(relations):
        _add(
            errors,
            "overlapping_intervals",
            "target interval overlaps another operation",
            operation_index=position,
            related=relations[position],
        )


def _check_references(
    operations: list[OperationSpec],
    manifest_dir: Path,
    errors: list[dict[str, Any]],
) -> None:
    cache: dict[Path, WavData | WavFormatError | FileNotFoundError] = {}
    for spec in operations:
        if spec.operation_type != "splice" or spec.reference_file is None:
            continue
        reference_path = Path(spec.reference_file)
        if not reference_path.is_absolute():
            reference_path = manifest_dir / reference_path
        reference_path = reference_path.resolve()
        spec.reference_path = reference_path

        cached = cache.get(reference_path)
        if cached is None:
            try:
                cached = read_wav(reference_path)
            except FileNotFoundError as exc:
                cached = exc
            except (OSError, WavFormatError) as exc:
                cached = WavFormatError(str(exc))
            cache[reference_path] = cached

        if isinstance(cached, FileNotFoundError):
            _add(
                errors,
                "missing_reference",
                f"reference WAV not found: {spec.reference_file}",
                operation_index=spec.position,
                field_name="reference_file",
                details={"reference_file": spec.reference_file},
            )
            continue
        if isinstance(cached, WavFormatError):
            _add(
                errors,
                "reference_format_mismatch",
                f"reference WAV is unreadable or not PCM: {cached}",
                operation_index=spec.position,
                field_name="reference_file",
                details={"reference_file": spec.reference_file},
            )
            continue
        format_valid = cached.sample_rate == SAMPLE_RATE and cached.channels == 1
        if not format_valid:
            _add(
                errors,
                "reference_format_mismatch",
                "reference WAV must be 48000 Hz mono",
                operation_index=spec.position,
                field_name="reference_file",
                details={
                    "reference_file": spec.reference_file,
                    "actual_sample_rate": cached.sample_rate,
                    "actual_channels": cached.channels,
                },
            )
        else:
            spec.reference_audio = cached
        if spec.reference_start is not None and spec.reference_end is not None:
            if spec.reference_start >= cached.sample_count or spec.reference_end > cached.sample_count:
                _add(
                    errors,
                    "reference_out_of_bounds",
                    "reference interval exceeds reference WAV length",
                    operation_index=spec.position,
                    field_name="reference_end",
                    details={
                        "reference_start": spec.reference_start,
                        "reference_end": spec.reference_end,
                        "reference_sample_count": cached.sample_count,
                    },
                )


def parse_manifest(
    path: str | Path,
    source_sample_count: int | None,
    errors: list[dict[str, Any]],
) -> Manifest:
    manifest_path = Path(path)
    try:
        raw_bytes = manifest_path.read_bytes()
    except OSError as exc:
        _add(
            errors,
            "manifest_read_error",
            f"cannot read repair manifest: {exc}",
            details={"manifest": str(path)},
        )
        return Manifest(quantization_bits=None, sample_rate=None)

    try:
        text = raw_bytes.decode("utf-8-sig")
        obj = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, _DuplicateKeyError) as exc:
        _add(
            errors,
            "invalid_json",
            f"invalid repair manifest JSON: {exc}",
        )
        return Manifest(quantization_bits=None, sample_rate=None)

    if not isinstance(obj, dict):
        _add(errors, "invalid_manifest", "repair manifest must be a JSON object")
        return Manifest(quantization_bits=None, sample_rate=None)

    quantization_bits = _parse_quantization(obj, errors)
    sample_rate: int | None = None
    if "sample_rate" in obj:
        value = obj["sample_rate"]
        if isinstance(value, bool) or not isinstance(value, int):
            _add(
                errors,
                "invalid_field",
                "manifest sample_rate must be an integer",
                field_name="sample_rate",
            )
        else:
            sample_rate = value
            if sample_rate != SAMPLE_RATE:
                _add(
                    errors,
                    "manifest_sample_rate_mismatch",
                    "manifest sample_rate must be 48000",
                    field_name="sample_rate",
                    details={"actual_sample_rate": sample_rate},
                )

    raw_operations = obj.get("operations")
    operations: list[OperationSpec] = []
    if raw_operations is None:
        _add(
            errors,
            "missing_field",
            "manifest is missing required array 'operations'",
            field_name="operations",
        )
    elif not isinstance(raw_operations, list):
        _add(
            errors,
            "invalid_field",
            "manifest field 'operations' must be an array",
            field_name="operations",
        )
    else:
        for position, raw_operation in enumerate(raw_operations):
            operations.append(_parse_operation(raw_operation, position, errors))

    _check_overlaps(operations, errors)

    if source_sample_count is not None:
        for spec in operations:
            if not spec.interval_valid:
                continue
            assert spec.end is not None
            if spec.end > source_sample_count:
                _add(
                    errors,
                    "interval_out_of_bounds",
                    "target interval exceeds input audio sample count",
                    operation_index=spec.position,
                    field_name="end_sample",
                    details={
                        "start_sample": spec.start,
                        "end_sample": spec.end,
                        "source_sample_count": source_sample_count,
                    },
                )

    _check_references(operations, manifest_path.resolve().parent, errors)
    return Manifest(
        quantization_bits=quantization_bits,
        sample_rate=sample_rate,
        operations=operations,
    )

