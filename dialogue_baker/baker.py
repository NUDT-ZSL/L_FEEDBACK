"""Baking pipeline: plan audio, mix, pack, index, and report."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    BAKER_VERSION,
    CHANNELS,
    INDEX_SCHEMA_VERSION,
    PACK_FORMAT_VERSION,
    SAMPLE_FORMAT,
    SAMPLE_RATE,
)
from .errors import BakeFailure, Issue
from .input import EntryPlan, LoadedInput, Syllable
from .pack import (
    ChunkPayload,
    build_pack,
    ensure_pack_file,
    parse_header,
    validate_chunk,
    verify_pack_file,
)
from .synth import (
    bounded_stretch,
    dbfs,
    mix_stats,
    pcm16_bytes,
    pcm16_from_bytes,
    place_and_mix_pcm,
    quantize_pcm16,
    synthesize_line,
)
from .util import (
    alignment_error_samples,
    loads_json_strict,
    object_hash,
    samples_to_seconds,
    seconds_text,
    sha256_bytes,
    write_json_atomic,
)

EXIT_OK = 0
EXIT_DEGRADED = 2
EXIT_FATAL = 3


@dataclass
class OldResource:
    index: dict[str, Any]
    pack_path: Path
    pack_data: bytes
    entries_by_id: dict[str, dict[str, Any]]
    chunks_by_entry: dict[str, dict[str, Any]]
    warnings: list[Issue] = field(default_factory=list)


@dataclass
class RenderedEntry:
    plan: EntryPlan
    pcm: np.ndarray
    skipped: bool
    stretch: dict[str, Any]
    subtitle: dict[str, Any]
    loudness: dict[str, Any]
    alignment: dict[str, Any]
    float_audio: np.ndarray | None = None


@dataclass
class LocaleMix:
    locale: str
    reused: bool
    scale: float
    triggered: bool
    pre_peak: float
    stats: dict[str, Any]
    mix: np.ndarray


@dataclass
class BakeResult:
    index: dict[str, Any]
    report: dict[str, Any]
    pack_path: Path
    exit_code: int


def _read_old_resource(output_dir: Path) -> tuple[OldResource | None, list[Issue]]:
    warnings: list[Issue] = []
    index_path = output_dir / "index.json"
    if not index_path.is_file():
        return None, warnings
    try:
        index = loads_json_strict(index_path.read_bytes())
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        warnings.append(
            Issue(
                "OLD_INDEX_INVALID",
                "previous index is unreadable; a full rebuild is required",
                {"path": str(index_path), "error": str(exc)},
                severity="warning",
            )
        )
        return None, warnings
    if not isinstance(index, dict):
        warnings.append(
            Issue(
                "OLD_INDEX_INVALID",
                "previous index root is not an object",
                {"path": str(index_path)},
                severity="warning",
            )
        )
        return None, warnings
    if index.get("schema_version") != INDEX_SCHEMA_VERSION:
        warnings.append(
            Issue(
                "OLD_INDEX_VERSION_UNSUPPORTED",
                "previous index schema is unsupported",
                {
                    "schema_version": index.get("schema_version"),
                    "expected": INDEX_SCHEMA_VERSION,
                },
                severity="warning",
            )
        )
        return None, warnings
    core_for_hash = dict(index)
    recorded_version = core_for_hash.pop("resource_version", None)
    core_pack = core_for_hash.get("pack")
    if isinstance(core_pack, dict):
        core_pack = dict(core_pack)
        core_pack.pop("file", None)
        core_for_hash["pack"] = core_pack
    if not isinstance(recorded_version, str) or object_hash(core_for_hash) != recorded_version:
        warnings.append(
            Issue(
                "OLD_INDEX_INVALID",
                "previous index content does not match its resource version",
                severity="warning",
            )
        )
        return None, warnings

    pack_meta = index.get("pack")
    if not isinstance(pack_meta, dict):
        warnings.append(
            Issue(
                "OLD_PACK_INVALID",
                "previous index has no pack metadata",
                severity="warning",
            )
        )
        return None, warnings
    filename = pack_meta.get("file")
    expected_size = pack_meta.get("size")
    expected_hash = pack_meta.get("sha256")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not isinstance(expected_size, int)
        or not isinstance(expected_hash, str)
    ):
        warnings.append(
            Issue(
                "OLD_PACK_INVALID",
                "previous pack metadata is malformed",
                severity="warning",
            )
        )
        return None, warnings
    pack_path = output_dir / filename
    header: dict[str, int | str]
    try:
        pack_data = verify_pack_file(pack_path, expected_hash, expected_size)
        header = parse_header(pack_data)
        if header["sample_format"] != SAMPLE_FORMAT:
            raise ValueError("sample format mismatch")
    except (OSError, ValueError) as exc:
        warnings.append(
            Issue(
                "OLD_PACK_INVALID",
                "previous pack is incomplete or corrupt; a full rebuild is required",
                {"path": str(pack_path), "error": str(exc)},
                severity="warning",
            )
        )
        return None, warnings

    chunk_table = index.get("chunk_table")
    entries = index.get("entries")
    if not isinstance(chunk_table, list) or not isinstance(entries, list):
        warnings.append(
            Issue(
                "OLD_INDEX_INVALID",
                "previous index lacks chunk_table or entries",
                severity="warning",
            )
        )
        return None, warnings
    if header["chunk_count"] != len(chunk_table):
        warnings.append(
            Issue(
                "OLD_PACK_INVALID",
                "previous pack header chunk count does not match the index",
                severity="warning",
            )
        )
        return None, warnings
    timeline_meta = index.get("timeline")
    if (
        not isinstance(timeline_meta, dict)
        or header["timeline_samples"] != timeline_meta.get("timeline_end_sample")
    ):
        warnings.append(
            Issue(
                "OLD_INDEX_INVALID",
                "previous pack timeline does not match the index",
                severity="warning",
            )
        )
        return None, warnings
    chunks_by_entry: dict[str, dict[str, Any]] = {}
    for chunk in chunk_table:
        if not isinstance(chunk, dict) or not isinstance(chunk.get("entry_id"), str):
            warnings.append(
                Issue(
                    "OLD_INDEX_INVALID",
                    "previous chunk table is malformed",
                    severity="warning",
                )
            )
            return None, warnings
        chunks_by_entry[chunk["entry_id"]] = chunk
    entries_by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("entry_id"), str):
            warnings.append(
                Issue(
                    "OLD_INDEX_INVALID",
                    "previous entry list is malformed",
                    severity="warning",
                )
            )
            return None, warnings
        entries_by_id[entry["entry_id"]] = entry

    return (
        OldResource(
            index=index,
            pack_path=pack_path,
            pack_data=pack_data,
            entries_by_id=entries_by_id,
            chunks_by_entry=chunks_by_entry,
            warnings=warnings,
        ),
        warnings,
    )


def _prepare_stretch_checks(loaded: LoadedInput) -> None:
    for entry in loaded.entries:
        if entry.issue is not None or entry.line is None:
            continue
        natural = entry.line.natural_samples
        ratio = entry.duration_samples / natural
        if (
            ratio < loaded.config.stretch_min_ratio - 1e-12
            or ratio > loaded.config.stretch_max_ratio + 1e-12
        ):
            issue = Issue(
                "STRETCH_LIMIT_EXCEEDED",
                "required duration ratio is outside configured bounds",
                {
                    "slot_id": entry.slot_id,
                    "locale": entry.locale,
                    "table": entry.table_path,
                    "natural_samples": natural,
                    "target_samples": entry.duration_samples,
                    "ratio": round(ratio, 12),
                    "min_ratio": loaded.config.stretch_min_ratio,
                    "max_ratio": loaded.config.stretch_max_ratio,
                },
            )
            entry.issue = issue
            loaded.issues.append(issue)


def _timeline_dict(
    start_sample: int,
    end_sample: int,
    sample_rate: int,
) -> dict[str, Any]:
    start_seconds = samples_to_seconds(start_sample, sample_rate)
    end_seconds = samples_to_seconds(end_sample, sample_rate)
    return {
        "start_sample": int(start_sample),
        "end_sample": int(end_sample),
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "start_seconds_text": seconds_text(start_sample, sample_rate),
        "end_seconds_text": seconds_text(end_sample, sample_rate),
        "duration_samples": int(end_sample - start_sample),
    }


def _alignment_dict(start_sample: int, end_sample: int, sample_rate: int) -> dict[str, Any]:
    start_seconds = samples_to_seconds(start_sample, sample_rate)
    end_seconds = samples_to_seconds(end_sample, sample_rate)
    start_error = alignment_error_samples(start_sample, start_seconds, sample_rate)
    end_error = alignment_error_samples(end_sample, end_seconds, sample_rate)
    return {
        "start_error_samples": start_error,
        "end_error_samples": end_error,
        "max_error_samples": max(start_error, end_error),
    }


def _subtitle_metadata(
    entry: EntryPlan,
    line_text: str,
    syllables: tuple[Syllable, ...],
    target_samples: int,
    sample_rate: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_total = sum(item.duration_samples for item in syllables)
    mapped: list[dict[str, Any]] = []
    cumulative = 0
    for syllable in syllables:
        start_offset = (cumulative * target_samples + source_total // 2) // source_total
        cumulative += syllable.duration_samples
        end_offset = (cumulative * target_samples + source_total // 2) // source_total
        start = entry.start_sample + start_offset
        end = entry.start_sample + end_offset
        mapped.append(
            {
                "text": syllable.text,
                "source_duration_samples": syllable.duration_samples,
                "start_sample": start,
                "end_sample": end,
                "start_seconds": samples_to_seconds(start, sample_rate),
                "end_seconds": samples_to_seconds(end, sample_rate),
                "start_seconds_text": seconds_text(start, sample_rate),
                "end_seconds_text": seconds_text(end, sample_rate),
                "alignment": _alignment_dict(start, end, sample_rate),
            }
        )
    if mapped:
        mapped[-1]["end_sample"] = entry.start_sample + target_samples
        mapped[-1]["end_seconds"] = samples_to_seconds(
            entry.start_sample + target_samples, sample_rate
        )
        mapped[-1]["end_seconds_text"] = seconds_text(
            entry.start_sample + target_samples, sample_rate
        )
        mapped[-1]["alignment"] = _alignment_dict(
            mapped[-1]["start_sample"],
            mapped[-1]["end_sample"],
            sample_rate,
        )

    entry_end = entry.start_sample + target_samples
    subtitle = {
        "text": line_text,
        "timeline": _timeline_dict(entry.start_sample, entry_end, sample_rate),
        "syllables": mapped,
    }
    alignment = _alignment_dict(entry.start_sample, entry_end, sample_rate)
    return subtitle, alignment


def _stretch_metadata(
    natural_samples: int,
    target_samples: int,
    ratio: float,
    loaded: LoadedInput,
) -> dict[str, Any]:
    return {
        "natural_samples": natural_samples,
        "target_samples": target_samples,
        "ratio": round(ratio, 12),
        "min_ratio": loaded.config.stretch_min_ratio,
        "max_ratio": loaded.config.stretch_max_ratio,
        "adjusted": natural_samples != target_samples,
    }


def _old_locale_is_reusable(
    loaded: LoadedInput,
    old: OldResource | None,
    locale: str,
) -> tuple[bool, dict[str, np.ndarray]]:
    if old is None:
        return False, {}
    if old.index.get("baker_version") != BAKER_VERSION:
        return False, {}
    if old.index.get("config_hash") != loaded.config_hash:
        return False, {}
    old_locales = old.index.get("locales")
    if not isinstance(old_locales, dict):
        return False, {}
    old_locale = old_locales.get(locale)
    if not isinstance(old_locale, dict):
        return False, {}
    if old_locale.get("input_signature") != loaded.locale_signature(locale):
        return False, {}
    if not isinstance(old_locale.get("mix"), dict):
        return False, {}

    current = [
        entry for entry in loaded.entries if entry.locale == locale and entry.issue is None
    ]
    payloads: dict[str, np.ndarray] = {}
    placed: list[tuple[int, np.ndarray]] = []
    for entry in current:
        old_entry = old.entries_by_id.get(entry.entry_id)
        chunk = old.chunks_by_entry.get(entry.entry_id)
        if old_entry is None or chunk is None:
            return False, {}
        if old_entry.get("input_hash") != entry.input_hash:
            return False, {}
        timeline = old_entry.get("timeline")
        if not isinstance(timeline, dict):
            return False, {}
        if (
            timeline.get("start_sample") != entry.start_sample
            or timeline.get("end_sample")
            != entry.start_sample + entry.duration_samples
        ):
            return False, {}
        if not all(
            key in old_entry
            for key in ("stretch", "subtitle", "loudness", "alignment")
        ):
            return False, {}
        try:
            data = validate_chunk(
                old.pack_data,
                chunk,
                expected_sample_count=entry.duration_samples,
            )
        except (TypeError, ValueError):
            return False, {}
        pcm = pcm16_from_bytes(data)
        expected_loudness = old_entry.get("loudness")
        if not isinstance(expected_loudness, dict):
            return False, {}
        from .synth import pcm_stats

        actual_loudness = pcm_stats(pcm, loaded.config.sample_rate)
        if (
            expected_loudness.get("sample_peak") != actual_loudness["sample_peak"]
            or expected_loudness.get("rms_dbfs") != actual_loudness["rms_dbfs"]
        ):
            return False, {}
        payloads[entry.entry_id] = pcm
        placed.append((entry.start_sample, pcm))

    old_mix = old_locale["mix"]
    mix = place_and_mix_pcm(placed, loaded.timeline_samples)
    actual_mix_stats = mix_stats(mix, loaded.config.sample_rate)
    if (
        old_mix.get("sample_peak") != actual_mix_stats["sample_peak"]
        or old_mix.get("rms_dbfs") != actual_mix_stats["rms_dbfs"]
        or old_mix.get("timeline_samples") != actual_mix_stats["timeline_samples"]
    ):
        return False, {}
    if not isinstance(old_mix.get("mix_scale"), (int, float)):
        return False, {}
    if old_mix.get("mix_scale") < 0 or old_mix.get("mix_scale") > 1:
        return False, {}
    if not isinstance(old_mix.get("peak_protection_triggered"), bool):
        return False, {}
    if not isinstance(old_mix.get("pre_protection_peak"), (int, float)):
        return False, {}
    if not isinstance(old_mix.get("pre_protection_peak_dbfs"), (int, float)):
        return False, {}
    return True, payloads


def _render_reused_entries(
    loaded: LoadedInput,
    old: OldResource,
    locale: str,
    payloads: dict[str, np.ndarray],
) -> list[RenderedEntry]:
    rendered: list[RenderedEntry] = []
    for entry in loaded.entries:
        if entry.locale != locale or entry.issue is not None:
            continue
        old_entry = old.entries_by_id[entry.entry_id]
        pcm = payloads[entry.entry_id]
        rendered.append(
            RenderedEntry(
                plan=entry,
                pcm=pcm,
                skipped=True,
                stretch=old_entry["stretch"],
                subtitle=old_entry["subtitle"],
                loudness=old_entry["loudness"],
                alignment=old_entry["alignment"],
            )
        )
    return rendered


def _render_new_entries(loaded: LoadedInput, locale: str) -> list[RenderedEntry]:
    rendered: list[RenderedEntry] = []
    for entry in loaded.entries:
        if entry.locale != locale or entry.issue is not None:
            continue
        assert entry.line is not None
        assert entry.voice is not None
        natural = entry.line.natural_samples
        source = synthesize_line(entry.line, entry.voice, loaded.config)
        stretched, ratio, error = bounded_stretch(
            source,
            entry.duration_samples,
            loaded.config.stretch_min_ratio,
            loaded.config.stretch_max_ratio,
        )
        if error is not None or stretched is None:
            raise BakeFailure(
                "SYNTHESIS_FAILED",
                "audio could not be fitted into its slot",
                {
                    "entry_id": entry.entry_id,
                    "error": error,
                    "ratio": ratio,
                },
            )
        if stretched.size != entry.duration_samples:
            raise BakeFailure(
                "SAMPLE_BOUNDARY_MISMATCH",
                "rendered sample count does not equal the slot duration",
                {
                    "entry_id": entry.entry_id,
                    "actual": int(stretched.size),
                    "expected": entry.duration_samples,
                },
            )
        subtitle, alignment = _subtitle_metadata(
            entry,
            entry.line.text,
            entry.line.syllables,
            entry.duration_samples,
            loaded.config.sample_rate,
        )
        rendered.append(
            RenderedEntry(
                plan=entry,
                pcm=quantize_pcm16(stretched * entry.voice.linear_gain),
                skipped=False,
                stretch=_stretch_metadata(
                    natural, entry.duration_samples, ratio, loaded
                ),
                subtitle=subtitle,
                loudness={},
                alignment=alignment,
                float_audio=stretched * entry.voice.linear_gain,
            )
        )
    return rendered


def _mix_locale(
    loaded: LoadedInput,
    entries: list[RenderedEntry],
    *,
    reused_mix: dict[str, Any] | None = None,
) -> LocaleMix:
    placed: list[tuple[int, np.ndarray]] = [
        (entry.plan.start_sample, entry.pcm) for entry in entries
    ]
    mix = place_and_mix_pcm(placed, loaded.timeline_samples)
    stats = mix_stats(mix, loaded.config.sample_rate)
    if reused_mix is not None:
        return LocaleMix(
            locale=entries[0].plan.locale if entries else "",
            reused=True,
            scale=float(reused_mix["mix_scale"]),
            triggered=bool(reused_mix["peak_protection_triggered"]),
            pre_peak=float(reused_mix["pre_protection_peak"]),
            stats={
                **stats,
                "target_peak_dbfs": loaded.config.target_peak_dbfs,
                "pre_protection_peak": float(reused_mix["pre_protection_peak"]),
                "pre_protection_peak_dbfs": float(
                    reused_mix["pre_protection_peak_dbfs"]
                ),
                "mix_scale": float(reused_mix["mix_scale"]),
                "peak_protection_triggered": bool(
                    reused_mix["peak_protection_triggered"]
                ),
                "mix_pcm_i32le_sha256": sha256_bytes(
                    np.asarray(mix, dtype="<i4").tobytes()
                ),
            },
            mix=mix,
        )

    float_mix = np.zeros(loaded.timeline_samples, dtype=np.float64)
    # New entries are quantized only after a locale-wide protection pass so all
    # simultaneously speaking roles use the same scale factor.
    for entry in entries:
        assert entry.plan.voice is not None
        assert entry.float_audio is not None
        start = entry.plan.start_sample
        float_mix[start : start + entry.plan.duration_samples] += (
            entry.float_audio
        )

    pre_peak = float(np.max(np.abs(float_mix))) if float_mix.size else 0.0
    target = loaded.config.target_peak
    triggered = pre_peak > target
    scale = (target / pre_peak) if triggered and pre_peak > 0 else 1.0
    for entry in entries:
        assert entry.plan.voice is not None
        assert entry.float_audio is not None
        entry.pcm = quantize_pcm16(entry.float_audio * scale)

    mix = place_and_mix_pcm(
        [(entry.plan.start_sample, entry.pcm) for entry in entries],
        loaded.timeline_samples,
    )
    stats = mix_stats(mix, loaded.config.sample_rate)
    if stats["clipped_sample_count"]:
        loaded.issues.append(
            Issue(
                "MIX_CLIPPING_DETECTED",
                "integer mix exceeded full scale after peak protection",
                {
                    "locale": entries[0].plan.locale if entries else "",
                    "clipped_sample_count": stats["clipped_sample_count"],
                },
                severity="warning",
            )
        )
    return LocaleMix(
        locale=entries[0].plan.locale if entries else "",
        reused=False,
        scale=scale,
        triggered=triggered,
        pre_peak=pre_peak,
        stats={
            **stats,
            "target_peak_dbfs": loaded.config.target_peak_dbfs,
            "pre_protection_peak": round(pre_peak, 12),
            "pre_protection_peak_dbfs": round(dbfs(pre_peak), 9),
            "mix_scale": round(scale, 12),
            "peak_protection_triggered": triggered,
            "mix_pcm_i32le_sha256": sha256_bytes(
                np.asarray(mix, dtype="<i4").tobytes()
            ),
        },
        mix=mix,
    )


def _finalize_rendered_loudness(
    entries: list[RenderedEntry],
    sample_rate: int,
) -> None:
    from .synth import pcm_stats

    for entry in entries:
        if not entry.skipped:
            entry.loudness = pcm_stats(entry.pcm, sample_rate)


def _entry_index_record(
    loaded: LoadedInput,
    rendered: RenderedEntry,
    chunk: dict[str, Any],
) -> dict[str, Any]:
    entry = rendered.plan
    end = entry.start_sample + entry.duration_samples
    alignment = _alignment_dict(
        entry.start_sample, end, loaded.config.sample_rate
    )
    if alignment["max_error_samples"] > 1:
        raise BakeFailure(
            "ALIGNMENT_ERROR",
            "subtitle timestamp does not round-trip within one sample",
            {"entry_id": entry.entry_id, **alignment},
        )
    for syllable in rendered.subtitle.get("syllables", []):
        syllable_alignment = syllable.get("alignment", {})
        if syllable_alignment.get("max_error_samples", 0) > 1:
            raise BakeFailure(
                "ALIGNMENT_ERROR",
                "syllable timestamp does not round-trip within one sample",
                {"entry_id": entry.entry_id, **syllable_alignment},
            )
    return {
        "entry_id": entry.entry_id,
        "slot_id": entry.slot_id,
        "locale": entry.locale,
        "role": entry.role_name,
        "role_gain_db": entry.voice.gain_db if entry.voice is not None else None,
        "applied_linear_gain": (
            entry.voice.linear_gain if entry.voice is not None else None
        ),
        "table": entry.table_path,
        "input_hash": entry.input_hash,
        "text": rendered.subtitle["text"],
        "timeline": _timeline_dict(
            entry.start_sample, end, loaded.config.sample_rate
        ),
        "subtitle": rendered.subtitle,
        "stretch": rendered.stretch,
        "chunk": {
            "chunk_id": chunk["chunk_id"],
            "offset": chunk["offset"],
            "length": chunk["length"],
            "sample_count": chunk["sample_count"],
            "sha256": chunk["sha256"],
        },
        "loudness": rendered.loudness,
        "alignment": alignment,
    }


def _report_entry_record(
    loaded: LoadedInput,
    plan: EntryPlan,
    rendered: RenderedEntry | None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "entry_id": plan.entry_id,
        "slot_id": plan.slot_id,
        "locale": plan.locale,
        "role": plan.role_name,
        "table": plan.table_path,
    }
    if plan.issue is not None:
        return {
            **base,
            "status": "error",
            "error_code": plan.issue.code,
            "message": plan.issue.message,
            "context": plan.issue.context,
        }
    start = plan.start_sample
    end = start + plan.duration_samples
    if rendered is None:
        return {
            **base,
            "status": "uncommitted",
            "error_code": "OUTPUT_NOT_COMMITTED",
            "message": "entry was not part of a committed resource",
            "context": {},
            "timeline": _timeline_dict(start, end, loaded.config.sample_rate),
        }
    return {
        **base,
        "status": "skipped" if rendered.skipped else "baked",
        "timeline": _timeline_dict(start, end, loaded.config.sample_rate),
        "stretch": rendered.stretch,
        "alignment": rendered.alignment,
        "loudness": rendered.loudness,
    }


def _error_code_counter(issues: list[Issue]) -> dict[str, int]:
    counter = Counter(
        issue.code for issue in issues if issue.severity == "error"
    )
    return dict(sorted(counter.items()))


def _build_report(
    *,
    loaded: LoadedInput,
    mode: str,
    old: OldResource | None,
    rendered_entries: list[RenderedEntry],
    rendered_by_id: dict[str, RenderedEntry],
    locale_mixes: dict[str, LocaleMix],
    warnings: list[Issue],
    resource_version: str | None,
    final_status: str,
    exit_code: int,
    failure: Issue | None = None,
) -> dict[str, Any]:
    issues = list(loaded.issues) + warnings
    if failure is not None:
        issues.append(failure)
    errors = [issue for issue in issues if issue.severity == "error"]
    warning_list = [issue for issue in issues if issue.severity == "warning"]
    error_codes = _error_code_counter(errors)
    entry_error_count = sum(1 for plan in loaded.entries if plan.issue is not None)

    entries_report = [
        _report_entry_record(loaded, plan, rendered_by_id.get(plan.entry_id))
        for plan in loaded.entries
    ]
    stretch_details = [
        {
            "entry_id": item["entry_id"],
            "locale": item["locale"],
            "slot_id": item["slot_id"],
            **item["stretch"],
        }
        for item in entries_report
        if item["status"] in {"baked", "skipped"}
    ]
    for item in entries_report:
        if item.get("error_code") == "STRETCH_LIMIT_EXCEEDED":
            stretch_details.append(
                {
                    "entry_id": item["entry_id"],
                    "locale": item["locale"],
                    "slot_id": item["slot_id"],
                    "status": "rejected",
                    **{
                        key: item["context"].get(key)
                        for key in (
                            "natural_samples",
                            "target_samples",
                            "ratio",
                            "min_ratio",
                            "max_ratio",
                        )
                    },
                }
            )
    alignment_details = [
        {
            "entry_id": item["entry_id"],
            "locale": item["locale"],
            "slot_id": item["slot_id"],
            **item["alignment"],
        }
        for item in entries_report
        if "alignment" in item
    ]
    max_alignment_error = max(
        (
            item["max_error_samples"]
            for item in alignment_details
            if isinstance(item.get("max_error_samples"), int)
        ),
        default=0,
    )

    locales_report: dict[str, Any] = {}
    for locale in loaded.locales:
        plans = [entry for entry in loaded.entries if entry.locale == locale]
        final_entries = [entry for entry in plans if entry.issue is None]
        skipped = [
            entry
            for entry in final_entries
            if rendered_by_id.get(entry.entry_id) is not None
            and rendered_by_id[entry.entry_id].skipped
        ]
        baked = [
            entry
            for entry in final_entries
            if rendered_by_id.get(entry.entry_id) is not None
            and not rendered_by_id[entry.entry_id].skipped
        ]
        adjusted = [
            entry
            for entry in final_entries
            if rendered_by_id.get(entry.entry_id) is not None
            and rendered_by_id[entry.entry_id].stretch.get("adjusted", False)
        ]
        mix = locale_mixes.get(locale)
        locales_report[locale] = {
            "expected_entries": len(plans),
            "entry_count": len(final_entries),
            "skipped_entries": len(skipped),
            "baked_entries": len(baked),
            "error_entries": len(plans) - len(final_entries),
            "stretch_adjusted_entries": len(adjusted),
            "peak_protection_triggered": bool(mix.triggered) if mix else False,
            "peak_protection_trigger_count": int(bool(mix.triggered)) if mix else 0,
            "mix": mix.stats if mix else None,
        }

    skipped_total = sum(
        1 for entry in rendered_entries if entry.skipped
    )
    baked_total = len(rendered_entries) - skipped_total
    peak_trigger_total = sum(
        1 for mix in locale_mixes.values() if mix.triggered
    )
    report = {
        "report_schema_version": 1,
        "baker_version": BAKER_VERSION,
        "mode": mode,
        "final_status": final_status,
        "exit_code": exit_code,
        "resource_version": resource_version,
        "input_version": loaded.input_version,
        "config_hash": loaded.config_hash,
        "project_hash": loaded.project_hash,
        "project_id": loaded.manifest_raw.get("project_id"),
        "summary": {
            "expected_entries": len(loaded.entries),
            "entry_count": len(rendered_entries),
            "baked_entries": baked_total,
            "skipped_entries": skipped_total,
            "error_count": len(errors),
            "entry_error_count": entry_error_count,
            "global_error_count": len(errors) - entry_error_count,
            "warning_count": len(warning_list),
            "locale_entry_counts": {
                locale: locales_report[locale]["entry_count"]
                for locale in loaded.locales
            },
            "locale_skipped_counts": {
                locale: locales_report[locale]["skipped_entries"]
                for locale in loaded.locales
            },
            "peak_protection_trigger_count": peak_trigger_total,
            "max_alignment_error_samples": max_alignment_error,
            "error_codes": error_codes,
        },
        "locales": locales_report,
        "stretch_details": stretch_details,
        "alignment_details": alignment_details,
        "entries": entries_report,
        "errors": [issue.to_dict() for issue in errors],
        "warnings": [issue.to_dict() for issue in warning_list],
        "inputs": {
            "project": str(loaded.manifest_path),
            "config_hash": loaded.config_hash,
            "project_hash": loaded.project_hash,
            "tables": loaded.table_files,
        },
    }
    if old is not None:
        report["previous_resource_version"] = old.index.get("resource_version")
    return report


def _final_status(loaded: LoadedInput, warnings: list[Issue]) -> tuple[str, int]:
    if any(issue.severity == "error" for issue in loaded.issues):
        return "degraded", EXIT_DEGRADED
    return "ok", EXIT_OK


def _build_failure_report(
    loaded: LoadedInput,
    mode: str,
    failure: Issue,
    old: OldResource | None,
) -> dict[str, Any]:
    warnings = list(old.warnings) if old is not None else []
    report = _build_report(
        loaded=loaded,
        mode=mode,
        old=old,
        rendered_entries=[],
        rendered_by_id={},
        locale_mixes={},
        warnings=warnings,
        resource_version=None,
        final_status="failed",
        exit_code=EXIT_FATAL,
        failure=failure,
    )
    return report


def bake_project(
    loaded: LoadedInput,
    output_dir: Path,
    *,
    mode: str,
) -> BakeResult:
    if mode not in {"full", "incremental"}:
        raise ValueError("mode must be full or incremental")
    output_dir.mkdir(parents=True, exist_ok=True)

    old: OldResource | None = None
    old_warnings: list[Issue] = []
    if mode == "incremental":
        old, old_warnings = _read_old_resource(output_dir)

    _prepare_stretch_checks(loaded)
    warnings = list(old_warnings)
    reusable_payloads: dict[str, dict[str, np.ndarray]] = {}
    reused_locales: set[str] = set()
    if old is not None:
        for locale in loaded.locales:
            reusable, payloads = _old_locale_is_reusable(loaded, old, locale)
            if reusable:
                reused_locales.add(locale)
                reusable_payloads[locale] = payloads

    rendered_entries: list[RenderedEntry] = []
    for locale in loaded.locales:
        if locale in reused_locales:
            rendered_entries.extend(
                _render_reused_entries(
                    loaded, old, locale, reusable_payloads[locale]
                )
            )
        else:
            rendered_entries.extend(_render_new_entries(loaded, locale))

    rendered_by_id = {item.plan.entry_id: item for item in rendered_entries}
    locale_mixes: dict[str, LocaleMix] = {}
    for locale in loaded.locales:
        locale_entries = [
            entry for entry in rendered_entries if entry.plan.locale == locale
        ]
        reused_mix = None
        if locale in reused_locales:
            assert old is not None
            old_locale = old.index["locales"][locale]
            reused_mix = old_locale["mix"]
        locale_mixes[locale] = _mix_locale(
            loaded,
            locale_entries,
            reused_mix=reused_mix,
        )

    _finalize_rendered_loudness(rendered_entries, loaded.config.sample_rate)

    locale_order = {locale: index for index, locale in enumerate(loaded.locales)}
    rendered_entries.sort(
        key=lambda item: (
            locale_order[item.plan.locale],
            item.plan.start_sample,
            item.plan.order,
            item.plan.slot_id,
        )
    )
    payloads = [
        ChunkPayload(
            entry_id=item.plan.entry_id,
            locale=item.plan.locale,
            sample_count=item.plan.duration_samples,
            data=pcm16_bytes(item.pcm),
        )
        for item in rendered_entries
    ]
    built = build_pack(payloads, loaded.config, loaded.timeline_samples)
    chunk_by_entry = {
        chunk["entry_id"]: chunk for chunk in built.chunks
    }
    entries_index = [
        _entry_index_record(
            loaded,
            rendered,
            chunk_by_entry[rendered.plan.entry_id],
        )
        for rendered in rendered_entries
    ]
    for locale in loaded.locales:
        locate_entries = [
            item for item in entries_index if item["locale"] == locale
        ]
        for item in locate_entries:
            alignment = item["alignment"]
            if alignment["max_error_samples"] > 1:
                raise BakeFailure(
                    "ALIGNMENT_ERROR",
                    "entry boundary is not sample aligned",
                    {"entry_id": item["entry_id"], **alignment},
                )

    timeline_samples = loaded.timeline_samples
    locale_index = {
        locale: {
            "input_signature": loaded.locale_signature(locale),
            "entry_count": sum(
                1 for item in entries_index if item["locale"] == locale
            ),
            "entry_ids": [
                item["entry_id"]
                for item in entries_index
                if item["locale"] == locale
            ],
            "mix": locale_mixes[locale].stats,
        }
        for locale in loaded.locales
    }
    error_codes = _error_code_counter(loaded.issues)
    core = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "baker_version": BAKER_VERSION,
        "pack_format_version": PACK_FORMAT_VERSION,
        "input_version": loaded.input_version,
        "project_hash": loaded.project_hash,
        "config_hash": loaded.config_hash,
        "project_id": loaded.manifest_raw.get("project_id"),
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "sample_format": SAMPLE_FORMAT,
        "timeline": {
            "timeline_start_sample": 0,
            "timeline_end_sample": timeline_samples,
            "timeline_end_seconds": samples_to_seconds(
                timeline_samples, SAMPLE_RATE
            ),
            "timeline_end_seconds_text": seconds_text(
                timeline_samples, SAMPLE_RATE
            ),
        },
        "pack": {
            "size": len(built.data),
            "sha256": built.sha256,
        },
        "chunk_table": built.chunks,
        "entries": entries_index,
        "locales": locale_index,
        "summary": {
            "entry_count": len(entries_index),
            "error_codes": error_codes,
            "degraded": bool(error_codes),
        },
    }
    resource_version = object_hash(core)
    pack_filename = f"dialogue-{resource_version}.pack"
    index_obj = {
        **core,
        "resource_version": resource_version,
        "pack": {
            **core["pack"],
            "file": pack_filename,
        },
    }
    status, exit_code = _final_status(loaded, warnings)
    report = _build_report(
        loaded=loaded,
        mode=mode,
        old=old,
        rendered_entries=rendered_entries,
        rendered_by_id=rendered_by_id,
        locale_mixes=locale_mixes,
        warnings=warnings,
        resource_version=resource_version,
        final_status=status,
        exit_code=exit_code,
    )

    pack_path = output_dir / pack_filename
    try:
        ensure_pack_file(pack_path, built.data, built.sha256)
        index_path = output_dir / "index.json"
        write_json_atomic(index_path, index_obj)
        # The index switch is the commit point. Cleanup happens only after the
        # new index is durably installed and therefore cannot break the old
        # pack before the new index exists.
        for candidate in output_dir.glob("dialogue-*.pack"):
            if candidate != pack_path:
                try:
                    candidate.unlink()
                except OSError:
                    pass  # A stale pack is harmless and is retried next run.
        write_json_atomic(output_dir / "bake_report.json", report)
    except Exception as exc:
        failure = Issue(
            "OUTPUT_WRITE_FAILED",
            "failed to commit baked resources",
            {"output_dir": str(output_dir), "error": str(exc)},
        )
        failure_report = _build_failure_report(loaded, mode, failure, old)
        try:
            write_json_atomic(output_dir / "bake_report.json", failure_report)
        except OSError:
            pass  # The primary failure is still surfaced by raise/exit code.
        if isinstance(exc, BakeFailure):
            raise
        raise BakeFailure(
            "OUTPUT_WRITE_FAILED",
            "failed to commit baked resources",
            {"output_dir": str(output_dir), "error": str(exc)},
        ) from exc

    return BakeResult(
        index=index_obj,
        report=report,
        pack_path=pack_path,
        exit_code=exit_code,
    )


def write_fatal_report(
    output_dir: Path,
    *,
    mode: str,
    issue: Issue,
    project_path: Path,
    config_path: Path,
) -> None:
    available_version: str | None = None
    index_path = output_dir / "index.json"
    if index_path.is_file():
        try:
            available_version = loads_json_strict(index_path.read_bytes()).get(
                "resource_version"
            )
        except (OSError, ValueError, AttributeError):
            available_version = None
    report = {
        "report_schema_version": 1,
        "baker_version": BAKER_VERSION,
        "mode": mode,
        "final_status": "failed",
        "exit_code": EXIT_FATAL,
        "resource_version": None,
        "input_version": None,
        "summary": {
            "expected_entries": 0,
            "entry_count": 0,
            "baked_entries": 0,
            "skipped_entries": 0,
            "error_count": 1,
            "error_entry_count": 1,
            "global_error_count": 0,
            "warning_count": 0,
            "locale_entry_counts": {},
            "locale_skipped_counts": {},
            "peak_protection_trigger_count": 0,
            "max_alignment_error_samples": 0,
            "error_codes": {issue.code: 1},
        },
        "locales": {},
        "stretch_details": [],
        "alignment_details": [],
        "entries": [],
        "errors": [issue.to_dict()],
        "warnings": [],
        "inputs": {
            "project": str(project_path),
            "config": str(config_path),
        },
        "available_resource_version": available_version,
        "note": (
            "This failure path did not write a new pack or index. Any existing "
            "index.json and its referenced pack were left intact."
        ),
    }
    write_json_atomic(output_dir / "bake_report.json", report)
