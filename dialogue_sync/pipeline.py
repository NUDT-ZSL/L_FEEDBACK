"""End-to-end dialogue conform pipeline."""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio import (
    EntryAudio,
    mix_entries,
    normalize_loudness,
    prepare_entry_audio,
    read_fragment,
    write_fragment_wav,
    write_mix_wav,
)
from .errors import (
    ConfigError,
    DialogueSyncError,
    EntryError,
    ExternalToolError,
)
from .fftools import (
    FFmpegTools,
    atomic_ffmpeg_output,
    atomic_write_bytes,
    atomic_write_json,
)
from .model import (
    DialogueEntry,
    Project,
    derive_total_duration,
    validate_timeline_against_subtitles,
)
from .srt import (
    SubtitleCue,
    SubtitleDocument,
    find_overlaps,
    read_srt,
)
from .state import CheckpointStore, sha256_bytes, sha256_file
from .timecode import format_timecode, ms_to_samples


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _signed_timecode(milliseconds: int) -> str:
    sign = "-" if milliseconds < 0 else ""
    return f"{sign}{format_timecode(abs(milliseconds))}"


def _json_input_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read input {path}: {exc}") from exc


def compute_manifest(
    project: Project,
    *,
    srt_bytes: bytes,
    total_duration_ms: int,
) -> tuple[str, dict[str, str]]:
    timeline_bytes = _json_input_bytes(project.timeline_path)
    source_hashes: dict[str, str] = {}
    for entry in project.entries:
        key = str(entry.source_path)
        if key in source_hashes:
            continue
        if not entry.source_path.is_file():
            source_hashes[key] = "MISSING"
            continue
        try:
            source_hashes[key] = sha256_file(entry.source_path)
        except OSError as exc:
            raise ConfigError(f"cannot hash source {entry.source_path}: {exc}") from exc

    input_hashes = {
        "timeline_sha256": sha256_bytes(timeline_bytes),
        "srt_sha256": sha256_bytes(srt_bytes),
        **{f"source:{key}": value for key, value in sorted(source_hashes.items())},
    }
    manifest = {
        "schema": 1,
        "audio": asdict(project.audio),
        "output": asdict(project.output),
        "total_duration_ms": total_duration_ms,
        "entries": [
            {
                "ordinal": entry.ordinal,
                "id": entry.entry_id,
                "source": str(entry.source_path),
                "source_start_ms": entry.source_start_ms,
                "source_offset_ms": entry.source_offset_ms,
                "source_end_ms": entry.source_end_ms,
                "target_start_ms": entry.target_start_ms,
                "target_end_ms": entry.target_end_ms,
                "target_duration_ms": entry.target_duration_ms,
                "subtitle_index": entry.subtitle_index,
                "weight": entry.weight,
            }
            for entry in project.entries
        ],
        "input_hashes": input_hashes,
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded), input_hashes


def _entry_base_report(
    entry: DialogueEntry,
    cue: SubtitleCue | None,
    subtitle_update: dict[str, Any] | None,
) -> dict[str, Any]:
    target_duration = entry.target_duration_ms or 0
    report: dict[str, Any] = {
        "index": entry.ordinal,
        "entry_id": entry.entry_id,
        "source_file": entry.source_file_as_given,
        "source_path": str(entry.source_path),
        "source_timecode": {
            "start": format_timecode(entry.source_start_ms),
            "end": None,
            "file_offset": format_timecode(entry.source_offset_ms),
        },
        "target_timecode": {
            "start": format_timecode(entry.target_start_ms),
            "end": format_timecode(entry.target_start_ms + target_duration),
            "duration": format_timecode(target_duration),
        },
        "input_validation_errors": entry.field_errors,
        "subtitle_index": entry.subtitle_index,
        "timeline_offset_ms": entry.target_start_ms - entry.source_start_ms,
        "timeline_offset": _signed_timecode(
            entry.target_start_ms - entry.source_start_ms
        ),
        "original_subtitle_timecode": (
            {
                "start": format_timecode(cue.original_start_ms),
                "end": format_timecode(cue.original_end_ms),
            }
            if cue is not None
            else None
        ),
        "subtitle_update": subtitle_update,
        "status": "pending",
        "operation": "none",
        "operation_label": "无",
        "operations": [],
        "stretch_rate": None,
        "atempo": None,
        "applied_gain_db": 0.0,
        "gain_breakdown_db": {
            "weight": 0.0,
            "entry_clip_trim": 0.0,
            "mix_trim": 0.0,
            "loudness_normalization": 0.0,
        },
        "entry_weight": entry.weight,
        "error_code": None,
        "error_codes": [],
        "warnings": [],
        "message": "",
    }
    if subtitle_update is not None and subtitle_update.get("clamped"):
        report["error_code"] = "E_TIMECODE"
        report["error_codes"] = ["E_TIMECODE"]
        report["message"] = (
            "mapped subtitle interval was negative or zero-length and was clamped"
        )
    report["original_timecode"] = report["source_timecode"]
    return report


def _map_subtitles(
    project: Project,
    document: SubtitleDocument,
) -> tuple[dict[int, SubtitleCue], dict[int, dict[str, Any]], list[str]]:
    cues = document.by_index()
    updates: dict[int, dict[str, Any]] = {}
    warnings: list[str] = []
    for entry in project.entries:
        cue = cues[entry.subtitle_index]
        if entry.field_errors:
            updates[entry.ordinal] = {
                "index": cue.index,
                "skipped": True,
                "reason": "E_TIMECODE",
            }
            continue
        offset = entry.target_start_ms - entry.source_start_ms
        raw_start = cue.start_ms + offset
        explicit_duration = any(
            key in entry.raw
            for key in (
                "target_duration",
                "target_duration_tc",
                "targetDuration",
            )
        )
        if entry.target_end_ms is not None:
            raw_end = entry.target_end_ms
        elif explicit_duration and entry.target_duration_ms is not None:
            raw_end = entry.target_start_ms + entry.target_duration_ms
        else:
            raw_end = cue.end_ms + offset
        clamped = raw_start < 0 or raw_end <= raw_start or raw_end <= 0
        new_start = max(0, raw_start)
        new_end = max(new_start + 1, raw_end)
        if clamped:
            warnings.append(
                f"entry {entry.ordinal} subtitle mapping was clamped to a positive "
                "non-zero interval"
            )
        cue.start_ms = new_start
        cue.end_ms = new_end
        cue.updated_by.append(entry.ordinal)
        updates[entry.ordinal] = {
            "index": cue.index,
            "offset_ms": offset,
            "offset": _signed_timecode(offset),
            "original_start": format_timecode(cue.original_start_ms),
            "original_end": format_timecode(cue.original_end_ms),
            "mapped_start": format_timecode(new_start),
            "mapped_end": format_timecode(new_end),
            "clamped": clamped,
            "warning_codes": ["W_SUBTITLE_CLAMPED"] if clamped else [],
        }
    return cues, updates, warnings


def _validate_input_probe(entry: DialogueEntry, probe: Any) -> None:
    if not probe.codec_name.startswith("pcm_"):
        raise EntryError("E_FORMAT", f"source codec {probe.codec_name!r} is not PCM WAV")
    if probe.sample_rate not in (44_100, 48_000):
        raise EntryError(
            "E_FORMAT", f"source sample rate {probe.sample_rate} is not 44.1 or 48 kHz"
        )
    if probe.channels not in (1, 2):
        raise EntryError("E_FORMAT", f"source channel count {probe.channels} is unsupported")
    if probe.bit_depth not in (16, 24):
        raise EntryError(
            "E_FORMAT", f"source bit depth {probe.bit_depth} is not 16 or 24-bit"
        )


def _skip_report(
    report: dict[str, Any],
    *,
    code: str,
    message: str,
    probe: Any | None = None,
    entry: DialogueEntry | None = None,
) -> dict[str, Any]:
    report["status"] = "skipped"
    report["operation"] = "skip"
    report["operation_label"] = "跳过"
    report["operations"] = ["skip"]
    report["error_code"] = code
    report["error_codes"] = [code]
    report["message"] = message
    if probe is not None and entry is not None:
        file_duration_ms = int(round(probe.duration_seconds * 1_000))
        if entry.source_end_ms is not None:
            span = entry.source_end_ms - entry.source_start_ms
            source_end_ms = entry.source_end_ms
        else:
            span = file_duration_ms - entry.source_offset_ms
            source_end_ms = entry.source_start_ms + max(span, 0)
        if span > 0 and entry.target_duration_ms:
            report["stretch_rate"] = entry.target_duration_ms / float(span)
            report["atempo"] = span / float(entry.target_duration_ms)
        report["source_timecode"]["end"] = format_timecode(source_end_ms)
        report["source_audio"] = {
            "codec": probe.codec_name,
            "sample_rate": probe.sample_rate,
            "channels": probe.channels,
            "bit_depth": probe.bit_depth,
            "duration": format_timecode(int(round(probe.duration_seconds * 1_000))),
        }
    return report


def _process_entry(
    tools: FFmpegTools,
    entry: DialogueEntry,
    project: Project,
    report: dict[str, Any],
    store: CheckpointStore,
) -> tuple[dict[str, Any], bool]:
    probe = None
    try:
        if not entry.source_path.is_file():
            raise EntryError("E_IO", f"source WAV does not exist: {entry.source_path}")
        probe = tools.probe_audio(entry.source_path)
        _validate_input_probe(entry, probe)
        report["source_audio"] = {
            "codec": probe.codec_name,
            "sample_rate": probe.sample_rate,
            "channels": probe.channels,
            "bit_depth": probe.bit_depth,
            "duration": format_timecode(int(round(probe.duration_seconds * 1_000))),
        }
        if probe.channels == project.audio.channels:
            mapping = "stereo passthrough" if probe.channels == 2 else "mono passthrough"
        elif probe.channels == 1 and project.audio.channels == 2:
            mapping = "mono to dual-mono stereo at -3.01 dB per channel"
        else:
            mapping = "stereo to mono by averaging L/R"
        report["channel_mapping"] = {
            "source_channels": probe.channels,
            "output_channels": project.audio.channels,
            "method": mapping,
        }
        prepared = prepare_entry_audio(tools, entry, project.audio, probe)
        report["source_timecode"]["end"] = format_timecode(
            entry.source_end_ms
            if entry.source_end_ms is not None
            else entry.source_start_ms + prepared.source_span_ms
        )
        report["source_timecode"]["duration"] = format_timecode(prepared.source_span_ms)
        report["source_timecode"]["file_offset"] = format_timecode(
            prepared.source_offset_ms
        )
        report["status"] = "ok"
        report["operation"] = prepared.primary_operation
        report["operation_label"] = prepared.operation_label
        report["operations"] = prepared.operations
        report["stretch_rate"] = round(prepared.stretch_rate, 9)
        report["atempo"] = round(prepared.atempo, 9)
        report["residual_crop_ms"] = round(prepared.residual_crop_ms, 6)
        report["residual_pad_ms"] = round(prepared.residual_pad_ms, 6)
        report["source_peak_dbfs"] = (
            20.0 * math.log10(prepared.input_peak) if prepared.input_peak > 0 else None
        )
        report["fragment_peak_dbfs"] = (
            20.0 * math.log10(prepared.output_peak) if prepared.output_peak > 0 else None
        )
        report["gain_breakdown_db"]["weight"] = (
            20.0 * math.log10(entry.weight) if entry.weight > 0 else -math.inf
        )
        report["gain_breakdown_db"]["entry_clip_trim"] = prepared.clip_gain_db
        report["error_codes"].extend(prepared.error_codes)
        report["warnings"].extend(prepared.warnings)
        if "E_CLIP" in prepared.error_codes:
            report["message"] = "entry samples exceeded full scale and were attenuated"

        fragment = store.fragment_path(
            entry.ordinal,
            f"{entry.source_path}|{entry.source_start_ms}|{entry.target_start_ms}|"
            f"{entry.target_duration_ms}",
        )
        raw_temp = store.raw_dir / f"entry_{entry.ordinal:04d}.f32"
        write_fragment_wav(
            tools,
            fragment,
            prepared.samples,
            project.audio,
            raw_temp=raw_temp,
        )
        store.record_fragment(
            entry.ordinal,
            fragment,
            samples=prepared.samples.shape[0],
            duration_ms=prepared.target_duration_ms,
            weight=entry.weight,
        )
        report["fragment"] = store.relative_fragment_path(fragment)
        return report, True
    except EntryError as exc:
        if exc.warning:
            report["warnings"].append(exc.code)
            if exc.code not in report["error_codes"]:
                report["error_codes"].append(exc.code)
            report["message"] = exc.message
            return report, False
        return _skip_report(
            report,
            code=exc.code,
            message=exc.message,
            probe=probe,
            entry=entry,
        ), False
    except ExternalToolError as exc:
        return _skip_report(
            report,
            code="E_DECODE",
            message=str(exc),
            probe=probe,
            entry=entry,
        ), False
    except Exception as exc:
        report["status"] = "error"
        report["operation"] = "skip"
        report["operation_label"] = "跳过"
        report["operations"] = ["skip"]
        report["error_code"] = "E_INTERNAL"
        report["error_codes"] = ["E_INTERNAL"]
        report["message"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc(limit=6)
        return report, False


def _prepared_from_state(
    entry: DialogueEntry,
    report: dict[str, Any],
    store: CheckpointStore,
    tools: FFmpegTools,
    project: Project,
) -> EntryAudio:
    record = store.get_fragment(entry.ordinal)
    if record is None:
        raise ExternalToolError(
            f"entry {entry.ordinal} is marked complete but has no valid fragment"
        )
    fragment = store.fragment_absolute_path(str(record["path"]))
    samples = read_fragment(tools, fragment, project.audio)
    expected = int(record.get("samples", -1))
    if samples.shape[0] != expected:
        raise ExternalToolError(
            f"entry {entry.ordinal} fragment sample count changed ({samples.shape[0]} != {expected})"
        )
    if samples.shape[0] != ms_to_samples(entry.target_duration_ms or 0, project.audio.sample_rate):
        raise ExternalToolError(
            f"entry {entry.ordinal} fragment duration differs from its target duration"
        )
    return EntryAudio(
        entry=entry,
        samples=samples,
        source_probe=None,
        source_span_ms=int(round(samples.shape[0] * 1_000.0 / project.audio.sample_rate)),
        source_offset_ms=entry.source_offset_ms,
        target_duration_ms=entry.target_duration_ms or 0,
        stretch_rate=float(report.get("stretch_rate") or 1.0),
        atempo=float(report.get("atempo") or 1.0),
        primary_operation=str(report.get("operation") or "none"),
        operations=list(report.get("operations") or []),
    )


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.partial")
    try:
        with source.open("rb") as input_handle, temp.open("wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temp, target)
        try:
            descriptor = os.open(target.parent, os.O_RDONLY)
        except OSError:
            descriptor = None
        if descriptor is not None:
            try:
                os.fsync(descriptor)
            except OSError:
                pass
            finally:
                os.close(descriptor)
    except Exception:
        try:
            temp.unlink(missing_ok=True)
        finally:
            raise


def _ffmetadata_escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("=", "\\=")
        .replace(";", "\\;")
        .replace("#", "\\#")
    )


def _chapter_groups(project: Project, total_ms: int) -> list[dict[str, Any]]:
    grouped: dict[int, list[DialogueEntry]] = {}
    for entry in project.entries:
        if entry.field_errors or entry.target_start_ms >= total_ms:
            continue
        grouped.setdefault(entry.target_start_ms, []).append(entry)
    starts = sorted(grouped)
    result: list[dict[str, Any]] = []
    for position, start in enumerate(starts):
        entries = grouped[start]
        end = starts[position + 1] if position + 1 < len(starts) else total_ms
        if end <= start:
            end = min(total_ms, start + 1)
        indexes = ", ".join(str(entry.ordinal) for entry in entries)
        labels = ", ".join(entry.entry_id for entry in entries)
        title = f"Dialogue {indexes} [{labels}]"
        result.append(
            {
                "start_ms": start,
                "end_ms": end,
                "title": title,
                "entry_indexes": [entry.ordinal for entry in entries],
            }
        )
    return result


def _write_chapters(path: Path, groups: list[dict[str, Any]], title: str) -> None:
    lines = [";FFMETADATA1", f"title={_ffmetadata_escape(title)}"]
    for group in groups:
        lines.extend(
            [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={group['start_ms']}",
                f"END={group['end_ms']}",
                f"title={_ffmetadata_escape(group['title'])}",
            ]
        )
    atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))


def _package_and_verify(
    tools: FFmpegTools,
    project: Project,
    normalized_wav: Path,
    subtitle_bytes: bytes,
    chapters: list[dict[str, Any]],
    output_dir: Path,
    work_dir: Path,
    total_ms: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chapters_path = work_dir / "chapters.ffmeta"
    package_path = work_dir / "package.mkv"
    packaged_srt_path = work_dir / "updated.srt"
    _write_chapters(chapters_path, chapters, project.output.title)
    atomic_write_bytes(packaged_srt_path, subtitle_bytes)

    audio_out = output_dir / project.output.audio_filename
    srt_out = output_dir / project.output.subtitle_filename
    mkv_out = output_dir / project.output.container_filename

    expected_duration = total_ms / 1_000.0
    atomic_ffmpeg_output(
        tools,
        package_path,
        [
            "-i",
            str(normalized_wav),
            "-i",
            str(packaged_srt_path),
            "-f",
            "ffmetadata",
            "-i",
            str(chapters_path),
            "-map",
            "0:a:0",
            "-map",
            "1:s:0",
            "-map_metadata",
            "2",
            "-map_chapters",
            "2",
            "-c:a",
            "pcm_s24le",
            "-c:s",
            "srt",
            f"-metadata:s:a:0",
            f"language={project.output.language}",
            "-metadata:s:a:0",
            "title=Main Dialogue",
            "-metadata:s:s:0",
            f"language={project.output.subtitle_language}",
            "-metadata:s:s:0",
            "title=Dialogue Subtitles",
            "-metadata",
            f"title={project.output.title}",
        ],
    )

    probe = tools.probe_json(package_path)
    streams = probe.get("streams", [])
    audio_streams = [item for item in streams if item.get("codec_type") == "audio"]
    subtitle_streams = [item for item in streams if item.get("codec_type") == "subtitle"]
    if len(audio_streams) != 1:
        raise ExternalToolError("packaged MKV must contain exactly one audio stream")
    if len(subtitle_streams) != 1:
        raise ExternalToolError("packaged MKV must contain exactly one subtitle stream")
    audio_stream = audio_streams[0]
    subtitle_stream = subtitle_streams[0]
    audio_language = (audio_stream.get("tags") or {}).get("language")
    subtitle_language = (subtitle_stream.get("tags") or {}).get("language")
    if audio_language != project.output.language:
        raise ExternalToolError(
            f"MKV audio language {audio_language!r} != {project.output.language!r}"
        )
    if subtitle_language != project.output.subtitle_language:
        raise ExternalToolError(
            f"MKV subtitle language {subtitle_language!r} != "
            f"{project.output.subtitle_language!r}"
        )
    chapters_in_mkv = probe.get("chapters") or []
    if len(chapters_in_mkv) != len(chapters):
        raise ExternalToolError(
            f"MKV has {len(chapters_in_mkv)} chapters, expected {len(chapters)}"
        )
    for expected, actual in zip(chapters, chapters_in_mkv):
        tags = actual.get("tags") or {}
        actual_title = tags.get("title", "")
        if any(f"Dialogue {index}" not in actual_title for index in expected["entry_indexes"]):
            # Titles for shared-start groups list multiple indexes at once.
            if not all(str(index) in actual_title for index in expected["entry_indexes"]):
                raise ExternalToolError(
                    f"MKV chapter title {actual_title!r} lacks entry index"
                )
    format_duration = float((probe.get("format") or {}).get("duration") or 0.0)
    duration_delta_ms = abs(format_duration - expected_duration) * 1_000.0
    if duration_delta_ms > 50:
        raise ExternalToolError(
            f"MKV duration differs from target timeline by {duration_delta_ms:.3f} ms "
            "(limit 50 ms)"
        )
    if int(audio_stream.get("sample_rate", 0)) != 48_000:
        raise ExternalToolError("MKV audio is not 48 kHz")
    if int(audio_stream.get("channels", 0)) != project.audio.channels:
        raise ExternalToolError("MKV audio channel count differs from configuration")

    atomic_write_bytes(srt_out, subtitle_bytes)
    _atomic_copy(package_path, mkv_out)
    _atomic_copy(normalized_wav, audio_out)
    packaged = {
        "path": str(mkv_out),
        "audio_path": str(audio_out),
        "subtitle_path": str(srt_out),
        "language": audio_language,
        "subtitle_language": subtitle_language,
        "chapters": chapters,
        "duration_seconds": format_duration,
        "expected_duration_seconds": expected_duration,
        "duration_error_ms": round(duration_delta_ms, 3),
        "streams": [
            {
                "type": item.get("codec_type"),
                "codec": item.get("codec_name"),
                "sample_rate": item.get("sample_rate"),
                "channels": item.get("channels"),
                "bits_per_raw_sample": item.get("bits_per_raw_sample"),
                "language": (item.get("tags") or {}).get("language"),
                "title": (item.get("tags") or {}).get("title"),
            }
            for item in streams
        ],
        "verified": True,
    }
    return packaged


def _entry_error_codes(report: dict[str, Any]) -> list[str]:
    return [str(code) for code in report.get("error_codes", [])]


def run_build(
    project: Project,
    *,
    output_dir: Path,
    work_dir: Path,
    tools: FFmpegTools,
    force: bool = False,
    simulate_abort_after: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    document, srt_bytes = read_srt(project.srt_path)
    cues = document.by_index()
    validate_timeline_against_subtitles(project, cues)
    total_ms = derive_total_duration(project)
    manifest_hash, input_hashes = compute_manifest(
        project, srt_bytes=srt_bytes, total_duration_ms=total_ms
    )
    mapped_cues, subtitle_updates, mapping_warnings = _map_subtitles(project, document)
    overlaps = find_overlaps(document.cues)

    store = CheckpointStore(work_dir, manifest_hash)
    store.load(force=force)
    if mapping_warnings:
        for warning in mapping_warnings:
            if warning not in store.state.setdefault("warnings", []):
                store.state["warnings"].append(warning)

    total_count = len(project.entries)
    newly_processed = 0
    for entry in project.entries:
        if store.has_report(entry.ordinal):
            existing = store.get_report(entry.ordinal)
            if existing and existing.get("status") == "ok" and not store.get_fragment(
                entry.ordinal
            ):
                store.state["entry_reports"].pop(str(entry.ordinal), None)
            else:
                continue
        cue = mapped_cues.get(entry.subtitle_index)
        report = _entry_base_report(entry, cue, subtitle_updates[entry.ordinal])
        target_end_ms = entry.target_start_ms + (entry.target_duration_ms or 0)
        if entry.field_errors:
            report = _skip_report(
                report,
                code="E_TIMECODE",
                message="; ".join(
                    error["message"] for error in entry.field_errors
                ),
            )
            wrote_fragment = False
        elif entry.target_start_ms >= total_ms or target_end_ms > total_ms:
            report = _skip_report(
                report,
                code="E_RANGE",
                message=(
                    f"target interval {format_timecode(entry.target_start_ms)} -> "
                    f"{format_timecode(target_end_ms)} exceeds total timeline "
                    f"{format_timecode(total_ms)}"
                ),
            )
            wrote_fragment = False
        else:
            report, wrote_fragment = _process_entry(
                tools, entry, project, report, store
            )
        if report.get("status") == "ok" and not wrote_fragment:
            report = _skip_report(
                report,
                code="E_INTERNAL",
                message="entry was not persisted despite successful processing",
            )
        store.record_report(entry.ordinal, report)
        store.save()
        newly_processed += 1
        if progress:
            state = report.get("status")
            operation = report.get("operation")
            code = report.get("error_code") or ""
            print(
                f"[{entry.ordinal}/{total_count}] {entry.entry_id}: "
                f"{state} {operation} {code}".rstrip(),
                flush=True,
            )
        if simulate_abort_after is not None and newly_processed >= simulate_abort_after:
            print(
                f"SIMULATED ABORT after {newly_processed} checkpointed entries",
                file=sys.stderr,
                flush=True,
            )
            os._exit(75)

    reports = [
        store.get_report(entry.ordinal) or {}
        for entry in project.entries
    ]
    successful_reports = [report for report in reports if report.get("status") == "ok"]
    if not successful_reports:
        raise DialogueSyncError(
            "no dialogue entry was processed successfully; no output track can be created",
            code="E_NO_AUDIO",
        )

    prepared: list[EntryAudio] = []
    for entry in project.entries:
        report = store.get_report(entry.ordinal)
        if report and report.get("status") == "ok":
            prepared.append(
                _prepared_from_state(entry, report, store, tools, project)
            )
    mix = mix_entries(prepared, total_duration_ms=total_ms, audio=project.audio)
    mix_path = work_dir / "mix.wav"
    write_mix_wav(tools, mix_path, mix, project.audio)
    mix_record = {
        "path": "mix.wav",
        "sha256": sha256_file(mix_path),
        "pre_trim_peak_dbfs": (
            20.0 * math.log10(mix.pre_trim_peak) if mix.pre_trim_peak > 0 else None
        ),
        "clipping_detected": mix.clipping_detected,
        "mix_gain_db": round(mix.mix_gain_db, 6),
        "clipped_entry_ordinals": mix.clipped_entry_ordinals,
        "total_samples": mix.total_samples,
        "duration_ms": total_ms,
    }
    store.state["mix"] = mix_record
    store.state["phase"] = "mixed"
    for ordinal in mix.clipped_entry_ordinals:
        report = store.get_report(ordinal)
        if not report:
            continue
        if "E_CLIP" not in report.get("error_codes", []):
            report.setdefault("error_codes", []).append("E_CLIP")
        report["error_code"] = report.get("error_code") or "E_CLIP"
        report["gain_breakdown_db"]["mix_trim"] = round(mix.mix_gain_db, 6)
        message = (
            f"overlapping weighted mix exceeded full scale; global attenuation "
            f"{mix.mix_gain_db:.3f} dB applied"
        )
        if message not in report.setdefault("warnings", []):
            report["warnings"].append(message)
    for report in successful_reports:
        report["gain_breakdown_db"]["mix_trim"] = round(mix.mix_gain_db, 6)
    store.save()
    if progress:
        print(
            f"mixed {len(prepared)} entries; peak {mix_record['pre_trim_peak_dbfs']} dBFS; "
            f"mix trim {mix.mix_gain_db:.3f} dB",
            flush=True,
        )

    normalized_path = work_dir / "normalized.wav"
    cached = store.state.get("normalized")
    loudness: dict[str, Any] | None = None
    if (
        isinstance(cached, dict)
        and isinstance(cached.get("sha256"), str)
        and normalized_path.is_file()
        and sha256_file(normalized_path) == cached["sha256"]
        and isinstance(store.state.get("loudness"), dict)
    ):
        loudness = store.state["loudness"]
    if loudness is None:
        loudness = normalize_loudness(
            tools,
            mix_path,
            normalized_path,
            project.audio,
            mix.total_samples,
        )
        if not loudness.get("within_tolerance"):
            raise DialogueSyncError(
                "final loudness/true-peak validation failed after normalization: "
                f"I={loudness['after']['integrated_lufs']} LUFS, "
                f"TP={loudness['after']['true_peak_dbtp']} dBTP",
                code="E_LOUDNESS",
            )
        store.state["normalized"] = {
            "path": "normalized.wav",
            "sha256": sha256_file(normalized_path),
            "samples": mix.total_samples,
        }
        store.state["loudness"] = loudness
        store.state["phase"] = "normalized"
        store.save()
    if progress:
        print(
            f"R128: {loudness['after']['integrated_lufs']:.2f} LUFS, "
            f"TP {loudness['after']['true_peak_dbtp']:.2f} dBTP",
            flush=True,
        )

    output_probe = tools.probe_audio(normalized_path)
    duration_delta_ms = abs(output_probe.duration_seconds * 1_000.0 - total_ms)
    if (
        output_probe.sample_rate != 48_000
        or output_probe.channels != project.audio.channels
        or output_probe.bit_depth != 24
    ):
        raise DialogueSyncError(
            "normalized output does not match 48 kHz/24-bit/channel profile",
            code="E_OUTPUT_FORMAT",
        )
    if duration_delta_ms > 50:
        raise DialogueSyncError(
            f"normalized audio duration error {duration_delta_ms:.3f} ms exceeds 50 ms",
            code="E_DURATION",
        )

    loudness_gain = float(loudness.get("actual_gain_db") or 0.0)
    for report in reports:
        if report.get("status") != "ok":
            continue
        report["gain_breakdown_db"]["loudness_normalization"] = round(loudness_gain, 6)
        breakdown = report["gain_breakdown_db"]
        finite = [
            float(breakdown[key])
            for key in ("entry_clip_trim", "mix_trim", "loudness_normalization")
            if math.isfinite(float(breakdown[key]))
        ]
        weight_db = breakdown["weight"]
        if math.isfinite(float(weight_db)):
            finite.append(float(weight_db))
        report["applied_gain_db"] = round(sum(finite), 6)

    subtitle_bytes = document.render().encode("utf-8")
    chapters = _chapter_groups(project, total_ms)
    packaged = _package_and_verify(
        tools,
        project,
        normalized_path,
        subtitle_bytes,
        chapters,
        output_dir,
        work_dir,
        total_ms,
    )
    store.state["packaging"] = {
        **packaged,
        "srt_sha256": sha256_bytes(subtitle_bytes),
        "report_sha256": None,
    }
    store.state["phase"] = "packaged"
    store.save()
    if progress:
        print(
            f"packaged {project.output.container_filename}; "
            f"duration error {packaged['duration_error_ms']:.3f} ms",
            flush=True,
        )

    code_counts: dict[str, int] = {}
    for report in reports:
        for code in _entry_error_codes(report):
            code_counts[code] = code_counts.get(code, 0) + 1
    for overlap in overlaps:
        code = str(overlap["code"])
        code_counts[code] = code_counts.get(code, 0) + 1
    for warning in mapping_warnings:
        code = "W_SUBTITLE_CLAMPED"
        code_counts[code] = code_counts.get(code, 0) + 1

    hard_errors = [
        report
        for report in reports
        if report.get("status") in {"skipped", "error"}
    ]
    status = "partial" if hard_errors else "ok"
    report_document: dict[str, Any] = {
        "schema_version": 1,
        "tool": "dialogue-sync",
        "status": status,
        "generated_at": _utc_now(),
        "inputs": {
            "timeline": str(project.timeline_path),
            "srt": str(project.srt_path),
            "manifest_hash": manifest_hash,
            "sha256": input_hashes,
        },
        "settings": {
            "audio": asdict(project.audio),
            "output": asdict(project.output),
            "total_duration": format_timecode(total_ms),
        },
        "summary": {
            "entries_total": len(project.entries),
            "entries_succeeded": len(successful_reports),
            "entries_skipped_or_failed": len(hard_errors),
            "issue_codes": code_counts,
            "warning_codes": sorted(
                code for code in code_counts if code.startswith("W_")
            ),
        },
        "entries": reports,
        "failures": [
            {
                "entry_index": report.get("index"),
                "error_code": report.get("error_code"),
                "message": report.get("message"),
            }
            for report in hard_errors
        ],
        "warnings": [
            *store.state.get("warnings", []),
            *store.load_warnings,
            *[
                {"code": overlap["code"], "message": overlap["message"]}
                for overlap in overlaps
            ],
            *[
                {"code": "W_SUBTITLE_CLAMPED", "message": warning}
                for warning in mapping_warnings
            ],
        ],
        "mix": mix_record,
        "loudness": loudness,
        "subtitles": {
            "output": str(output_dir / project.output.subtitle_filename),
            "cues": len(document.cues),
            "updated_cues": len({entry.subtitle_index for entry in project.entries}),
            "overlaps": overlaps,
        },
        "container": packaged,
        "outputs": {
            "audio": str(output_dir / project.output.audio_filename),
            "subtitle": str(output_dir / project.output.subtitle_filename),
            "container": str(output_dir / project.output.container_filename),
            "report": str(output_dir / project.output.report_filename),
        },
        "state": {
            "work_dir": str(work_dir),
            "phase": "complete",
            "resumable": True,
        },
    }
    for overlap in overlaps:
        # Attach warning codes to both mapped entries for easier machine handling.
        for report in reports:
            update = report.get("subtitle_update") or {}
            if update.get("index") in {
                overlap.get("first_index"),
                overlap.get("second_index"),
            }:
                if "W_SUBTITLE_OVERLAP" not in report.setdefault("warnings", []):
                    report["warnings"].append("W_SUBTITLE_OVERLAP")

    report_bytes = json.dumps(
        report_document, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")
    atomic_write_bytes(output_dir / project.output.report_filename, report_bytes)
    store.state["packaging"]["report_sha256"] = sha256_bytes(report_bytes)
    store.state["phase"] = "complete"
    store.save()
    return report_document

