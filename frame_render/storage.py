from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    CHECKPOINT_VERSION,
    FORMAT_VERSION,
    RENDERER_VERSION,
    SCHEMA_VERSION,
)
from .jsonio import (
    JsonFormatError,
    atomic_write_json,
    load_json,
    sha256_file,
)
from .spec import ChunkSpec, RenderSpec


def checkpoint_path(work_dir: str | Path, chunk: ChunkSpec) -> Path:
    return Path(work_dir) / "checkpoints" / f"{chunk.chunk_id}.json"


def artifact_dir(work_dir: str | Path, chunk: ChunkSpec) -> Path:
    return Path(work_dir) / "chunks" / chunk.chunk_id


def manifest_path(work_dir: str | Path, chunk: ChunkSpec) -> Path:
    return artifact_dir(work_dir, chunk) / "manifest.json"


def task_path(work_dir: str | Path, chunk: ChunkSpec, attempt: int) -> Path:
    return Path(work_dir) / "tasks" / f"{chunk.chunk_id}_attempt_{attempt:03d}.json"


def result_path(work_dir: str | Path, chunk: ChunkSpec, attempt: int) -> Path:
    return Path(work_dir) / "results" / f"{chunk.chunk_id}_attempt_{attempt:03d}.json"


def failure_report_path(work_dir: str | Path, chunk: ChunkSpec, attempt: int) -> Path:
    return Path(work_dir) / "reports" / f"{chunk.chunk_id}_attempt_{attempt:03d}.json"


def worker_log_path(work_dir: str | Path, chunk: ChunkSpec, attempt: int) -> Path:
    return Path(work_dir) / "logs" / f"{chunk.chunk_id}_attempt_{attempt:03d}.log"


def frame_filename(frame_number: int) -> str:
    return f"frame_{frame_number:06d}.png"


@dataclass(frozen=True)
class CheckpointValidation:
    ok: bool
    reason: str
    record: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    checkpoint_file: Path | None = None
    artifact_dir: Path | None = None


def _failure(reason: str, checkpoint_file: Path | None = None) -> CheckpointValidation:
    return CheckpointValidation(
        ok=False,
        reason=reason,
        checkpoint_file=checkpoint_file,
    )


def _binding(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ValueError(f"{label} is not a sha256 binding")
    digest = value[len("sha256:") :]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} has an invalid digest")
    return digest


def validate_checkpoint(
    work_dir: str | Path,
    spec: RenderSpec,
    chunk: ChunkSpec,
    *,
    verify_artifacts: bool = True,
) -> CheckpointValidation:
    record_file = checkpoint_path(work_dir, chunk)
    if not record_file.is_file():
        return _failure("checkpoint_missing", record_file)
    try:
        record = load_json(record_file)
    except (OSError, JsonFormatError) as exc:
        return _failure(f"checkpoint_unreadable:{exc}", record_file)
    if not isinstance(record, dict):
        return _failure("checkpoint_not_object", record_file)

    expected_manifest_rel = (
        Path("chunks") / chunk.chunk_id / "manifest.json"
    ).as_posix()
    checks = (
        ("checkpoint_version", CHECKPOINT_VERSION),
        ("format_version", FORMAT_VERSION),
        ("schema_version", SCHEMA_VERSION),
        ("renderer_version", RENDERER_VERSION),
        ("chunk_id", chunk.chunk_id),
        ("frame_start", chunk.frame_start),
        ("frame_end", chunk.frame_end),
        ("fingerprint", chunk.fingerprint),
        ("geometry_digest", chunk.geometry_digest),
        ("sampling_digest", chunk.sampling_digest),
        ("artifact_manifest", expected_manifest_rel),
    )
    for key, expected in checks:
        actual = record.get(key)
        if actual != expected:
            return _failure(f"checkpoint_field_mismatch:{key}", record_file)
    try:
        artifact_checksum = _binding(record.get("artifact_checksum"), "artifact_checksum")
    except ValueError as exc:
        return _failure(f"checkpoint_bad_checksum:{exc}", record_file)

    chunk_dir = artifact_dir(work_dir, chunk)
    if not chunk_dir.is_dir():
        return _failure("artifact_missing", record_file)
    manifest_file = chunk_dir / "manifest.json"
    if not manifest_file.is_file():
        return _failure("manifest_missing", record_file)
    try:
        if sha256_file(manifest_file) != artifact_checksum:
            return _failure("manifest_checksum_mismatch", record_file)
        manifest = load_json(manifest_file)
    except (OSError, JsonFormatError) as exc:
        return _failure(f"manifest_unreadable:{exc}", record_file)
    if not isinstance(manifest, dict):
        return _failure("manifest_not_object", record_file)

    manifest_checks = (
        ("format_version", FORMAT_VERSION),
        ("schema_version", SCHEMA_VERSION),
        ("renderer_version", RENDERER_VERSION),
        ("chunk_id", chunk.chunk_id),
        ("frame_start", chunk.frame_start),
        ("frame_end", chunk.frame_end),
        ("fingerprint", chunk.fingerprint),
        ("geometry_digest", chunk.geometry_digest),
        ("sampling_digest", chunk.sampling_digest),
        ("width", spec.width),
        ("height", spec.height),
    )
    for key, expected in manifest_checks:
        if manifest.get(key) != expected:
            return _failure(f"manifest_field_mismatch:{key}", record_file)

    entries = manifest.get("frames")
    if not isinstance(entries, list):
        return _failure("manifest_frames_not_array", record_file)
    expected_frames = [frame.frame for frame in chunk.frames]
    if len(entries) != len(expected_frames):
        return _failure("manifest_frame_count_mismatch", record_file)
    seen: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            return _failure("manifest_frame_entry_not_object", record_file)
        frame_number = entry.get("frame")
        if not isinstance(frame_number, int) or isinstance(frame_number, bool):
            return _failure("manifest_frame_number_invalid", record_file)
        if frame_number in seen:
            return _failure(f"manifest_duplicate_frame:{frame_number}", record_file)
        seen.add(frame_number)
        if entry.get("file") != f"frames/{frame_filename(frame_number)}":
            return _failure(f"manifest_frame_path_invalid:{frame_number}", record_file)
        try:
            expected_checksum = _binding(
                entry.get("sha256"), f"frame {frame_number} sha256"
            )
        except ValueError as exc:
            return _failure(f"manifest_frame_checksum_invalid:{exc}", record_file)
        byte_count = entry.get("bytes")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
            return _failure(f"manifest_frame_size_invalid:{frame_number}", record_file)
        if verify_artifacts:
            image_file = chunk_dir / "frames" / frame_filename(frame_number)
            if not image_file.is_file():
                return _failure(f"frame_file_missing:{frame_number}", record_file)
            try:
                if image_file.stat().st_size != byte_count:
                    return _failure(
                        f"frame_size_mismatch:{frame_number}", record_file
                    )
                if sha256_file(image_file) != expected_checksum:
                    return _failure(
                        f"frame_checksum_mismatch:{frame_number}", record_file
                    )
            except OSError as exc:
                return _failure(f"frame_unreadable:{frame_number}:{exc}", record_file)
    if seen != set(expected_frames):
        return _failure("manifest_frame_set_mismatch", record_file)

    return CheckpointValidation(
        ok=True,
        reason="ok",
        record=record,
        manifest=manifest,
        checkpoint_file=record_file,
        artifact_dir=chunk_dir,
    )


def write_checkpoint(
    work_dir: str | Path,
    spec: RenderSpec,
    chunk: ChunkSpec,
    artifact_checksum: str,
) -> Path:
    target = checkpoint_path(work_dir, chunk)
    record = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "format_version": FORMAT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "renderer_version": RENDERER_VERSION,
        "chunk_id": chunk.chunk_id,
        "frame_start": chunk.frame_start,
        "frame_end": chunk.frame_end,
        "frame_count": len(chunk.frames),
        "fingerprint": chunk.fingerprint,
        "geometry_digest": chunk.geometry_digest,
        "sampling_digest": chunk.sampling_digest,
        "artifact_manifest": (
            Path("chunks") / chunk.chunk_id / "manifest.json"
        ).as_posix(),
        "artifact_checksum": f"sha256:{artifact_checksum}",
    }
    atomic_write_json(target, record)
    return target


def cleanup_partial_artifacts(work_dir: str | Path, chunk: ChunkSpec) -> int:
    chunks_dir = Path(work_dir) / "chunks"
    if not chunks_dir.is_dir():
        return 0
    removed = 0
    pattern = f".{chunk.chunk_id}.partial-*"
    for candidate in chunks_dir.glob(pattern):
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)
            removed += 1
    return removed


def cleanup_all_partials(work_dir: str | Path) -> int:
    chunks_dir = Path(work_dir) / "chunks"
    if not chunks_dir.is_dir():
        return 0
    removed = 0
    for candidate in chunks_dir.glob(".*.partial-*"):
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)
            removed += 1
    return removed

