from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    ASSEMBLY_VERSION,
    FORMAT_VERSION,
    RENDERER_VERSION,
    SCHEMA_VERSION,
)
from .jsonio import (
    JsonFormatError,
    atomic_write_json,
    canonical_dumps,
    load_json,
    sha256_file,
    sha256_text,
)
from .spec import ChunkSpec, RenderSpec
from .storage import (
    CheckpointValidation,
    frame_filename,
    validate_checkpoint,
)


class AssemblyError(RuntimeError):
    def __init__(self, summary: dict[str, Any]) -> None:
        super().__init__(summary.get("error", "assembly failed"))
        self.summary = summary


@dataclass
class WorkAnalysis:
    expected_frames: list[int]
    chunks: list[ChunkSpec]
    validations: dict[str, CheckpointValidation]
    corrupted_chunks: list[dict[str, Any]]
    frame_owners: dict[int, list[str]]
    missing_frames: list[int]
    duplicate_frames: list[int]
    successful_frames: int

    def summary(self, error: str = "assembly_integrity") -> dict[str, Any]:
        return {
            "status": "error",
            "error": error,
            "chunks_total": len(self.chunks),
            "valid_chunks": len(self.validations),
            "successful_frames": self.successful_frames,
            "expected_frames": len(self.expected_frames),
            "missing_frames": self.missing_frames,
            "duplicate_frames": self.duplicate_frames,
            "corrupted_chunks": self.corrupted_chunks,
        }


def analyze_workdir(
    work_dir: str | Path,
    spec: RenderSpec,
    chunk_size: int,
) -> WorkAnalysis:
    chunks = spec.chunks(chunk_size)
    expected_frames = [frame.frame for frame in spec.frames]
    validations: dict[str, CheckpointValidation] = {}
    corrupted_chunks: list[dict[str, Any]] = []
    frame_owners: dict[int, list[str]] = {}
    for chunk in chunks:
        validation = validate_checkpoint(work_dir, spec, chunk)
        if not validation.ok:
            corrupted_chunks.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "frame_start": chunk.frame_start,
                    "frame_end": chunk.frame_end,
                    "reason": validation.reason,
                }
            )
            continue
        validations[chunk.chunk_id] = validation
        if validation.manifest is None:
            corrupted_chunks.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "frame_start": chunk.frame_start,
                    "frame_end": chunk.frame_end,
                    "reason": "manifest_missing_after_validation",
                }
            )
            continue
        for entry in validation.manifest["frames"]:
            frame_owners.setdefault(entry["frame"], []).append(chunk.chunk_id)
    duplicate_frames = sorted(
        frame for frame, owners in frame_owners.items() if len(owners) > 1
    )
    missing_frames = sorted(set(expected_frames) - set(frame_owners))
    return WorkAnalysis(
        expected_frames=expected_frames,
        chunks=chunks,
        validations=validations,
        corrupted_chunks=corrupted_chunks,
        frame_owners=frame_owners,
        missing_frames=missing_frames,
        duplicate_frames=duplicate_frames,
        successful_frames=len(frame_owners),
    )


def _copy_and_hash(source: Path, destination: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(source, "rb") as source_handle, open(destination, "wb") as target_handle:
        while True:
            block = source_handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            byte_count += len(block)
            target_handle.write(block)
    return digest.hexdigest(), byte_count


def _install_directory(staging: Path, output: Path) -> None:
    if output.exists() and not output.is_dir():
        raise OSError(f"output path exists and is not a directory: {output}")
    backup: Path | None = None
    if output.exists():
        backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
        os.replace(output, backup)
    try:
        os.replace(staging, output)
    except BaseException:
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def assemble(
    work_dir: str | Path,
    output_dir: str | Path,
    spec: RenderSpec,
    chunk_size: int,
) -> dict[str, Any]:
    analysis = analyze_workdir(work_dir, spec, chunk_size)
    if analysis.corrupted_chunks or analysis.missing_frames or analysis.duplicate_frames:
        raise AssemblyError(analysis.summary())

    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.assemble-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        frame_records: list[dict[str, Any]] = []
        source_fingerprints: dict[str, str] = {}
        for chunk in sorted(analysis.chunks, key=lambda item: item.frame_start):
            validation = analysis.validations[chunk.chunk_id]
            manifest = validation.manifest
            if manifest is None or validation.artifact_dir is None:
                raise AssemblyError(
                    {
                        "status": "error",
                        "error": "assembly_internal_missing_validation",
                        "successful_frames": analysis.successful_frames,
                        "missing_frames": analysis.missing_frames,
                        "corrupted_chunks": analysis.corrupted_chunks,
                    }
                )
            source_fingerprints[chunk.chunk_id] = chunk.fingerprint
            for entry in manifest["frames"]:
                frame_number = entry["frame"]
                source = validation.artifact_dir / entry["file"]
                destination = staging / frame_filename(frame_number)
                copied_checksum, copied_size = _copy_and_hash(source, destination)
                expected_checksum = entry["sha256"][len("sha256:") :]
                if (
                    copied_checksum != expected_checksum
                    or copied_size != entry["bytes"]
                ):
                    raise AssemblyError(
                        {
                            "status": "error",
                            "error": "frame_copy_verification_failed",
                            "successful_frames": analysis.successful_frames,
                            "missing_frames": analysis.missing_frames,
                            "corrupted_chunks": [
                                {
                                    "chunk_id": chunk.chunk_id,
                                    "frame_start": chunk.frame_start,
                                    "frame_end": chunk.frame_end,
                                    "reason": f"frame_copy_mismatch:{frame_number}",
                                }
                            ],
                        }
                    )
                frame_records.append(
                    {
                        "frame": frame_number,
                        "file": frame_filename(frame_number),
                        "bytes": copied_size,
                        "sha256": f"sha256:{copied_checksum}",
                        "chunk_id": chunk.chunk_id,
                        "chunk_fingerprint": chunk.fingerprint,
                    }
                )
        frame_records.sort(key=lambda item: item["frame"])
        sequence_digest = sha256_text(canonical_dumps(frame_records))
        manifest = {
            "format_version": FORMAT_VERSION,
            "assembly_version": ASSEMBLY_VERSION,
            "schema_version": SCHEMA_VERSION,
            "renderer_version": RENDERER_VERSION,
            "frame_start": spec.frame_start,
            "frame_end": spec.frame_end,
            "frame_count": len(frame_records),
            "width": spec.width,
            "height": spec.height,
            "source_fingerprints": source_fingerprints,
            "frames": frame_records,
            "sequence_digest": f"sha256:{sequence_digest}",
        }
        atomic_write_json(staging / "sequence_manifest.json", manifest)
        verification = verify_sequence(staging)
        if not verification["ok"]:
            raise AssemblyError(
                {
                    "status": "error",
                    "error": "assembled_sequence_verification_failed",
                    "successful_frames": analysis.successful_frames,
                    "missing_frames": analysis.missing_frames,
                    "corrupted_chunks": analysis.corrupted_chunks,
                    "sequence_errors": verification.get("errors", []),
                }
            )
        _install_directory(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "ok",
        "command": "assemble",
        "frame_start": spec.frame_start,
        "frame_end": spec.frame_end,
        "successful_frames": len(frame_records),
        "frames": len(frame_records),
        "chunks": len(analysis.chunks),
        "output": str(output),
        "sequence_digest": f"sha256:{sequence_digest}",
    }


def verify_sequence(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    manifest_file = output / "sequence_manifest.json"
    errors: list[str] = []
    if not manifest_file.is_file():
        return {"ok": False, "errors": ["sequence_manifest_missing"]}
    try:
        manifest = load_json(manifest_file)
    except (OSError, JsonFormatError) as exc:
        return {"ok": False, "errors": [f"sequence_manifest_unreadable:{exc}"]}
    if not isinstance(manifest, dict):
        return {"ok": False, "errors": ["sequence_manifest_not_object"]}
    entries = manifest.get("frames")
    if not isinstance(entries, list):
        return {"ok": False, "errors": ["sequence_frames_not_array"]}
    frame_start = manifest.get("frame_start")
    frame_end = manifest.get("frame_end")
    if (
        not isinstance(frame_start, int)
        or not isinstance(frame_end, int)
        or frame_end < frame_start
    ):
        errors.append("sequence_frame_range_invalid")
        frame_start = 0
        frame_end = -1
    expected = list(range(frame_start, frame_end + 1))
    seen: list[int] = []
    normalized_entries: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("sequence_frame_entry_not_object")
            continue
        frame_number = entry.get("frame")
        if not isinstance(frame_number, int):
            errors.append("sequence_frame_number_invalid")
            continue
        seen.append(frame_number)
        file_name = entry.get("file")
        if file_name != frame_filename(frame_number):
            errors.append(f"sequence_frame_name_invalid:{frame_number}")
            continue
        file_path = output / file_name
        if not file_path.is_file():
            errors.append(f"sequence_frame_missing:{frame_number}")
            continue
        checksum = entry.get("sha256")
        if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
            errors.append(f"sequence_frame_checksum_invalid:{frame_number}")
            continue
        if sha256_file(file_path) != checksum[len("sha256:") :]:
            errors.append(f"sequence_frame_checksum_mismatch:{frame_number}")
        if file_path.stat().st_size != entry.get("bytes"):
            errors.append(f"sequence_frame_size_mismatch:{frame_number}")
        normalized_entries.append(
            {
                "frame": frame_number,
                "file": file_name,
                "bytes": entry.get("bytes"),
                "sha256": checksum,
                "chunk_id": entry.get("chunk_id"),
                "chunk_fingerprint": entry.get("chunk_fingerprint"),
            }
        )
    if sorted(seen) != expected:
        errors.append("sequence_frame_numbers_not_continuous")
    if len(seen) != len(set(seen)):
        errors.append("sequence_duplicate_frame_numbers")
    normalized_entries.sort(key=lambda item: item["frame"])
    expected_digest = sha256_text(canonical_dumps(normalized_entries))
    if manifest.get("sequence_digest") != f"sha256:{expected_digest}":
        errors.append("sequence_digest_mismatch")
    return {
        "ok": not errors,
        "errors": errors,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_count": len(seen),
        "sequence_digest": manifest.get("sequence_digest"),
    }

