from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any

from .constants import (
    FAILURE_REPORT_VERSION,
    FORMAT_VERSION,
    RENDERER_VERSION,
    SCHEMA_VERSION,
)
from .jsonio import (
    JsonFormatError,
    atomic_write_json,
    json_ready,
    load_json,
    sha256_file,
)
from .renderer import RayScene, save_png
from .spec import ChunkSpec, RenderSpec, load_spec
from .storage import frame_filename, write_checkpoint


class WorkerError(RuntimeError):
    pass


class InjectedRecoverableError(RuntimeError):
    pass


def _describe_fault(chunk_id: str, attempt: int) -> dict[str, Any] | None:
    raw = os.environ.get("OFR_TEST_FAULT")
    if not raw:
        return None
    try:
        fault = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(fault, dict) or fault.get("chunk_id") != chunk_id:
        return None
    expected_attempt = fault.get("attempt")
    if expected_attempt is not None and expected_attempt != attempt:
        return None
    return fault


def _manifest_frame_entry(staging: Path, frame_number: int) -> dict[str, Any]:
    relative = Path("frames") / frame_filename(frame_number)
    image_path = staging / relative
    return {
        "frame": frame_number,
        "file": relative.as_posix(),
        "bytes": image_path.stat().st_size,
        "sha256": f"sha256:{sha256_file(image_path)}",
    }


def render_task(task_path: str | Path, result_path: str | Path) -> int:
    task_file = Path(task_path).resolve()
    result_file = Path(result_path).resolve()
    completed_frames: list[int] = []
    task: dict[str, Any] | None = None
    staging: Path | None = None
    failure_report: Path | None = None

    try:
        task = load_json(task_file)
        if not isinstance(task, dict):
            raise WorkerError("task must be a JSON object")
        if task.get("format_version") != FORMAT_VERSION:
            raise WorkerError("task format version mismatch")
        failure_report = Path(task["failure_report_path"]).resolve()
        failure_report.parent.mkdir(parents=True, exist_ok=True)
        result_file.parent.mkdir(parents=True, exist_ok=True)
        spec = load_spec(task["spec_path"])
        chunk = spec.chunk_for_range(task["frame_start"], task["frame_end"])
        if task.get("chunk_id") != chunk.chunk_id:
            raise WorkerError("task chunk id does not match frame range")
        if task.get("fingerprint") != chunk.fingerprint:
            raise WorkerError("task input fingerprint changed before rendering")
        if task.get("geometry_digest") != chunk.geometry_digest:
            raise WorkerError("task geometry digest changed before rendering")
        if task.get("sampling_digest") != chunk.sampling_digest:
            raise WorkerError("task sampling digest changed before rendering")
        attempt = int(task.get("attempt", 1))
        target_artifact = Path(task["artifact_dir"]).resolve()
        target_checkpoint = Path(task["checkpoint_path"]).resolve()
        target_artifact.parent.mkdir(parents=True, exist_ok=True)
        target_checkpoint.parent.mkdir(parents=True, exist_ok=True)

        for stale in target_artifact.parent.glob(f".{chunk.chunk_id}.partial-*"):
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
        if target_artifact.exists():
            if target_artifact.is_dir():
                shutil.rmtree(target_artifact)
            else:
                target_artifact.unlink()

        staging = target_artifact.parent / (
            f".{chunk.chunk_id}.partial-{os.getpid()}-{uuid.uuid4().hex}"
        )
        (staging / "frames").mkdir(parents=True, exist_ok=False)

        fault = _describe_fault(chunk.chunk_id, attempt)
        fault_after = int(fault.get("after_frames", 0)) if fault else -1
        ray_scene = RayScene(spec)
        entries: list[dict[str, Any]] = []
        for frame in chunk.frames:
            image = ray_scene.render_frame(frame)
            save_png(image, staging / "frames" / frame_filename(frame.frame))
            entries.append(_manifest_frame_entry(staging, frame.frame))
            completed_frames.append(frame.frame)
            if fault and len(completed_frames) >= fault_after:
                if fault.get("mode") == "hard_exit":
                    os._exit(int(fault.get("exit_code", 23)))
                if fault.get("mode") == "recoverable":
                    raise InjectedRecoverableError(
                        str(fault.get("reason", "injected recoverable worker failure"))
                    )

        manifest = {
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
            "width": spec.width,
            "height": spec.height,
            "frames": entries,
        }
        atomic_write_json(staging / "manifest.json", manifest)
        if target_artifact.exists():
            shutil.rmtree(target_artifact)
        os.replace(staging, target_artifact)
        staging = None
        checksum = sha256_file(target_artifact / "manifest.json")
        checkpoint = write_checkpoint(
            Path(task["work_dir"]), spec, chunk, checksum
        )
        result = {
            "status": "ok",
            "chunk_id": chunk.chunk_id,
            "fingerprint": chunk.fingerprint,
            "frame_start": chunk.frame_start,
            "frame_end": chunk.frame_end,
            "attempt": attempt,
            "frames_written": len(entries),
            "artifact_checksum": f"sha256:{checksum}",
            "checkpoint": str(checkpoint),
        }
        atomic_write_json(result_file, result)
        return 0
    except BaseException as exc:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        reason = f"{type(exc).__name__}: {exc}"
        report = {
            "report_version": FAILURE_REPORT_VERSION,
            "status": "failed",
            "recoverable": True,
            "chunk_id": task.get("chunk_id") if isinstance(task, dict) else None,
            "fingerprint": task.get("fingerprint") if isinstance(task, dict) else None,
            "attempt": task.get("attempt") if isinstance(task, dict) else None,
            "reason": reason,
            "completed_frames": completed_frames,
            "chunk_completed": False,
            "completed_chunks": task.get("completed_chunks_before_task", [])
            if isinstance(task, dict)
            else [],
            "traceback": traceback.format_exc(),
        }
        if failure_report is not None:
            try:
                atomic_write_json(failure_report, json_ready(report))
            except OSError:
                pass
        try:
            atomic_write_json(
                result_file,
                {
                    "status": "failed",
                    "recoverable": True,
                    "chunk_id": report["chunk_id"],
                    "attempt": report["attempt"],
                    "reason": reason,
                    "completed_frames": completed_frames,
                    "failure_report": str(failure_report) if failure_report else None,
                },
            )
        except OSError:
            pass
        print(reason, file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render one frame chunk")
    parser.add_argument("task", help="task JSON path")
    parser.add_argument("result", help="result JSON path")
    args = parser.parse_args(argv)
    return render_task(args.task, args.result)


if __name__ == "__main__":
    raise SystemExit(main())

