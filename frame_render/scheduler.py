from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import FORMAT_VERSION, RENDERER_VERSION, SCHEMA_VERSION, STATE_VERSION
from .jsonio import JsonFormatError, atomic_write_json, load_json
from .spec import ChunkSpec, RenderSpec
from .storage import (
    artifact_dir,
    checkpoint_path,
    cleanup_all_partials,
    cleanup_partial_artifacts,
    failure_report_path,
    result_path,
    task_path,
    validate_checkpoint,
    worker_log_path,
)


@dataclass
class ChunkState:
    fingerprint: str
    frame_start: int
    frame_end: int
    attempts: int = 0
    retry_count: int = 0
    last_status: str = "pending"
    last_reason: str = ""
    completed: bool = False
    failure_reports: list[str] = field(default_factory=list)


@dataclass
class ActiveJob:
    chunk: ChunkSpec
    attempt: int
    process: subprocess.Popen
    log_handle: Any
    result_file: Path
    report_file: Path
    log_file: Path


class SchedulerInterrupted(RuntimeError):
    def __init__(self, summary: dict[str, Any]) -> None:
        super().__init__("scheduler interrupted")
        self.summary = summary


class Scheduler:
    def __init__(
        self,
        spec: RenderSpec,
        work_dir: str | Path,
        *,
        chunk_size: int,
        jobs: int,
        max_retries: int = 1,
        verbose: bool = False,
    ) -> None:
        if jobs <= 0:
            raise ValueError("jobs must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.spec = spec
        self.work_dir = Path(work_dir).resolve()
        self.chunk_size = chunk_size
        self.jobs = jobs
        self.max_retries = max_retries
        self.verbose = verbose
        self.chunks = spec.chunks(chunk_size)
        self.state_file = self.work_dir / "state" / "scheduler_state.json"
        self.state_warning = ""
        self.chunk_state: dict[str, ChunkState] = {}
        self.invalid_checkpoints: list[dict[str, Any]] = []
        self.reused_chunks: list[str] = []
        self.rendered_chunks: list[str] = []
        self.recoverable_failures: list[dict[str, Any]] = []
        self.forced_terminations: list[dict[str, Any]] = []
        self.partial_dirs_removed = 0
        self._load_state()

    def _load_state(self) -> None:
        raw: Any = None
        if self.state_file.is_file():
            try:
                raw = load_json(self.state_file)
            except (OSError, JsonFormatError) as exc:
                self.state_warning = f"state_unreadable:{exc}"
        if (
            not isinstance(raw, dict)
            or raw.get("state_version") != STATE_VERSION
            or raw.get("renderer_version") != RENDERER_VERSION
            or not isinstance(raw.get("chunks"), dict)
        ):
            if raw is not None:
                self.state_warning = self.state_warning or "state_version_or_shape_mismatch"
            raw = {"chunks": {}}
        loaded = raw["chunks"]
        for chunk in self.chunks:
            existing = loaded.get(chunk.chunk_id)
            same_version = (
                isinstance(existing, dict)
                and existing.get("fingerprint") == chunk.fingerprint
                and existing.get("frame_start") == chunk.frame_start
                and existing.get("frame_end") == chunk.frame_end
            )
            if not same_version:
                self.chunk_state[chunk.chunk_id] = ChunkState(
                    fingerprint=chunk.fingerprint,
                    frame_start=chunk.frame_start,
                    frame_end=chunk.frame_end,
                )
                continue
            reports = existing.get("failure_reports", [])
            self.chunk_state[chunk.chunk_id] = ChunkState(
                fingerprint=chunk.fingerprint,
                frame_start=chunk.frame_start,
                frame_end=chunk.frame_end,
                attempts=max(0, int(existing.get("attempts", 0))),
                retry_count=max(0, int(existing.get("retry_count", 0))),
                last_status=str(existing.get("last_status", "pending")),
                last_reason=str(existing.get("last_reason", "")),
                completed=False,
                failure_reports=[str(item) for item in reports]
                if isinstance(reports, list)
                else [],
            )

    def _state_document(self) -> dict[str, Any]:
        return {
            "state_version": STATE_VERSION,
            "format_version": FORMAT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "renderer_version": RENDERER_VERSION,
            "chunks": {
                chunk_id: {
                    "fingerprint": state.fingerprint,
                    "frame_start": state.frame_start,
                    "frame_end": state.frame_end,
                    "attempts": state.attempts,
                    "retry_count": state.retry_count,
                    "last_status": state.last_status,
                    "last_reason": state.last_reason,
                    "completed": state.completed,
                    "failure_reports": state.failure_reports,
                }
                for chunk_id, state in sorted(self.chunk_state.items())
            },
        }

    def _persist_state(self) -> None:
        atomic_write_json(self.state_file, self._state_document())

    def _log(self, message: str) -> None:
        log_file = self.work_dir / "scheduler.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")
        if self.verbose:
            print(message, file=sys.stderr)

    def _validate_existing(self) -> list[ChunkSpec]:
        pending: list[ChunkSpec] = []
        for chunk in self.chunks:
            validation = validate_checkpoint(self.work_dir, self.spec, chunk)
            state = self.chunk_state[chunk.chunk_id]
            if validation.ok:
                state.completed = True
                state.last_status = "reused"
                state.last_reason = "valid_checkpoint"
                self.reused_chunks.append(chunk.chunk_id)
                continue
            state.completed = False
            state.last_status = "pending"
            state.last_reason = validation.reason
            record_exists = (
                validation.checkpoint_file is not None
                and validation.checkpoint_file.exists()
            )
            artifact_exists = artifact_dir(self.work_dir, chunk).exists()
            if record_exists or artifact_exists:
                self.invalid_checkpoints.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "frame_start": chunk.frame_start,
                        "frame_end": chunk.frame_end,
                        "reason": validation.reason,
                    }
                )
            pending.append(chunk)
        return pending

    def _task_document(self, chunk: ChunkSpec, attempt: int) -> dict[str, Any]:
        return {
            "task_version": 1,
            "format_version": FORMAT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "renderer_version": RENDERER_VERSION,
            "spec_path": str(self.spec.source_path),
            "work_dir": str(self.work_dir),
            "chunk_id": chunk.chunk_id,
            "frame_start": chunk.frame_start,
            "frame_end": chunk.frame_end,
            "fingerprint": chunk.fingerprint,
            "geometry_digest": chunk.geometry_digest,
            "sampling_digest": chunk.sampling_digest,
            "attempt": attempt,
            "completed_chunks_before_task": [
                item.chunk_id
                for item in self.chunks
                if self.chunk_state[item.chunk_id].completed
            ],
            "artifact_dir": str(artifact_dir(self.work_dir, chunk)),
            "checkpoint_path": str(checkpoint_path(self.work_dir, chunk)),
            "result_path": str(
                result_path(self.work_dir, chunk, attempt).resolve()
            ),
            "failure_report_path": str(
                failure_report_path(self.work_dir, chunk, attempt).resolve()
            ),
        }

    def _launch(
        self, chunk: ChunkSpec, run_attempts: dict[str, int]
    ) -> ActiveJob | None:
        state = self.chunk_state[chunk.chunk_id]
        attempt = state.attempts + 1
        run_attempts[chunk.chunk_id] = run_attempts.get(chunk.chunk_id, 0) + 1
        state.attempts = attempt
        state.retry_count = max(0, attempt - 1)
        state.last_status = "running"
        state.last_reason = ""
        self.partial_dirs_removed += cleanup_partial_artifacts(
            self.work_dir, chunk
        )
        task_file = task_path(self.work_dir, chunk, attempt)
        result_file = result_path(self.work_dir, chunk, attempt).resolve()
        report_file = failure_report_path(self.work_dir, chunk, attempt).resolve()
        log_file = worker_log_path(self.work_dir, chunk, attempt)
        result_file.unlink(missing_ok=True)
        report_file.unlink(missing_ok=True)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(task_file, self._task_document(chunk, attempt))
        self._persist_state()

        package_root = str(Path(__file__).resolve().parent.parent)
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            package_root
            if not existing_pythonpath
            else package_root + os.pathsep + existing_pythonpath
        )
        command = [
            sys.executable,
            "-m",
            "frame_render.worker",
            str(task_file),
            str(result_file),
        ]
        log_handle = open(log_file, "w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                cwd=package_root,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name != "nt"),
            )
        except BaseException:
            log_handle.close()
            raise
        self._log(
            f"launch {chunk.chunk_id} attempt={attempt} pid={process.pid}"
        )
        return ActiveJob(
            chunk=chunk,
            attempt=attempt,
            process=process,
            log_handle=log_handle,
            result_file=result_file,
            report_file=report_file,
            log_file=log_file,
        )

    def _read_optional_json(self, path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            value = load_json(path)
        except (OSError, JsonFormatError):
            return None
        return value if isinstance(value, dict) else None

    def _finish_job(
        self,
        job: ActiveJob,
        return_code: int,
        pending: deque[ChunkSpec],
        run_attempts: dict[str, int],
    ) -> None:
        job.log_handle.close()
        validation = validate_checkpoint(
            self.work_dir, self.spec, job.chunk
        )
        state = self.chunk_state[job.chunk.chunk_id]
        if validation.ok:
            state.completed = True
            state.last_status = "completed"
            state.last_reason = "valid_checkpoint"
            self.rendered_chunks.append(job.chunk.chunk_id)
            self._log(
                f"complete {job.chunk.chunk_id} attempt={job.attempt} rc={return_code}"
            )
            self._persist_state()
            return

        report = self._read_optional_json(job.report_file)
        result = self._read_optional_json(job.result_file)
        if report is not None:
            reason = str(report.get("reason", "worker_reported_failure"))
            state.last_status = "recoverable_failure"
            state.failure_reports.append(str(job.report_file))
            self.recoverable_failures.append(
                {
                    "chunk_id": job.chunk.chunk_id,
                    "attempt": job.attempt,
                    "reason": reason,
                    "report": str(job.report_file),
                    "completed_frames": report.get("completed_frames", []),
                }
            )
        elif result is not None and result.get("status") == "failed":
            reason = str(result.get("reason", "worker_result_failed"))
            state.last_status = "recoverable_failure"
        else:
            reason = (
                f"forced_exit_without_report:return_code={return_code};"
                f"checkpoint={validation.reason}"
            )
            state.last_status = "forced_termination"
            self.forced_terminations.append(
                {
                    "chunk_id": job.chunk.chunk_id,
                    "attempt": job.attempt,
                    "return_code": return_code,
                    "checkpoint_validation": validation.reason,
                    "log": str(job.log_file),
                }
            )
        state.last_reason = reason
        state.completed = False
        self._log(
            f"failure {job.chunk.chunk_id} attempt={job.attempt} "
            f"rc={return_code} reason={reason}"
        )
        self._persist_state()
        if run_attempts.get(job.chunk.chunk_id, 0) < 1 + self.max_retries:
            pending.append(job.chunk)

    def _terminate_active(self, active: dict[str, ActiveJob]) -> None:
        for job in list(active.values()):
            if job.process.poll() is None:
                job.process.terminate()
        deadline = time.monotonic() + 3.0
        for job in list(active.values()):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                job.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                job.process.kill()
                job.process.wait(timeout=3.0)
            job.log_handle.close()

    def _summary(self, status: str, started: float) -> dict[str, Any]:
        unfinished = [
            {
                "chunk_id": chunk.chunk_id,
                "frame_start": chunk.frame_start,
                "frame_end": chunk.frame_end,
                "attempts": self.chunk_state[chunk.chunk_id].attempts,
                "retry_count": self.chunk_state[chunk.chunk_id].retry_count,
                "last_status": self.chunk_state[chunk.chunk_id].last_status,
                "last_reason": self.chunk_state[chunk.chunk_id].last_reason,
            }
            for chunk in self.chunks
            if not self.chunk_state[chunk.chunk_id].completed
        ]
        attempts = {
            chunk.chunk_id: self.chunk_state[chunk.chunk_id].attempts
            for chunk in self.chunks
            if self.chunk_state[chunk.chunk_id].attempts
        }
        retry_counts = {
            chunk.chunk_id: self.chunk_state[chunk.chunk_id].retry_count
            for chunk in self.chunks
            if self.chunk_state[chunk.chunk_id].retry_count
        }
        return {
            "status": status,
            "command": "render",
            "frame_start": self.spec.frame_start,
            "frame_end": self.spec.frame_end,
            "chunk_size": self.chunk_size,
            "jobs": self.jobs,
            "chunks_total": len(self.chunks),
            "chunks_completed": sum(
                1
                for chunk in self.chunks
                if self.chunk_state[chunk.chunk_id].completed
            ),
            "chunks_rendered": len(self.rendered_chunks),
            "chunks_reused": len(self.reused_chunks),
            "completed_chunks": [
                chunk.chunk_id
                for chunk in self.chunks
                if self.chunk_state[chunk.chunk_id].completed
            ],
            "rendered_chunks": self.rendered_chunks,
            "reused_chunks": self.reused_chunks,
            "invalid_checkpoints": self.invalid_checkpoints,
            "unfinished_chunks": unfinished,
            "attempts": attempts,
            "retry_counts": retry_counts,
            "recoverable_failures": self.recoverable_failures,
            "forced_terminations": self.forced_terminations,
            "partial_dirs_removed": self.partial_dirs_removed,
            "state_warning": self.state_warning,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }

    def run(self) -> dict[str, Any]:
        started = time.monotonic()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.partial_dirs_removed += cleanup_all_partials(self.work_dir)
        pending = deque(self._validate_existing())
        self._persist_state()
        active: dict[str, ActiveJob] = {}
        run_attempts: dict[str, int] = {}
        interrupted = False
        try:
            while pending or active:
                while pending and len(active) < self.jobs:
                    chunk = pending.popleft()
                    if run_attempts.get(chunk.chunk_id, 0) >= 1 + self.max_retries:
                        continue
                    try:
                        job = self._launch(chunk, run_attempts)
                    except BaseException as exc:
                        state = self.chunk_state[chunk.chunk_id]
                        state.last_status = "launch_failure"
                        state.last_reason = f"{type(exc).__name__}: {exc}"
                        self._persist_state()
                        self._log(
                            f"launch failure {chunk.chunk_id}: {state.last_reason}"
                        )
                        if run_attempts.get(chunk.chunk_id, 0) < 1 + self.max_retries:
                            pending.append(chunk)
                        continue
                    active[chunk.chunk_id] = job

                progressed = False
                for chunk_id, job in list(active.items()):
                    return_code = job.process.poll()
                    if return_code is None:
                        continue
                    del active[chunk_id]
                    self._finish_job(job, return_code, pending, run_attempts)
                    progressed = True
                if not progressed and active:
                    time.sleep(0.03)
        except KeyboardInterrupt:
            interrupted = True
            self._terminate_active(active)
            active.clear()
            self._persist_state()
        except BaseException:
            self._terminate_active(active)
            active.clear()
            self._persist_state()
            raise

        all_complete = all(
            self.chunk_state[chunk.chunk_id].completed for chunk in self.chunks
        )
        status = "interrupted" if interrupted else ("ok" if all_complete else "incomplete")
        summary = self._summary(status, started)
        self._persist_state()
        atomic_write_json(
            self.work_dir / "reports" / "last_run_summary.json", summary
        )
        if interrupted:
            raise SchedulerInterrupted(summary)
        return summary

