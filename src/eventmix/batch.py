"""Batch discovery, deterministic scheduling, checkpoint recovery, and reports."""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .checkpoint import CheckpointStore
from .engine import EngineConfig, build_task, task_report_from_result
from .errors import ConfigError, EventMixError, OutputCollisionError
from .locking import exclusive_output_locks
from .resample import SincConfig
from .resources import RenderLimits, ResourceGovernor
from .timeline import ProjectSpec, load_json_object, parse_project
from .util import canonical_json_bytes, display_path, sha256_bytes, sha256_file
from .version import BATCH_SCHEMA, PLAYBACK_SCHEMA, REPORT_SCHEMA, TOOL_VERSION
from .wav import read_wav_metadata


@dataclass(frozen=True)
class BatchJob:
    index: int
    id: str
    timeline_path: Path
    wav_path: Path
    timeline_output_path: Path
    overrides: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchOptions:
    output_dir: Path
    checkpoint_path: Path
    resume: bool = True
    jobs: int = 1
    max_open_wav: int = 8
    max_memory_mb: int = 256
    chunk_frames: int = 8192
    sinc_half_width: int = 16
    sinc_max_half_width: int = 64
    on_track_error: str | None = None
    bit_depth: int | None = None
    target_sample_rate: int | None = None
    silence_threshold_dbfs: float | None = None
    min_silence_seconds: float | None = None
    wav_name: str | None = None
    timeline_output_name: str | None = None


def _safe_job_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("._-")
    if cleaned:
        return cleaned
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"job-{suffix}"


def _resolve_under(base: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _job_value(job_raw: Mapping[str, Any], defaults: Mapping[str, Any], key: str) -> Any:
    if key in job_raw:
        return job_raw[key]
    nested_policy = job_raw.get("policy")
    if isinstance(nested_policy, dict) and key in nested_policy:
        return nested_policy[key]
    nested_engine = job_raw.get("engine")
    if isinstance(nested_engine, dict) and key in nested_engine:
        return nested_engine[key]
    if key in defaults:
        return defaults[key]
    nested_default_policy = defaults.get("policy")
    if isinstance(nested_default_policy, dict) and key in nested_default_policy:
        return nested_default_policy[key]
    nested_default_engine = defaults.get("engine")
    if isinstance(nested_default_engine, dict) and key in nested_default_engine:
        return nested_default_engine[key]
    return None


_JOB_OVERRIDE_KEYS = {
    "on_track_error",
    "bit_depth",
    "target_sample_rate",
    "silence_threshold_dbfs",
    "min_silence_seconds",
    "chunk_frames",
}


def load_jobs(
    input_path: Path | str,
    output_dir: Path | str,
    *,
    wav_name: str | None = None,
    timeline_output_name: str | None = None,
) -> tuple[list[BatchJob], dict[str, Any], dict[str, Any]]:
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_json_object(input_path, what="batch manifest or timeline")
    jobs: list[BatchJob] = []
    defaults: dict[str, Any] = {}
    if isinstance(raw.get("jobs"), list):
        schema = raw.get("schema_version")
        if schema not in (None, BATCH_SCHEMA):
            raise ConfigError(f"unsupported batch schema_version: {schema!r}")
        defaults_raw = raw.get("defaults", {})
        if defaults_raw is not None and not isinstance(defaults_raw, dict):
            raise ConfigError("batch defaults must be an object")
        defaults = dict(defaults_raw or {})
        for index, item in enumerate(raw["jobs"]):
            if not isinstance(item, dict):
                raise ConfigError(f"jobs[{index}] must be an object")
            job_id = item.get("id")
            if not isinstance(job_id, str) or not job_id:
                raise ConfigError(f"jobs[{index}].id must be a non-empty string")
            timeline_value = item.get("timeline", item.get("input"))
            if not isinstance(timeline_value, str) or not timeline_value:
                raise ConfigError(f"jobs[{index}].timeline must be a non-empty string")
            wav_value = item.get("wav", item.get("output_wav"))
            timeline_out_value = item.get(
                "timeline_output", item.get("output_timeline")
            )
            safe = _safe_job_id(job_id)
            if wav_value is None:
                wav_value = f"{safe}.wav"
            if timeline_out_value is None:
                timeline_out_value = f"{safe}.timeline.json"
            if not isinstance(wav_value, str) or not isinstance(timeline_out_value, str):
                raise ConfigError(f"jobs[{index}] output filenames must be strings")
            overrides: dict[str, Any] = {}
            for key in _JOB_OVERRIDE_KEYS:
                value = _job_value(item, defaults, key)
                if value is not None:
                    overrides[key] = value
            jobs.append(
                BatchJob(
                    index=index,
                    id=job_id,
                    timeline_path=_resolve_under(input_path.parent, timeline_value),
                    wav_path=_resolve_under(output_dir, wav_value),
                    timeline_output_path=_resolve_under(
                        output_dir, timeline_out_value
                    ),
                    overrides=overrides,
                )
            )
    elif isinstance(raw.get("tracks"), list):
        event = raw.get("event", {})
        event_id = event.get("id") if isinstance(event, dict) else None
        job_id = str(event_id or input_path.stem)
        safe = _safe_job_id(job_id)
        if wav_name is None:
            wav_name = f"{safe}.wav"
        if timeline_output_name is None:
            timeline_output_name = f"{safe}.timeline.json"
        jobs.append(
            BatchJob(
                index=0,
                id=job_id,
                timeline_path=input_path,
                wav_path=_resolve_under(output_dir, wav_name),
                timeline_output_path=_resolve_under(output_dir, timeline_output_name),
                overrides={},
            )
        )
    else:
        raise ConfigError(
            "input JSON is neither an eventmix timeline nor a batch manifest"
        )

    if not jobs:
        raise ConfigError("batch manifest contains no jobs")
    seen_ids: set[str] = set()
    for job in jobs:
        if job.id in seen_ids:
            raise ConfigError(f"duplicate job id: {job.id}")
        seen_ids.add(job.id)
    _preflight_output_collisions(jobs)
    return jobs, raw, defaults


def _preflight_output_collisions(jobs: list[BatchJob]) -> None:
    path_owners: dict[str, list[str]] = {}
    input_owners: dict[str, str] = {}
    for job in jobs:
        for output in (job.wav_path, job.timeline_output_path):
            key = os.path.normcase(str(output.resolve()))
            path_owners.setdefault(key, []).append(job.id)
        try:
            project = parse_project(job.timeline_path)
        except EventMixError:
            continue
        input_owners[os.path.normcase(str(job.timeline_path.resolve()))] = job.id
        for track in project.tracks.values():
            input_owners[os.path.normcase(str(track.path.resolve()))] = job.id
    for key, owners in path_owners.items():
        unique = sorted(set(owners))
        if len(owners) > 1 or len(unique) > 1:
            raise OutputCollisionError(
                f"output path collision at {key}: jobs {unique}"
            )
        if key in input_owners:
            raise OutputCollisionError(
                f"job {unique[0]!r} would overwrite input file {key}"
            )


def _engine_config(job: BatchJob, options: BatchOptions) -> EngineConfig:
    def choose(key: str):
        cli_value = getattr(options, key)
        if cli_value is not None:
            return cli_value
        return job.overrides.get(key)

    on_track_error = choose("on_track_error")
    bit_depth = choose("bit_depth")
    target_rate = choose("target_sample_rate")
    threshold = choose("silence_threshold_dbfs")
    min_silence = choose("min_silence_seconds")
    chunk_frames = choose("chunk_frames")
    if chunk_frames is None:
        chunk_frames = options.chunk_frames
    return EngineConfig(
        limits=RenderLimits(
            chunk_frames=int(chunk_frames),
            max_open_wav=options.max_open_wav,
            max_memory_bytes=options.max_memory_mb * 1024 * 1024,
        ),
        sinc=SincConfig(
            half_width=options.sinc_half_width,
            max_half_width=options.sinc_max_half_width,
        ),
        on_track_error=str(on_track_error) if on_track_error is not None else None,
        bit_depth=int(bit_depth) if bit_depth is not None else None,
        target_sample_rate=int(target_rate) if target_rate is not None else None,
        silence_threshold_dbfs=(
            float(threshold) if threshold is not None else None
        ),
        min_silence_seconds=(
            float(min_silence) if min_silence is not None else None
        ),
    )


def _engine_signature(config: EngineConfig) -> dict[str, Any]:
    return {
        "tool_version": TOOL_VERSION,
        "playback_schema": PLAYBACK_SCHEMA,
        "on_track_error": config.on_track_error,
        "bit_depth": config.bit_depth,
        "target_sample_rate": config.target_sample_rate,
        "silence_threshold_dbfs": config.silence_threshold_dbfs,
        "min_silence_seconds": config.min_silence_seconds,
        "chunk_frames": config.limits.chunk_frames,
        "resampler": {
            "half_width": config.sinc.half_width,
            "max_half_width": config.sinc.max_half_width,
            "beta": config.sinc.beta,
        },
    }


class _FileHashCache:
    def __init__(self):
        self._cache: dict[str, tuple[int | None, str]] = {}

    def get(self, path: Path) -> tuple[int | None, str]:
        key = str(path)
        if key in self._cache:
            return self._cache[key]
        try:
            stat = path.stat()
            value = (stat.st_size, sha256_file(path))
        except OSError:
            value = (None, "<missing>")
        self._cache[key] = value
        return value


def _fingerprint_job(
    job: BatchJob,
    project: ProjectSpec,
    config: EngineConfig,
    cache: _FileHashCache,
) -> str:
    timeline_size, timeline_sha = cache.get(job.timeline_path)
    tracks = []
    for track in project.tracks.values():
        size, digest = cache.get(track.path)
        tracks.append(
            {
                "id": track.id,
                "path": str(track.path),
                "size": size,
                "sha256": digest,
            }
        )
    payload = {
        "tool_version": TOOL_VERSION,
        "playback_schema": PLAYBACK_SCHEMA,
        "job_id": job.id,
        "timeline": {
            "path": str(job.timeline_path),
            "size": timeline_size,
            "sha256": timeline_sha,
        },
        "inputs": sorted(tracks, key=lambda item: item["id"]),
        "outputs": {
            "wav": str(job.wav_path),
            "timeline": str(job.timeline_output_path),
        },
        "engine": _engine_signature(config),
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _existing_task_owner(job: BatchJob) -> str | None:
    if job.timeline_output_path.exists():
        try:
            data = load_json_object(job.timeline_output_path, what="existing timeline")
            owner = data.get("task_id")
            if isinstance(owner, str):
                return owner
        except EventMixError:
            pass
    if job.wav_path.exists():
        try:
            metadata = read_wav_metadata(job.wav_path)
            timeline = metadata.get("timeline_json")
            if isinstance(timeline, dict) and isinstance(timeline.get("task_id"), str):
                return timeline["task_id"]
        except (EventMixError, OSError):
            pass
    return None


def _remove_outputs(job: BatchJob) -> None:
    for path in (job.wav_path, job.timeline_output_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _outputs_record(project_result) -> dict[str, dict[str, Any]]:
    return {
        "wav": {
            "path": str(project_result.wav_path),
            "size": project_result.output_wav_size,
            "sha256": project_result.output_wav_sha256,
        },
        "timeline": {
            "path": str(project_result.timeline_path),
            "size": project_result.output_timeline_size,
            "sha256": project_result.output_timeline_sha256,
        },
    }


def _error_record(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, EventMixError):
        code = exc.code
    else:
        code = "internal_error"
    return {
        "code": code,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "fatal": True,
    }


def run_batch(
    input_path: Path | str,
    options: BatchOptions,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    report_progress = progress or (lambda _: None)
    jobs, _, _ = load_jobs(
        input_path,
        options.output_dir,
        wav_name=options.wav_name,
        timeline_output_name=options.timeline_output_name,
    )
    checkpoint_key = os.path.normcase(str(options.checkpoint_path.resolve()))
    for job in jobs:
        for output in (job.wav_path, job.timeline_output_path):
            if os.path.normcase(str(output.resolve())) == checkpoint_key:
                raise ConfigError("checkpoint path collides with a job output path")
    store = CheckpointStore(options.checkpoint_path, resume=options.resume)
    checkpoint = store.load()
    configs = {job.id: _engine_config(job, options) for job in jobs}
    projects: dict[str, ProjectSpec] = {}
    fingerprints: dict[str, str] = {}
    cache = _FileHashCache()
    errors: list[dict[str, Any]] = []
    precomputed_failures: dict[str, dict[str, Any]] = {}
    for job in jobs:
        try:
            project = parse_project(job.timeline_path)
            projects[job.id] = project
            fingerprints[job.id] = _fingerprint_job(
                job, project, configs[job.id], cache
            )
        except Exception as exc:
            record = _error_record(exc)
            record["task_id"] = job.id
            errors.append(record)
            precomputed_failures[job.id] = record
            report_progress(f"[{job.id}] preflight failed: {exc}")

    governor = ResourceGovernor(
        max_open_wav=options.max_open_wav,
        max_memory_bytes=options.max_memory_mb * 1024 * 1024,
    )
    task_reports: list[dict[str, Any]] = []
    completed = 0
    skipped = 0
    failed = 0

    def execute(job: BatchJob) -> dict[str, Any]:
        if job.id in precomputed_failures:
            return {
                "task_id": job.id,
                "status": "failed",
                "degraded": False,
                "warnings": [],
                "error": precomputed_failures[job.id],
            }
        project = projects[job.id]
        config = configs[job.id]
        fingerprint = fingerprints[job.id]
        with exclusive_output_locks([job.wav_path, job.timeline_output_path]):
            latest = store.load().get("jobs", {}).get(job.id)
            if store.is_reusable(latest, fingerprint):
                summary = copy.deepcopy(latest.get("summary", {}))
                summary["task_id"] = job.id
                summary["status"] = "skipped"
                summary.setdefault("warnings", [])
                summary["degraded"] = bool(summary.get("degraded", False))
                report_progress(f"[{job.id}] checkpoint hit; skipped")
                return summary

            owner = _existing_task_owner(job)
            if owner is not None and owner != job.id:
                raise OutputCollisionError(
                    f"job {job.id!r} output path already belongs to task {owner!r}"
                )
            _remove_outputs(job)
            report_progress(f"[{job.id}] rendering")
            try:
                result = build_task(
                    task_id=job.id,
                    project=project,
                    wav_path=job.wav_path,
                    timeline_path=job.timeline_output_path,
                    governor=governor,
                    config=config,
                )
                summary = task_report_from_result(result)
                outputs = _outputs_record(result)
                store.update(
                    job.id,
                    store.completed_record(
                        fingerprint=fingerprint,
                        outputs=outputs,
                        summary=summary,
                    ),
                )
                report_progress(
                    f"[{job.id}] complete ({len(result.warnings)} warning entries)"
                )
                return summary
            except Exception as exc:
                _remove_outputs(job)
                error = _error_record(exc)
                error["task_id"] = job.id
                store.update(
                    job.id,
                    store.failed_record(fingerprint=fingerprint, error=error),
                )
                report_progress(f"[{job.id}] failed: {exc}")
                return {
                    "task_id": job.id,
                    "status": "failed",
                    "degraded": False,
                    "warnings": [],
                    "error": error,
                }

    if options.jobs < 1:
        raise ConfigError("--jobs must be >= 1")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(options.jobs, max(1, len(jobs)))
    ) as pool:
        futures = {pool.submit(execute, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                task_report = future.result()
            except Exception as exc:
                error = _error_record(exc)
                error["task_id"] = job.id
                errors.append(error)
                task_report = {
                    "task_id": job.id,
                    "status": "failed",
                    "degraded": False,
                    "warnings": [],
                    "error": error,
                }
            if task_report.get("status") == "skipped":
                skipped += 1
            elif task_report.get("status") == "failed":
                failed += 1
                task_error = task_report.get("error")
                if isinstance(task_error, dict) and not any(
                    existing.get("task_id") == job.id
                    and existing.get("code") == task_error.get("code")
                    for existing in errors
                ):
                    errors.append(task_error)
            else:
                completed += 1
            task_reports.append(task_report)

    task_reports.sort(key=lambda item: next(j.index for j in jobs if j.id == item["task_id"]))
    warning_count = sum(
        1
        for task in task_reports
        for warning in task.get("warnings", [])
        if warning.get("severity") in {"warning", "error"}
    )
    degraded = any(task.get("degraded") for task in task_reports)
    if failed or errors:
        status = "fatal"
        exit_code = 2
    elif degraded or warning_count:
        status = "partial"
        exit_code = 1
    else:
        status = "ok"
        exit_code = 0
    report = {
        "schema_version": REPORT_SCHEMA,
        "tool_version": TOOL_VERSION,
        "status": status,
        "exit_code": exit_code,
        "invocation": {
            "input": display_path(input_path),
            "output_dir": display_path(options.output_dir),
            "checkpoint": display_path(options.checkpoint_path),
            "resume": options.resume,
            "jobs": options.jobs,
            "max_open_wav": options.max_open_wav,
            "max_memory_mb": options.max_memory_mb,
        },
        "summary": {
            "total": len(jobs),
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "warnings": warning_count,
            "degraded": degraded,
        },
        "checkpoint": {
            "path": display_path(options.checkpoint_path),
            "reset_reason": store.reset_reason,
        },
        "tasks": task_reports,
        "errors": errors,
    }
    return report

