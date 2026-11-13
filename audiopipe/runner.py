"""Pipeline scanning, scheduling, processing and result persistence."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import threading
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Optional

from . import __version__, codes
from .journal import Journal
from .manifest import manifest_error_details, parse_manifest
from .model import Job, JobFailure, JobOutcome, Manifest, PipelineConfig, TrackInfo
from .tools import Toolchain
from .util import (
    SCHEMA_VERSION,
    atomic_write_json,
    file_sha256,
    probe_wav,
    size_mb,
    utc_now,
    write_pcm16_wav_atomic,
)


_FINGERPRINT_VERSION = b"AUDIOPIPE-CONTENT-FP-V1"
_MIB = 1024 * 1024


class PipelineRunner:
    def __init__(self, config: PipelineConfig, stop_event: Optional[threading.Event] = None) -> None:
        self.config = config
        self.root = Path(config.root).resolve()
        self.stop_event = stop_event or threading.Event()
        self.journal_path = Path(config.journal_path).resolve() if config.journal_path else self.root / "audio_pipeline_jobs.json"
        self.result_path = Path(config.result_path).resolve() if config.result_path else self.root / "audio_pipeline_result.json"
        self.work_dir = (
            Path(config.work_dir).resolve()
            if config.work_dir
            else self.root / ".audio-pipeline-tmp"
        )
        self.toolchain = Toolchain(config.ffmpeg, config.ffprobe, config.ffmpeg_timeout_sec)
        self.journal = Journal(self.journal_path, self.root)
        self.run_id = uuid.uuid4().hex
        self.started_at = utc_now()
        self.fatal_error: Optional[dict[str, Any]] = None
        self.interrupted = False
        self.jobs: list[Job] = []

    def run(self) -> dict[str, Any]:
        if not self.root.is_dir():
            self._fatal_without_root(
                JobFailure(codes.ROOT_INVALID, f"show root is not a directory: {self.root}", fatal=True)
            )
            return self._finish([])
        if not os.access(self.root, os.R_OK | os.X_OK):
            self.fatal_error = JobFailure(
                codes.ROOT_INVALID,
                f"show root is not readable/searchable: {self.root}",
                fatal=True,
            ).as_dict()
            return self._finish([])

        try:
            self.jobs = self._scan_jobs()
        except OSError as exc:
            self.fatal_error = JobFailure(
                codes.ROOT_INVALID,
                f"cannot scan show root {self.root}: {exc}",
                fatal=True,
            ).as_dict()
            return self._finish([])
        try:
            self.toolchain.validate()
            self._prepare_work_dir()
            self.journal.load()
        except JobFailure as exc:
            if not exc.fatal:
                exc = JobFailure(codes.INTERNAL_ERROR, exc.message, fatal=True, details=exc.details)
            self.fatal_error = exc.as_dict()
            outcomes = [
                JobOutcome(
                    job=job,
                    status="failed",
                    error=exc.as_dict(),
                    fatal=True,
                )
                for job in self.jobs
            ]
            if exc.code != codes.JOURNAL_CORRUPT:
                self._best_effort_journal(outcomes)
            return self._finish(outcomes)
        except OSError as exc:
            fatal = JobFailure(codes.TEMP_IO_ERROR, f"workspace setup failed: {exc}", fatal=True)
            self.fatal_error = fatal.as_dict()
            outcomes = [
                JobOutcome(job=job, status="failed", error=fatal.as_dict(), fatal=True)
                for job in self.jobs
            ]
            self._best_effort_journal(outcomes)
            return self._finish(outcomes)

        outcomes = self._run_jobs()
        return self._finish(outcomes)

    def _scan_jobs(self) -> list[Job]:
        if not self.root.is_dir():
            return []
        root_resolved = self.root.resolve()
        work_resolved = self.work_dir.resolve()
        jobs: list[Job] = []
        def raise_walk_error(error: OSError) -> None:
            raise error

        for current, dirnames, filenames in os.walk(
            root_resolved, followlinks=False, onerror=raise_walk_error
        ):
            current_path = Path(current)
            kept: list[str] = []
            for name in sorted(dirnames):
                candidate = current_path / name
                try:
                    resolved = candidate.resolve()
                except OSError:
                    continue
                if work_resolved != root_resolved and (
                    resolved == work_resolved or _is_relative_to(resolved, work_resolved)
                ):
                    continue
                kept.append(name)
            dirnames[:] = kept
            if "manifest.json" not in filenames:
                continue
            manifest_path = current_path / "manifest.json"
            relative_dir = current_path.relative_to(root_resolved)
            job_id = "." if str(relative_dir) == "." else relative_dir.as_posix()
            manifest_rel = manifest_path.relative_to(root_resolved).as_posix()
            jobs.append(
                Job(
                    job_id=job_id,
                    job_dir=current_path,
                    manifest_path=manifest_path,
                    manifest_rel=manifest_rel,
                )
            )
        jobs.sort(key=lambda job: job.job_id)
        return jobs

    def _prepare_work_dir(self) -> None:
        try:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            (self.work_dir / "jobs").mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise JobFailure(
                codes.TEMP_IO_ERROR,
                f"cannot create temp workspace {self.work_dir}: {exc}",
                fatal=True,
            ) from exc
        try:
            usage = shutil.disk_usage(self.work_dir)
        except OSError as exc:
            raise JobFailure(
                codes.TEMP_IO_ERROR,
                f"cannot inspect temp workspace {self.work_dir}: {exc}",
                fatal=True,
            ) from exc
        if usage.free < self.config.min_free_mb * _MIB:
            raise JobFailure(
                codes.TEMP_SPACE_INSUFFICIENT,
                f"temp workspace has only {size_mb(usage.free):.1f} MiB free; "
                f"at least {self.config.min_free_mb} MiB reserve is required",
                fatal=True,
            )

    def _run_jobs(self) -> list[JobOutcome]:
        outcomes: dict[str, JobOutcome] = {}
        next_index = 0
        running: dict[Future[JobOutcome], Job] = {}
        stop_submitting = False

        with ThreadPoolExecutor(max_workers=max(1, self.config.jobs), thread_name_prefix="audiopipe") as executor:
            while next_index < len(self.jobs) or running:
                if self.stop_event.is_set() and not self.interrupted:
                    self.interrupted = True
                    stop_submitting = True

                if not stop_submitting:
                    while next_index < len(self.jobs) and len(running) < max(1, self.config.jobs):
                        job = self.jobs[next_index]
                        next_index += 1
                        future = executor.submit(self._process_job, job)
                        running[future] = job

                if not running:
                    if stop_submitting:
                        break
                    continue

                done, _ = wait(running.keys(), timeout=0.2, return_when=FIRST_COMPLETED)
                for future in done:
                    job = running.pop(future)
                    try:
                        outcome = future.result()
                    except Exception as exc:  # defensive: no worker exception is allowed to escape
                        outcome = JobOutcome(
                            job=job,
                            status="failed",
                            error={
                                "code": codes.INTERNAL_ERROR,
                                "message": f"unhandled worker error: {type(exc).__name__}: {exc}",
                            },
                        )
                    outcomes[job.job_id] = outcome
                    self._record_outcome(outcome)
                    if outcome.fatal:
                        self.fatal_error = outcome.error
                        stop_submitting = True
                    if self.fatal_error is not None:
                        stop_submitting = True

                if self.stop_event.is_set():
                    stop_submitting = True

            if running:
                for future in list(running):
                    job = running.pop(future)
                    try:
                        outcome = future.result()
                    except Exception as exc:
                        outcome = JobOutcome(
                            job=job,
                            status="failed",
                            error={
                                "code": codes.INTERNAL_ERROR,
                                "message": f"unhandled worker error: {type(exc).__name__}: {exc}",
                            },
                        )
                    outcomes[job.job_id] = outcome
                    self._record_outcome(outcome)

        if next_index < len(self.jobs):
            fatal_payload = self.fatal_error
            for job in self.jobs[next_index:]:
                if job.job_id in outcomes:
                    continue
                if fatal_payload is not None:
                    outcome = JobOutcome(job=job, status="failed", error=fatal_payload, fatal=True)
                    outcomes[job.job_id] = outcome
                    self._record_outcome(outcome)
                elif self.interrupted:
                    outcomes[job.job_id] = JobOutcome(
                        job=job,
                        status="interrupted",
                        error={
                            "code": codes.INTERRUPTED,
                            "message": "job was not started because Ctrl+C requested a graceful drain",
                        },
                    )

        return [outcomes[job.job_id] for job in self.jobs if job.job_id in outcomes]

    def _record_outcome(self, outcome: JobOutcome) -> None:
        if outcome.status == "interrupted":
            return
        entry = outcome.journal_entry(utc_now())
        try:
            self.journal.upsert(entry)
            self.journal.save()
        except JobFailure as exc:
            self.fatal_error = exc.as_dict()
        except OSError as exc:
            self.fatal_error = JobFailure(
                codes.JOURNAL_WRITE_FAILED,
                f"cannot persist journal {self.journal_path}: {exc}",
                fatal=True,
            ).as_dict()

    def _best_effort_journal(self, outcomes: list[JobOutcome]) -> None:
        for outcome in outcomes:
            self.journal.upsert(outcome.journal_entry(utc_now()))
        try:
            self.journal.save()
        except JobFailure:
            pass

    def _process_job(self, job: Job) -> JobOutcome:
        temp_dir: Optional[Path] = None
        try:
            _raw_manifest, manifest, fingerprint, manifest_error = self._load_and_fingerprint(job)
            if manifest_error is not None:
                return JobOutcome(
                    job=job,
                    status="failed",
                    fingerprint=fingerprint,
                    error=manifest_error.as_dict(),
                )
            assert manifest is not None
            manifest_values = manifest.as_dict()

            cached = self.journal.cache_hit(job.job_id, fingerprint)
            if cached is not None:
                return JobOutcome(
                    job=job,
                    status="skipped",
                    fingerprint=fingerprint,
                    output_rel=cached.get("output"),
                    output_size=cached.get("output_size"),
                    measured_dialogue_lufs=_optional_float(cached.get("measured_dialogue_lufs")),
                    requested_gain_db=_optional_float(cached.get("requested_gain_db")),
                    applied_gain_db=_optional_float(cached.get("applied_gain_db")),
                    clamped=bool(cached.get("clamped", False)),
                    clamp=str(cached.get("clamp", "none")),
                    manifest_values=manifest_values,
                    from_cache=True,
                )

            tracks = self._probe_tracks(job, manifest)
            self._check_temp_estimate(tracks)
            temp_dir = self._create_job_temp(job)

            dialogue_norm = temp_dir / "dialogue-48k-stereo.wav"
            music_norm = temp_dir / "music-48k-stereo.wav"
            ambience_norm = temp_dir / "ambience-48k-stereo.wav"
            mix_float = temp_dir / "mix-pre-gain-f32.wav"
            final_raw = temp_dir / "mix-final-s16le.pcm"

            self._ensure_capacity(temp_dir, 1 * _MIB)
            self.toolchain.decode_to_pcm48_stereo(tracks[0].path, dialogue_norm, "dialogue")
            self._enforce_temp_limit(temp_dir)
            self._ensure_capacity(temp_dir, 1 * _MIB)
            self.toolchain.decode_to_pcm48_stereo(tracks[1].path, music_norm, "music")
            self._enforce_temp_limit(temp_dir)
            self._ensure_capacity(temp_dir, 1 * _MIB)
            self.toolchain.decode_to_pcm48_stereo(tracks[2].path, ambience_norm, "ambience")
            self._enforce_temp_limit(temp_dir)

            dialogue_lufs = self.toolchain.measure_integrated_lufs(dialogue_norm)
            requested_gain = float(manifest.target_lufs) - dialogue_lufs
            applied_gain, clamped, clamp = _clamp_gain(requested_gain, manifest)
            requested_gain = round(requested_gain, 4)
            applied_gain = round(applied_gain, 4)

            self._ensure_capacity(temp_dir, 1 * _MIB)
            self.toolchain.mix_tracks(
                dialogue_norm,
                music_norm,
                ambience_norm,
                mix_float,
                manifest.music_fade_start_sec,
                manifest.music_fade_duration_sec,
                manifest.music_gain_db,
                manifest.ambience_gain_db,
            )
            self._enforce_temp_limit(temp_dir)

            self._ensure_capacity(temp_dir, 1 * _MIB)
            self.toolchain.apply_gain_to_raw_pcm(mix_float, final_raw, applied_gain)
            self._enforce_temp_limit(temp_dir)

            output_abs = self._output_path(job)
            output_rel = output_abs.relative_to(self.root).as_posix()
            try:
                output_abs.parent.mkdir(parents=True, exist_ok=True)
                wav_info = write_pcm16_wav_atomic(final_raw, output_abs)
                verified = probe_wav(output_abs)
                if verified["data_size"] != wav_info["data_size"]:
                    raise ValueError("post-replace WAV verification changed data size")
            except OverflowError as exc:
                raise JobFailure(codes.OUTPUT_TOO_LARGE, str(exc)) from exc
            except ValueError as exc:
                raise JobFailure(codes.OUTPUT_INVALID, f"generated WAV is invalid: {exc}") from exc
            except OSError as exc:
                raise JobFailure(codes.OUTPUT_IO_ERROR, f"cannot write output WAV: {exc}") from exc

            return JobOutcome(
                job=job,
                status="completed",
                fingerprint=fingerprint,
                output_rel=output_rel,
                output_size=verified["file_size"],
                measured_dialogue_lufs=round(dialogue_lufs, 4),
                requested_gain_db=requested_gain,
                applied_gain_db=applied_gain,
                clamped=clamped,
                clamp=clamp,
                track_info=[track.as_dict() for track in tracks],
                manifest_values=manifest_values,
                output_validated=True,
            )
        except JobFailure as exc:
            return JobOutcome(
                job=job,
                status="failed",
                fingerprint=locals().get("fingerprint", ""),
                error=exc.as_dict(),
                fatal=exc.fatal,
                track_info=[],
            )
        except OSError as exc:
            return JobOutcome(
                job=job,
                status="failed",
                fingerprint=locals().get("fingerprint", ""),
                error={
                    "code": codes.TEMP_IO_ERROR if temp_dir is not None else codes.INTERNAL_ERROR,
                    "message": f"filesystem error: {exc}",
                },
            )
        except Exception as exc:
            return JobOutcome(
                job=job,
                status="failed",
                fingerprint=locals().get("fingerprint", ""),
                error={
                    "code": codes.INTERNAL_ERROR,
                    "message": f"{type(exc).__name__}: {exc}",
                },
            )
        finally:
            if temp_dir is not None:
                try:
                    shutil.rmtree(temp_dir)
                except OSError:
                    pass

    def _load_and_fingerprint(
        self, job: Job
    ) -> tuple[bytes, Optional[Manifest], str, Optional[JobFailure]]:
        try:
            raw = job.manifest_path.read_bytes()
        except OSError as exc:
            failure = JobFailure(codes.MANIFEST_INVALID, f"cannot read manifest: {exc}")
            fingerprint = hashlib.sha256(
                _FINGERPRINT_VERSION + job.job_id.encode("utf-8") + f"READ_ERROR:{exc}".encode("utf-8")
            ).hexdigest()
            return b"", None, fingerprint, failure

        manifest: Optional[Manifest] = None
        manifest_failure: Optional[JobFailure] = None
        try:
            manifest = parse_manifest(job, raw)
        except ValueError as exc:
            manifest_failure = JobFailure(
                codes.MANIFEST_INVALID,
                str(exc),
                details=manifest_error_details(exc),
            )

        digest = hashlib.sha256()
        digest.update(_FINGERPRINT_VERSION)
        digest.update(b"\0")
        digest.update(__version__.encode("ascii"))
        digest.update(b"\0")
        digest.update(job.job_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw)
        if manifest is not None:
            for role, path in (
                ("dialogue", manifest.dialogue_path),
                ("music", manifest.music_path),
                ("ambience", manifest.ambience_path),
            ):
                digest.update(b"\0")
                digest.update(role.encode("ascii"))
                digest.update(b"\0")
                digest.update(path.encode("utf-8"))
                digest.update(b"\0")
                source = (job.job_dir / Path(path.replace("\\", "/"))).resolve()
                if source.is_file():
                    try:
                        digest.update(file_sha256(source).encode("ascii"))
                    except OSError as exc:
                        digest.update(f"READ_ERROR:{type(exc).__name__}".encode("ascii"))
                elif source.exists():
                    digest.update(b"NOT_A_REGULAR_FILE")
                else:
                    digest.update(b"MISSING")
        return raw, manifest, digest.hexdigest(), manifest_failure

    def _probe_tracks(self, job: Job, manifest: Manifest) -> list[TrackInfo]:
        specs = (
            ("dialogue", manifest.dialogue_path),
            ("music", manifest.music_path),
            ("ambience", manifest.ambience_path),
        )
        tracks: list[TrackInfo] = []
        for role, relative in specs:
            source = (job.job_dir / Path(relative.replace("\\", "/"))).resolve()
            tracks.append(self.toolchain.probe(role, source))
        return tracks

    def _create_job_temp(self, job: Job) -> Path:
        digest = hashlib.sha256(job.job_id.encode("utf-8")).hexdigest()[:20]
        path = self.work_dir / "jobs" / f"job-{digest}"
        try:
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise JobFailure(codes.TEMP_IO_ERROR, f"cannot create per-job temp dir {path}: {exc}") from exc
        return path

    def _check_temp_estimate(self, tracks: list[TrackInfo]) -> None:
        durations = [track.duration_sec for track in tracks if track.duration_sec is not None]
        if len(durations) != 3:
            try:
                free = shutil.disk_usage(self.work_dir).free
            except OSError as exc:
                raise JobFailure(codes.TEMP_IO_ERROR, f"cannot inspect temp free space: {exc}") from exc
            reserve = (self.config.min_free_mb + 2) * _MIB
            if free < reserve:
                raise JobFailure(
                    codes.TEMP_SPACE_INSUFFICIENT,
                    f"not enough free temp space for tracks with unknown duration: {size_mb(free):.1f} MiB",
                    details={"free_bytes": free, "reserve_bytes": reserve},
                )
            return
        duration = max(float(value) for value in durations)
        frames = int(math.ceil(duration * 48000.0))
        needed = (
            3 * (frames * 4 + 1024)
            + (frames * 8 + 1024)
            + (frames * 4)
            + 2 * _MIB
        )
        needed = int(needed * 1.10)
        limit = self.config.job_temp_limit_mb * _MIB
        if needed > limit:
            raise JobFailure(
                codes.TEMP_SPACE_INSUFFICIENT,
                f"job needs about {size_mb(needed):.1f} MiB of temp space, "
                f"but per-job limit is {self.config.job_temp_limit_mb} MiB",
                details={"estimated_bytes": needed, "limit_bytes": limit},
            )
        try:
            free = shutil.disk_usage(self.work_dir).free
        except OSError as exc:
            raise JobFailure(codes.TEMP_IO_ERROR, f"cannot inspect temp free space: {exc}") from exc
        reserve = self.config.min_free_mb * _MIB
        if free < needed + reserve:
            raise JobFailure(
                codes.TEMP_SPACE_INSUFFICIENT,
                f"temp filesystem has {size_mb(free):.1f} MiB free; "
                f"job needs about {size_mb(needed):.1f} MiB plus {self.config.min_free_mb} MiB reserve",
                details={"free_bytes": free, "estimated_bytes": needed, "reserve_bytes": reserve},
            )

    def _ensure_capacity(self, directory: Path, additional: int) -> None:
        current = _directory_size(directory)
        limit = self.config.job_temp_limit_mb * _MIB
        if current + additional > limit:
            raise JobFailure(
                codes.TEMP_SPACE_INSUFFICIENT,
                f"per-job temp workspace would exceed {self.config.job_temp_limit_mb} MiB",
                details={"current_bytes": current, "additional_bytes": additional, "limit_bytes": limit},
            )
        try:
            free = shutil.disk_usage(directory if directory.exists() else self.work_dir).free
        except OSError as exc:
            raise JobFailure(codes.TEMP_IO_ERROR, f"cannot inspect temp free space: {exc}") from exc
        reserve = self.config.min_free_mb * _MIB
        if free < additional + reserve:
            raise JobFailure(
                codes.TEMP_SPACE_INSUFFICIENT,
                f"insufficient temp workspace: {size_mb(free):.1f} MiB free, "
                f"{size_mb(additional):.1f} MiB needed plus reserve",
                details={"free_bytes": free, "additional_bytes": additional, "reserve_bytes": reserve},
            )

    def _enforce_temp_limit(self, temp_dir: Path) -> None:
        current = _directory_size(temp_dir)
        limit = self.config.job_temp_limit_mb * _MIB
        if current > limit:
            raise JobFailure(
                codes.TEMP_SPACE_INSUFFICIENT,
                f"per-job temp workspace reached {size_mb(current):.1f} MiB, "
                f"exceeding {self.config.job_temp_limit_mb} MiB",
                details={"current_bytes": current, "limit_bytes": limit},
            )

    def _output_path(self, job: Job) -> Path:
        if job.job_id == ".":
            return self.root / "renders" / "mix.wav"
        return self.root / "renders" / Path(job.job_id) / "mix.wav"

    def _finish(self, outcomes: list[JobOutcome]) -> dict[str, Any]:
        jobs_payload = [outcome.as_result() for outcome in outcomes]
        jobs_payload.sort(key=lambda value: value["id"])
        counts = {
            "total": len(jobs_payload),
            "completed": sum(1 for item in jobs_payload if item["status"] == "completed"),
            "skipped": sum(1 for item in jobs_payload if item["status"] == "skipped"),
            "failed": sum(1 for item in jobs_payload if item["status"] == "failed"),
            "interrupted": sum(1 for item in jobs_payload if item["status"] == "interrupted"),
        }
        assert counts["total"] == sum(
            counts[key] for key in ("completed", "skipped", "failed", "interrupted")
        )
        if self.fatal_error is not None:
            exit_code = codes.EXIT_FATAL
            exit_reason = "fatal"
        elif self.interrupted:
            exit_code = codes.EXIT_INTERRUPTED
            exit_reason = "interrupted"
        elif counts["failed"]:
            exit_code = codes.EXIT_PARTIAL
            exit_reason = "partial_failure"
        else:
            exit_code = codes.EXIT_OK
            exit_reason = "success"

        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": __version__,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": utc_now(),
            "root": str(self.root),
            "journal": str(self.journal_path),
            "result_file": str(self.result_path),
            "work_dir": str(self.work_dir),
            "settings": {
                "jobs": self.config.jobs,
                "job_temp_limit_mb": self.config.job_temp_limit_mb,
                "min_free_mb": self.config.min_free_mb,
            },
            "tools": {
                "ffmpeg": str(self.toolchain.ffmpeg),
                "ffprobe": str(self.toolchain.ffprobe),
                "versions": self.toolchain.versions,
            },
            "counts": counts,
            "exit_code": exit_code,
            "exit_reason": exit_reason,
            "interrupted": self.interrupted,
            "fatal_error": self.fatal_error,
            "jobs": jobs_payload,
        }

        try:
            atomic_write_json(self.result_path, result)
        except OSError as exc:
            if self.fatal_error is None:
                self.fatal_error = JobFailure(
                    codes.RESULT_WRITE_FAILED,
                    f"cannot write result file {self.result_path}: {exc}",
                    fatal=True,
                ).as_dict()
                result["fatal_error"] = self.fatal_error
                result["exit_reason"] = "fatal"
                result["exit_code"] = codes.EXIT_FATAL
        return result

    def _fatal_without_root(self, failure: JobFailure) -> None:
        self.fatal_error = failure.as_dict()
        if self.config.result_path is None:
            self.result_path = Path.cwd() / "audio_pipeline_result.json"
        else:
            self.result_path = Path(self.config.result_path).resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _directory_size(directory: Path) -> int:
    total = 0
    try:
        for root, _, files in os.walk(directory):
            for filename in files:
                try:
                    total += (Path(root) / filename).stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


def _optional_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if math.isfinite(result):
            return result
    return None


def _clamp_gain(requested: float, manifest: Manifest) -> tuple[float, bool, str]:
    if requested >= manifest.max_boost_db:
        return manifest.max_boost_db, True, "max_boost"
    if requested <= -manifest.max_attenuation_db:
        return -manifest.max_attenuation_db, True, "max_attenuation"
    return requested, False, "none"
