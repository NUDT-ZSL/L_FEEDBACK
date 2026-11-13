from __future__ import annotations

import argparse
import os
import shutil
import signal
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localizer import __version__, audio
from localizer.errors import (
    EXIT_INTERRUPTED,
    EXIT_MUX,
    EXIT_OK,
    EXIT_PROCESSING,
    EXIT_TIMELINE,
    LocalizerError,
    OperationCancelled,
    emit_json_error,
)
from localizer.ffmpeg import FFmpegExecutionError, FFmpegRunner
from localizer.manifest import LanguageTrack, Manifest, load_manifest
from localizer.outputs import OutputManager
from localizer.progress import ProgressReporter
from localizer.subtitles import build_webvtt_bytes


INTERRUPT_GRACE_SECONDS = 1.5


@dataclass(frozen=True)
class LanguageOutputs:
    code: str
    name: str
    stereo: Path
    mono: Path
    vtt: Path


@dataclass(frozen=True)
class LanguageResult:
    outputs: LanguageOutputs
    clip_count: int
    changes: list[dict[str, Any]]
    peak_before_normalization: dict[str, float]


@dataclass(frozen=True)
class LanguageFailure:
    language: str
    error: LocalizerError


@dataclass(frozen=True)
class CancelledLanguage:
    language: str


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="strict", newline="\n")
            except (TypeError, ValueError):
                pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="game-dub-localizer",
        description=(
            "Validate a cutscene manifest, synthesize per-language WAV/VTT "
            "artifacts and mux a deterministic multi-audio MP4."
        ),
    )
    parser.add_argument("manifest", type=Path, help="localization manifest JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override manifest output directory (relative to current directory)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="parallel language workers (default: 2, capped at CPU count)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="fail with E_EXISTS instead of replacing existing outputs",
    )
    parser.add_argument("--ffmpeg", default=None, help="path to ffmpeg executable")
    parser.add_argument("--ffprobe", default=None, help="path to ffprobe executable")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def _plan_outputs(
    manifest: Manifest,
    output_dir: Path,
) -> dict[str, LanguageOutputs]:
    result: dict[str, LanguageOutputs] = {}
    for track in manifest.languages:
        prefix = f"{manifest.output_name}.{track.code}"
        result[track.code] = LanguageOutputs(
            code=track.code,
            name=track.name,
            stereo=output_dir / f"{prefix}.stereo.wav",
            mono=output_dir / f"{prefix}.mono.wav",
            vtt=output_dir / f"{prefix}.vtt",
        )
    return result


def _all_planned_paths(
    manifest: Manifest,
    output_dir: Path,
    language_outputs: dict[str, LanguageOutputs],
) -> list[Path]:
    paths: list[Path] = [output_dir / f"{manifest.output_name}.mp4"]
    for output in language_outputs.values():
        paths.extend([output.stereo, output.mono, output.vtt])
    return paths


def _selected_output_dir(manifest: Manifest, override: Path | None) -> Path:
    if override is None:
        return manifest.output_dir
    path = override.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _process_language(
    track: LanguageTrack,
    output: LanguageOutputs,
    manifest: Manifest,
    outputs: OutputManager,
    progress: ProgressReporter,
    cancel: threading.Event,
) -> LanguageResult | LanguageFailure | CancelledLanguage:
    progress.emit("start", track.code, clips=len(track.clips))
    try:
        loaded: list[audio.LoadedClip] = []
        issues: list[dict[str, Any]] = []
        for clip in track.clips:
            if cancel.is_set():
                raise OperationCancelled()
            progress.emit(
                "inspect",
                track.code,
                clip_id=clip.clip_id,
                path=os.fspath(clip.path),
            )
            try:
                loaded.append(audio.load_clip(clip, manifest.target, cancel))
            except OperationCancelled:
                raise
            except Exception as exc:
                issues.append(
                    {
                        "clip_id": clip.clip_id,
                        "path": os.fspath(clip.path),
                        "reason": str(exc),
                    }
                )

        if issues:
            raise LocalizerError(
                "E_MEDIA",
                "one or more voice clips are missing or cannot be decoded",
                exit_code=EXIT_PROCESSING,
                details={
                    "language": track.code,
                    "skipped_language": True,
                    "issues": issues,
                },
            )

        changes = [
            {
                "clip_id": item.clip.clip_id,
                "path": os.fspath(item.clip.path),
                "differences": item.changes,
            }
            for item in loaded
            if item.changes
        ]
        progress.emit(
            "normalize",
            track.code,
            changed_clips=len(changes),
            changes=changes,
        )
        if cancel.is_set():
            raise OperationCancelled()

        mixed = audio.mix_language(
            loaded,
            manifest.target,
            manifest.total_ms,
            cancel,
        )
        progress.emit(
            "mix",
            track.code,
            frames=len(mixed.stereo),
            mono_peak_before_normalization=mixed.mono_peak_before_normalization,
            stereo_peak_before_normalization=mixed.stereo_peak_before_normalization,
        )

        outputs.write_atomic(
            output.stereo,
            lambda temp: audio.write_wav(
                temp,
                mixed.stereo,
                manifest.target.sample_rate,
                manifest.target.bit_depth,
            ),
            owner=track.code,
        )
        progress.emit("wav_stereo", track.code, path=os.fspath(output.stereo))
        outputs.write_atomic(
            output.mono,
            lambda temp: audio.write_wav(
                temp,
                mixed.mono,
                manifest.target.sample_rate,
                manifest.target.bit_depth,
            ),
            owner=track.code,
        )
        progress.emit("wav_mono", track.code, path=os.fspath(output.mono))
        outputs.write_atomic(
            output.vtt,
            lambda temp: temp.write_bytes(build_webvtt_bytes(track.clips)),
            owner=track.code,
        )
        progress.emit("vtt", track.code, path=os.fspath(output.vtt))

        if cancel.is_set():
            raise OperationCancelled()
        progress.emit("complete", track.code)
        return LanguageResult(
            outputs=output,
            clip_count=len(track.clips),
            changes=changes,
            peak_before_normalization={
                "mono": mixed.mono_peak_before_normalization,
                "stereo": mixed.stereo_peak_before_normalization,
            },
        )
    except OperationCancelled:
        progress.emit("interrupted", track.code)
        return CancelledLanguage(language=track.code)
    except LocalizerError as exc:
        emit_json_error(exc)
        progress.emit("failed", track.code, error=exc.code)
        return LanguageFailure(language=track.code, error=exc)
    except Exception as exc:
        error = LocalizerError(
            "E_PROCESS",
            f"language processing failed: {exc}",
            exit_code=EXIT_PROCESSING,
            details={"language": track.code, "skipped_language": True},
        )
        emit_json_error(error)
        progress.emit("failed", track.code, error=error.code)
        return LanguageFailure(language=track.code, error=error)


def _failure_payload(failure: LanguageFailure) -> dict[str, Any]:
    return {
        "language": failure.language,
        "code": failure.error.code,
        "message": failure.error.message,
        "details": failure.error.details,
    }


def _install_cancel_handlers(cancel: threading.Event) -> None:
    def handle_signal(_signum: int, _frame: Any) -> None:
        cancel.set()

    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, handle_signal)


def _await_workers(
    executor: ThreadPoolExecutor,
    futures: dict[Future[Any], LanguageTrack],
    cancel: threading.Event,
) -> list[Any]:
    pending = set(futures)
    results: list[Any] = []
    while pending and not cancel.is_set():
        completed, pending = wait(
            pending,
            timeout=0.1,
            return_when=FIRST_COMPLETED,
        )
        for future in completed:
            results.append(future.result())

    if cancel.is_set():
        for future in pending:
            future.cancel()
        deadline = time.monotonic() + INTERRUPT_GRACE_SECONDS
        for future in list(pending):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                results.append(future.result(timeout=remaining))
                pending.discard(future)
            except Exception:
                pass
    executor.shutdown(wait=False, cancel_futures=True)
    return results


def _interrupt_result(
    executor: ThreadPoolExecutor | None,
    futures: dict[Future[Any], LanguageTrack],
    outputs: OutputManager,
    runner: FFmpegRunner | None,
    progress: ProgressReporter,
    work_dir: Path | None,
) -> int:
    if runner is not None:
        runner.terminate_all()
    outputs.cleanup_uncommitted()
    if work_dir is not None:
        shutil.rmtree(work_dir, ignore_errors=True)
    error = LocalizerError(
        "E_INTERRUPTED",
        "processing interrupted; completed atomic outputs were retained",
        exit_code=EXIT_INTERRUPTED,
    )
    emit_json_error(error)
    progress.summary(
        ok=False,
        interrupted=True,
        exit_code=EXIT_INTERRUPTED,
        committed_outputs=sorted(os.fspath(path) for path in outputs.committed_outputs()),
    )
    if executor is not None:
        remaining = [
            future for future in futures if future.running()
        ]
        if remaining:
            deadline = time.monotonic() + 0.2
            while time.monotonic() < deadline and any(f.running() for f in remaining):
                time.sleep(0.01)
            if any(f.running() for f in remaining):
                outputs.cleanup_uncommitted()
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(EXIT_INTERRUPTED)
    return EXIT_INTERRUPTED


def _run(args: argparse.Namespace) -> int:
    cancel = threading.Event()
    _install_cancel_handlers(cancel)

    try:
        manifest = load_manifest(args.manifest)
    except LocalizerError as exc:
        emit_json_error(exc)
        return exc.exit_code

    if args.jobs < 1:
        error = LocalizerError(
            "E_TIMELINE",
            "--jobs must be at least 1",
            exit_code=EXIT_TIMELINE,
            details={"field": "jobs", "requested": args.jobs},
        )
        emit_json_error(error)
        return error.exit_code

    cpu_count = os.cpu_count() or 1
    jobs = min(args.jobs, cpu_count)
    output_dir = _selected_output_dir(manifest, args.output_dir)
    language_outputs = _plan_outputs(manifest, output_dir)
    planned_paths = _all_planned_paths(manifest, output_dir, language_outputs)
    outputs = OutputManager(no_overwrite=args.no_overwrite)

    try:
        outputs.preflight(planned_paths)
    except LocalizerError as exc:
        emit_json_error(exc)
        return exc.exit_code

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        error = LocalizerError(
            "E_PROCESS",
            f"cannot create output directory: {exc}",
            exit_code=EXIT_PROCESSING,
            details={"output_dir": os.fspath(output_dir)},
        )
        emit_json_error(error)
        return error.exit_code

    progress = ProgressReporter()
    progress.emit(
        "prepare",
        "*",
        manifest=os.fspath(manifest.path),
        output_dir=os.fspath(output_dir),
        total_ms=manifest.total_ms,
        jobs_requested=args.jobs,
        jobs_effective=jobs,
        cpu_count=cpu_count,
    )

    try:
        runner = FFmpegRunner(args.ffmpeg, args.ffprobe, cancel)
    except LocalizerError as exc:
        emit_json_error(exc)
        progress.summary(ok=False, exit_code=exc.exit_code)
        return exc.exit_code

    executor: ThreadPoolExecutor | None = None
    futures: dict[Future[Any], LanguageTrack] = {}
    results: list[Any] = []
    work_dir: Path | None = None
    try:
        executor = ThreadPoolExecutor(
            max_workers=jobs,
            thread_name_prefix="localizer",
        )
        for track in manifest.languages:
            output = language_outputs[track.code]
            future = executor.submit(
                _process_language,
                track,
                output,
                manifest,
                outputs,
                progress,
                cancel,
            )
            futures[future] = track
        results = _await_workers(executor, futures, cancel)
    except KeyboardInterrupt:
        cancel.set()

    if cancel.is_set():
        return _interrupt_result(
            executor,
            futures,
            outputs,
            runner,
            progress,
            work_dir,
        )

    successes = [item for item in results if isinstance(item, LanguageResult)]
    failures = [item for item in results if isinstance(item, LanguageFailure)]
    successes.sort(key=lambda item: item.outputs.code)

    if not successes:
        progress.summary(
            ok=False,
            exit_code=EXIT_PROCESSING,
            failed_languages=[_failure_payload(item) for item in failures],
            outputs={},
        )
        return EXIT_PROCESSING

    final_mp4 = output_dir / f"{manifest.output_name}.mp4"
    mux_error: LocalizerError | None = None
    validation: dict[str, Any] | None = None
    work_dir = output_dir / outputs.unique_work_name(
        f".{manifest.output_name}.intermediate"
    )
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        black_video = work_dir / "black.mp4"
        black_temp = outputs.reserve_temp(black_video, owner="*")
        progress.emit(
            "black_video",
            "*",
            duration_ms=manifest.total_ms,
            temporary_path=os.fspath(black_temp),
        )
        try:
            frame_count = runner.build_black_video(black_temp, manifest.total_ms)
            outputs.commit_temp(black_temp, black_video)
        except FFmpegExecutionError as exc:
            raise LocalizerError(
                "E_MUX",
                "FFmpeg failed while creating the black placeholder video",
                exit_code=EXIT_MUX,
                details={
                    "stage": "black_video",
                    "intermediate_dir": os.fspath(work_dir),
                    "ffmpeg_stderr": exc.stderr[-8000:],
                },
            ) from exc
        progress.emit("black_video_done", "*", frames=frame_count)
        if cancel.is_set():
            raise OperationCancelled()

        track_order = sorted(successes, key=lambda item: item.outputs.code)
        audio_tracks = [
            (item.outputs.code, item.outputs.stereo) for item in track_order
        ]
        final_temp = outputs.reserve_temp(final_mp4, owner="*")
        progress.emit(
            "mux",
            "*",
            tracks=[code for code, _ in audio_tracks],
            temporary_path=os.fspath(final_temp),
        )
        try:
            runner.mux(
                black_video,
                audio_tracks,
                final_temp,
                sample_rate=manifest.target.sample_rate,
            )
            validation = runner.validate_mux(
                final_temp,
                [code for code, _ in audio_tracks],
                expected_duration_ms=manifest.total_ms,
                expected_sample_rate=manifest.target.sample_rate,
            )
            outputs.commit_temp(final_temp, final_mp4)
        except FFmpegExecutionError as exc:
            raise LocalizerError(
                "E_MUX",
                "FFmpeg failed while muxing the MP4",
                exit_code=EXIT_MUX,
                details={
                    "stage": "mux",
                    "intermediate_dir": os.fspath(work_dir),
                    "ffmpeg_stderr": exc.stderr[-8000:],
                },
            ) from exc
        progress.emit("validate", "*")
        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir = None
    except OperationCancelled:
        cancel.set()
        return _interrupt_result(
            executor,
            futures,
            outputs,
            runner,
            progress,
            work_dir,
        )
    except LocalizerError as exc:
        mux_error = exc
        outputs.cleanup_uncommitted()
        emit_json_error(exc)

    if cancel.is_set():
        return _interrupt_result(
            executor,
            futures,
            outputs,
            runner,
            progress,
            work_dir,
        )

    failure_payloads = [_failure_payload(item) for item in failures]
    outputs_payload = {
        "video": os.fspath(final_mp4) if mux_error is None else None,
        "languages": [
            {
                "language": item.outputs.code,
                "name": item.outputs.name,
                "stereo_wav": os.fspath(item.outputs.stereo),
                "mono_wav": os.fspath(item.outputs.mono),
                "vtt": os.fspath(item.outputs.vtt),
                "clip_count": item.clip_count,
            }
            for item in successes
        ],
    }
    if mux_error is not None:
        exit_code = mux_error.exit_code
        progress.summary(
            ok=False,
            exit_code=exit_code,
            failed_languages=failure_payloads,
            mux_error=mux_error.payload(),
            outputs=outputs_payload,
        )
        return exit_code

    exit_code = EXIT_PROCESSING if failures else EXIT_OK
    progress.summary(
        ok=not failures,
        exit_code=exit_code,
        failed_languages=failure_payloads,
        outputs=outputs_payload,
        mux_validation=validation,
    )
    return exit_code


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except LocalizerError as exc:
        emit_json_error(exc)
        return exc.exit_code
    except KeyboardInterrupt:
        error = LocalizerError(
            "E_INTERRUPTED",
            "processing interrupted",
            exit_code=EXIT_INTERRUPTED,
        )
        emit_json_error(error)
        return EXIT_INTERRUPTED
    except Exception as exc:
        error = LocalizerError(
            "E_INTERNAL",
            f"unexpected internal failure: {exc}",
            exit_code=EXIT_PROCESSING,
        )
        emit_json_error(error)
        return EXIT_PROCESSING
