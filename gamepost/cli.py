from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .config import ExportConfig
from .errors import BUSINESS_CODES, CancelledError, EventValidationError, PipelineError
from .events import parse_events
from .fingerprint import write_json_atomic
from .pipeline import (
    MANIFEST_SCHEMA_VERSION,
    ExportRequest,
    run_export,
)
from .resources import (
    CancelFileWatcher,
    CancellationToken,
    MemoryQuota,
    estimate_memory_bytes,
)
from .samples import generate_sample_set
from .timeline import build_timeline
from .wav import read_wav


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--chunk-frames", type=int, default=None)
    parser.add_argument("--clip-threshold", type=int, default=None)
    parser.add_argument("--clip-min-run", type=int, default=None)
    parser.add_argument("--silence-threshold-dbfs", type=float, default=None)
    parser.add_argument("--silence-window-ms", type=float, default=None)
    parser.add_argument("--silence-hop-ms", type=float, default=None)
    parser.add_argument("--silence-min-ms", type=float, default=None)
    parser.add_argument("--silence-merge-ms", type=float, default=None)
    parser.add_argument("--silence-fade-ms", type=float, default=None)
    parser.add_argument("--target-lufs", type=float, default=None)
    parser.add_argument("--game-target-lufs", type=float, default=None)
    parser.add_argument("--mic-target-lufs", type=float, default=None)
    parser.add_argument("--game-mix-db", type=float, default=None)
    parser.add_argument("--mic-mix-db", type=float, default=None)
    parser.add_argument("--max-boost-db", type=float, default=None)
    parser.add_argument("--max-cut-db", type=float, default=None)
    parser.add_argument("--peak-limit-dbfs", type=float, default=None)
    parser.add_argument("--sinc-taps", type=int, default=None)
    parser.add_argument("--sinc-phases", type=int, default=None)
    parser.add_argument("--varispeed-min-rate", type=float, default=None)
    parser.add_argument("--varispeed-max-rate", type=float, default=None)
    parser.add_argument("--event-categories", default=None)


_CONFIG_ARG_NAMES = (
    "sample_rate",
    "chunk_frames",
    "clip_threshold",
    "clip_min_run",
    "silence_threshold_dbfs",
    "silence_window_ms",
    "silence_hop_ms",
    "silence_min_ms",
    "silence_merge_ms",
    "silence_fade_ms",
    "target_lufs",
    "game_target_lufs",
    "mic_target_lufs",
    "game_mix_db",
    "mic_mix_db",
    "max_boost_db",
    "max_cut_db",
    "peak_limit_dbfs",
    "sinc_taps",
    "sinc_phases",
    "varispeed_min_rate",
    "varispeed_max_rate",
    "event_categories",
)


def _config_from_namespace(namespace: argparse.Namespace) -> ExportConfig:
    values: dict[str, Any] = {}
    for name in _CONFIG_ARG_NAMES:
        value = getattr(namespace, name, None)
        if value is not None:
            values[name] = value
    if "event_categories" in values:
        categories = [
            item.strip()
            for item in str(values["event_categories"]).split(",")
            if item.strip()
        ]
        if not categories:
            raise ValueError("--event-categories cannot be empty")
        values["event_categories"] = tuple(categories)
    return ExportConfig().overlaid(values)


def _request_from_namespace(namespace: argparse.Namespace) -> ExportRequest:
    return ExportRequest(
        game=namespace.game,
        mic=namespace.mic,
        events=namespace.events,
        output=namespace.output,
        manifest=namespace.manifest,
        state=namespace.state,
        config=_config_from_namespace(namespace),
    )


def _install_signal_cancellation(token: CancellationToken) -> dict:
    previous: dict[int, Any] = {}

    def handler(signum, frame):
        token.cancel()

    for signal_name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, signal_name, None)
        if signum is None:
            continue
        try:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, handler)
        except (ValueError, OSError):
            pass
    return previous


def _restore_signals(previous: dict) -> None:
    for signum, handler in previous.items():
        try:
            signal.signal(signum, handler)
        except (ValueError, OSError):
            pass


def _exit_for_error(error: Exception) -> int:
    if isinstance(error, PipelineError):
        return error.exit_code
    return 1


def _write_cli_failure(
    report_path: str | None,
    status: str,
    error: Exception,
) -> None:
    payload = {
        "schema": MANIFEST_SCHEMA_VERSION,
        "status": status,
        "business_codes": BUSINESS_CODES,
        "exit_code": _exit_for_error(error),
        "error": (
            error.as_dict()
            if isinstance(error, PipelineError)
            else {
                "code": "INTERNAL_ERROR",
                "code_number": BUSINESS_CODES["INTERNAL_ERROR"]["number"],
                "exit_code": 1,
                "message": str(error),
                "details": {"type": type(error).__name__},
            }
        ),
    }
    if report_path:
        write_json_atomic(report_path, payload)


def command_export(namespace: argparse.Namespace) -> int:
    try:
        request = _request_from_namespace(namespace)
    except ValueError as exc:
        _write_cli_failure(namespace.manifest, "configuration_error", exc)
        print(f"configuration error: {exc}", file=sys.stderr)
        return 3
    token = CancellationToken()
    previous = _install_signal_cancellation(token)
    watcher = None
    if namespace.cancel_file:
        watcher = CancelFileWatcher(namespace.cancel_file, token).start()
    try:
        manifest = run_export(request, token=token)
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "output": manifest["output"],
                    "parameter_fingerprint": manifest["parameter_fingerprint"],
                    "manifest": str(request.manifest_path()),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except EventValidationError as error:
        print(
            f"{error.code}: {error.message} (report: {request.manifest_path()})",
            file=sys.stderr,
        )
        return error.exit_code
    except CancelledError as error:
        _write_cli_failure(str(request.manifest_path()), "cancelled", error)
        print(
            f"CANCELLED: completed chunks are preserved in {request.state_path()}",
            file=sys.stderr,
        )
        return error.exit_code
    except PipelineError as error:
        _write_cli_failure(str(request.manifest_path()), "error", error)
        print(f"{error.code}: {error.message}", file=sys.stderr)
        return error.exit_code
    except Exception as error:
        _write_cli_failure(str(request.manifest_path()), "error", error)
        print(f"INTERNAL_ERROR: {error}", file=sys.stderr)
        return 1
    finally:
        if watcher is not None:
            watcher.stop()
        _restore_signals(previous)


def command_validate_events(namespace: argparse.Namespace) -> int:
    try:
        config = _config_from_namespace(namespace)
        audio = read_wav(namespace.mic, config.sample_rate)
        events = parse_events(namespace.events, audio.frames, config)
        timeline = build_timeline(audio.frames, events, config)
        report = {
            "schema": MANIFEST_SCHEMA_VERSION,
            "status": "valid",
            "business_codes": BUSINESS_CODES,
            "exit_code": 0,
            "inputs": {
                "event_table": str(namespace.events),
                "narration": str(namespace.mic),
            },
            "timeline": timeline.to_dict(),
        }
        write_json_atomic(namespace.report, report)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "event_count": len(events),
                    "output_frames": timeline.output_frames,
                    "report": str(namespace.report),
                },
                sort_keys=True,
            )
        )
        return 0
    except EventValidationError as error:
        report = {
            "schema": MANIFEST_SCHEMA_VERSION,
            "status": "invalid_event_table",
            "business_codes": BUSINESS_CODES,
            "exit_code": error.exit_code,
            "primary_error": error.as_dict(),
            "errors": error.details.get("errors", []),
        }
        write_json_atomic(namespace.report, report)
        print(f"{error.code}: {error.message}", file=sys.stderr)
        return error.exit_code
    except PipelineError as error:
        report = {
            "schema": MANIFEST_SCHEMA_VERSION,
            "status": "input_error",
            "business_codes": BUSINESS_CODES,
            "exit_code": error.exit_code,
            "error": error.as_dict(),
        }
        write_json_atomic(namespace.report, report)
        print(f"{error.code}: {error.message}", file=sys.stderr)
        return error.exit_code
    except ValueError as error:
        report = {
            "schema": MANIFEST_SCHEMA_VERSION,
            "status": "configuration_error",
            "business_codes": BUSINESS_CODES,
            "exit_code": 3,
            "error": {
                "code": "INPUT_WAV_INVALID",
                "message": str(error),
            },
        }
        write_json_atomic(namespace.report, report)
        print(f"configuration error: {error}", file=sys.stderr)
        return 3


def _load_batch_jobs(path: str, base_config: ExportConfig) -> list[tuple[str, ExportRequest]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload.get("jobs") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("batch jobs must be an array or an object with a 'jobs' array")
    jobs: list[tuple[str, ExportRequest]] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"batch job {index} must be an object")
        job_id = str(item.get("id") or f"job-{index + 1}")
        if job_id in seen:
            raise ValueError(f"duplicate batch job id: {job_id}")
        seen.add(job_id)
        for key in ("game", "mic", "events", "output"):
            if not item.get(key):
                raise ValueError(f"batch job {job_id} is missing {key}")
        config = base_config.overlaid(item.get("config"))
        jobs.append(
            (
                job_id,
                ExportRequest(
                    game=str(item["game"]),
                    mic=str(item["mic"]),
                    events=str(item["events"]),
                    output=str(item["output"]),
                    manifest=(
                        str(item["manifest"]) if item.get("manifest") else None
                    ),
                    state=str(item["state"]) if item.get("state") else None,
                    config=config,
                    labels={
                        str(key): str(value)
                        for key, value in (item.get("labels") or {}).items()
                    },
                ),
            )
        )
    return jobs


def command_batch(namespace: argparse.Namespace) -> int:
    token = CancellationToken()
    previous = _install_signal_cancellation(token)
    watcher = None
    if namespace.cancel_file:
        watcher = CancelFileWatcher(namespace.cancel_file, token).start()
    quota = MemoryQuota(int(namespace.memory_mb * 1024 * 1024))
    try:
        base_config = _config_from_namespace(namespace)
        jobs = _load_batch_jobs(namespace.jobs, base_config)
    except Exception as error:
        _write_cli_failure(namespace.report, "batch_configuration_error", error)
        print(f"batch configuration error: {error}", file=sys.stderr)
        if watcher is not None:
            watcher.stop()
        _restore_signals(previous)
        return 3

    def execute(job_id: str, request: ExportRequest) -> dict:
        sizes: list[int] = []
        for path in (request.game, request.mic, request.events):
            try:
                sizes.append(os.path.getsize(path))
            except OSError:
                sizes.append(0)
        required = estimate_memory_bytes(*sizes)
        try:
            with quota.lease(required, token) as reserved:
                manifest = run_export(request, token=token)
            return {
                "id": job_id,
                "status": "complete",
                "exit_code": 0,
                "output": manifest["output"],
                "manifest": str(request.manifest_path()),
                "state": str(request.state_path()),
                "memory_lease_requested_bytes": required,
                "memory_lease_bytes": reserved,
            }
        except EventValidationError as error:
            return {
                "id": job_id,
                "status": "invalid_event_table",
                "exit_code": error.exit_code,
                "error": error.as_dict(),
                "manifest": str(request.manifest_path()),
                "state": str(request.state_path()),
            }
        except CancelledError as error:
            return {
                "id": job_id,
                "status": "cancelled",
                "exit_code": error.exit_code,
                "error": error.as_dict(),
                "manifest": str(request.manifest_path()),
                "state": str(request.state_path()),
            }
        except PipelineError as error:
            _write_cli_failure(str(request.manifest_path()), "error", error)
            return {
                "id": job_id,
                "status": "error",
                "exit_code": error.exit_code,
                "error": error.as_dict(),
                "manifest": str(request.manifest_path()),
                "state": str(request.state_path()),
            }
        except Exception as error:
            _write_cli_failure(str(request.manifest_path()), "error", error)
            return {
                "id": job_id,
                "status": "error",
                "exit_code": 1,
                "error": {"code": "INTERNAL_ERROR", "message": str(error)},
                "manifest": str(request.manifest_path()),
                "state": str(request.state_path()),
            }

    results: list[dict] = []
    try:
        with ThreadPoolExecutor(
            max_workers=namespace.workers,
            thread_name_prefix="gamepost-export",
        ) as executor:
            futures = {
                executor.submit(execute, job_id, request): job_id
                for job_id, request in jobs
            }
            for future in as_completed(futures):
                results.append(future.result())
    finally:
        if watcher is not None:
            watcher.stop()
        _restore_signals(previous)

    results.sort(key=lambda item: item["id"])
    exit_code = 0
    for result in results:
        if result["exit_code"] != 0:
            exit_code = result["exit_code"]
            if exit_code != 130:
                break
    report = {
        "schema": MANIFEST_SCHEMA_VERSION,
        "status": "complete" if exit_code == 0 else "partial",
        "business_codes": BUSINESS_CODES,
        "exit_code": exit_code,
        "workers": namespace.workers,
        "memory_quota_bytes": quota.limit_bytes,
        "jobs": results,
        "diagnostics": {
            "queue_rule": (
                "Jobs wait on the shared memory quota; the executor bounds "
                "concurrent workers and no resource exhaustion is fatal."
            ),
            "cancellation_rule": (
                "Cancellation is checked before each chunk write and leaves "
                "the last atomically committed chunk state resumable."
            ),
        },
    }
    write_json_atomic(namespace.report, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "jobs": len(results),
                "report": str(namespace.report),
            },
            sort_keys=True,
        )
    )
    return exit_code


def command_generate_samples(namespace: argparse.Namespace) -> int:
    result = generate_sample_set(namespace.out_dir, sample_rate=namespace.sample_rate)
    print(json.dumps(result, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gamepost",
        description="Deterministic, resumable game commentary mixer.",
    )
    parser.add_argument("--version", action="version", version="gamepost 1.0.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="run one resumable export")
    export_parser.add_argument("--game", required=True)
    export_parser.add_argument("--mic", required=True)
    export_parser.add_argument("--events", required=True)
    export_parser.add_argument("--out", dest="output", required=True)
    export_parser.add_argument("--manifest", default=None)
    export_parser.add_argument("--state", default=None)
    export_parser.add_argument("--cancel-file", default=None)
    _add_config_arguments(export_parser)
    export_parser.set_defaults(handler=command_export)

    validate_parser = subparsers.add_parser(
        "validate-events",
        help="validate an event table and write its timeline JSON report",
    )
    validate_parser.add_argument("--events", required=True)
    validate_parser.add_argument("--mic", required=True)
    validate_parser.add_argument("--report", required=True)
    _add_config_arguments(validate_parser)
    validate_parser.set_defaults(handler=command_validate_events)

    batch_parser = subparsers.add_parser(
        "batch",
        help="run multiple exports with shared worker and memory limits",
    )
    batch_parser.add_argument("--jobs", required=True)
    batch_parser.add_argument("--report", required=True)
    batch_parser.add_argument("--workers", type=int, default=2)
    batch_parser.add_argument("--memory-mb", type=float, default=512.0)
    batch_parser.add_argument("--cancel-file", default=None)
    _add_config_arguments(batch_parser)
    batch_parser.set_defaults(handler=command_batch)

    samples_parser = subparsers.add_parser(
        "gen-samples",
        help="generate deterministic local WAV and event fixtures",
    )
    samples_parser.add_argument("--out-dir", required=True)
    samples_parser.add_argument("--sample-rate", type=int, default=48_000)
    samples_parser.set_defaults(handler=command_generate_samples)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(argv)
    if namespace.command == "batch" and namespace.workers <= 0:
        parser.error("--workers must be positive")
    if namespace.command == "batch" and namespace.memory_mb <= 0:
        parser.error("--memory-mb must be positive")
    try:
        return int(namespace.handler(namespace))
    except PipelineError as error:
        print(f"{error.code}: {error.message}", file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
