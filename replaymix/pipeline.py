"""Bounded-concurrency replay processing, mixing, and output writing."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .checkpoint import CheckpointManager
from .errors import E_TRACK_PROCESSING_FAILED, TrackError
from .gain import MAX_GAIN_DB
from .mix import mix_tracks
from .report import build_report, write_report
from .track import (
    TrackResult,
    compute_track_signature,
    process_track,
)
from .timeline import EventTable, TrackSpec, load_event_table
from .wavio import inspect_wav, sha256_file, write_wav_pcm16


@dataclass
class MixConfig:
    target_peak_dbfs: float = -1.0
    max_gain_db: float = 20.0
    concurrency: int = 2
    checkpoint_version: int = 1
    checkpoint_enabled: bool = True
    report_name: str = "report.json"
    wav_name: str = "replay.wav"
    checkpoint_name: str = "checkpoint.jsonl"
    cache_dir_name: str = "checkpoint_cache"

    def __post_init__(self) -> None:
        if not 0 <= self.max_gain_db <= MAX_GAIN_DB:
            raise ValueError(f"max_gain_db must be in [0, {MAX_GAIN_DB:g}]")
        if self.target_peak_dbfs > 0:
            raise ValueError("target_peak_dbfs must be at or below 0 dBFS")
        if not isinstance(self.concurrency, int) or self.concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        if not isinstance(self.checkpoint_version, int) or self.checkpoint_version < 1:
            raise ValueError("checkpoint_version must be a positive integer")


@dataclass
class SessionResult:
    report: dict[str, Any]
    report_path: Path
    wav_path: Path
    checkpoint_path: Path | None
    effective_concurrency: int
    max_observed_concurrency: int


def _bounded_map(
    items: list[TrackSpec],
    worker: Callable[[TrackSpec], TrackResult],
    concurrency: int,
    on_result: Callable[[TrackResult], None] | None = None,
) -> tuple[list[TrackResult], int]:
    """Run at most ``concurrency`` jobs at once and return input-ordered results."""

    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if not items:
        return [], 0

    active = 0
    max_observed = 0
    lock = threading.Lock()

    def tracked(spec: TrackSpec) -> TrackResult:
        nonlocal active, max_observed
        with lock:
            active += 1
            max_observed = max(max_observed, active)
        try:
            return worker(spec)
        finally:
            with lock:
                active -= 1

    results: list[TrackResult] = []
    iterator = iter(items)
    futures: dict[Future[TrackResult], int] = {}
    next_index = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for _ in range(min(concurrency, len(items))):
            spec = next(iterator)
            futures[executor.submit(tracked, spec)] = next_index
            next_index += 1
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda item: futures[item]):
                index = futures.pop(future)
                result = future.result()
                if on_result is not None:
                    on_result(result)
                results.append((index, result))
                try:
                    spec = next(iterator)
                except StopIteration:
                    continue
                futures[executor.submit(tracked, spec)] = next_index
                next_index += 1
    results.sort(key=lambda item: item[0])
    return [result for _, result in results], max_observed


def _run_signature(table: EventTable, config: MixConfig) -> str:
    payload = {
        "checkpoint_version": config.checkpoint_version,
        "session": table.canonical(),
        "target_peak_dbfs": config.target_peak_dbfs,
        "max_gain_db": config.max_gain_db,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _failed_result(
    spec: TrackSpec, input_signature: str, exc: Exception
) -> TrackResult:
    return TrackResult(
        track_id=spec.track_id,
        spec=spec,
        status="failed",
        errors=[
            TrackError(
                code=E_TRACK_PROCESSING_FAILED,
                message=f"{type(exc).__name__}: {exc}",
            )
        ],
        input_signature=input_signature,
    )


def process_session(
    event_table_path: str | Path,
    output_dir: str | Path,
    *,
    config: MixConfig | None = None,
    on_track_complete: Callable[[TrackResult], None] | None = None,
) -> SessionResult:
    cfg = config or MixConfig()
    table = load_event_table(event_table_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    tracks = sorted(table.tracks, key=lambda item: item.sort_key)
    effective_concurrency = min(max(1, cfg.concurrency), max(1, len(tracks)))
    checkpoint_path = output / cfg.checkpoint_name
    cache_dir = output / cfg.cache_dir_name
    manager = CheckpointManager(
        path=checkpoint_path,
        cache_dir=cache_dir,
        version=cfg.checkpoint_version,
        run_signature=_run_signature(table, cfg),
        enabled=cfg.checkpoint_enabled,
    )
    manager.load()

    signatures: dict[int, str] = {}
    if cfg.checkpoint_enabled:
        for index, spec in enumerate(tracks):
            signatures[index] = compute_track_signature(
                spec=spec,
                table=table,
                target_peak_dbfs=cfg.target_peak_dbfs,
                max_gain_db=cfg.max_gain_db,
                checkpoint_version=cfg.checkpoint_version,
            )
    else:
        for index in range(len(tracks)):
            signatures[index] = ""

    results_by_index: dict[int, TrackResult] = {}
    pending: list[tuple[int, TrackSpec]] = []
    index_by_spec_id: dict[int, int] = {}
    for index, spec in enumerate(tracks):
        index_by_spec_id[id(spec)] = index
        restored = manager.restore_result(
            spec=spec, input_signature=signatures[index]
        )
        if restored is not None:
            results_by_index[index] = restored
        else:
            pending.append((index, spec))

    def worker(spec: TrackSpec) -> TrackResult:
        index = index_by_spec_id[id(spec)]
        signature = signatures[index]
        try:
            return process_track(
                spec=spec,
                table=table,
                target_peak_dbfs=cfg.target_peak_dbfs,
                max_gain_db=cfg.max_gain_db,
                input_signature=signature,
            )
        except Exception as exc:
            return _failed_result(spec, signature, exc)

    def on_result(result: TrackResult) -> None:
        manager.recomputed_track_ids.append(result.track_id)
        manager.put(result)
        if on_track_complete is not None:
            on_track_complete(result)

    pending_specs = [spec for _, spec in pending]
    processed, max_observed = _bounded_map(
        pending_specs, worker, effective_concurrency, on_result
    )
    for result in processed:
        index = index_by_spec_id[id(result.spec)]
        results_by_index[index] = result

    results = [results_by_index[index] for index in range(len(tracks))]
    mix = mix_tracks(results, table.total_frames)
    wav_path = output / cfg.wav_name
    write_wav_pcm16(
        wav_path,
        mix.samples,
        sample_rate_hz=48_000,
        channels=1,
    )
    wav_info = inspect_wav(wav_path)
    if (
        wav_info.frames != table.total_frames
        or wav_info.sample_rate_hz != 48_000
        or wav_info.channels != 1
    ):
        raise RuntimeError(
            "encapsulated WAV does not match the session timeline: "
            f"frames={wav_info.frames}/{table.total_frames}, "
            f"rate={wav_info.sample_rate_hz}, channels={wav_info.channels}"
        )

    report_path = output / cfg.report_name
    checkpoint_summary = manager.summary(enabled=cfg.checkpoint_enabled)
    report = build_report(
        table=table,
        results=results,
        mix=mix,
        config=cfg,
        checkpoint_summary=checkpoint_summary,
        wav_path=wav_path,
        report_path=report_path,
        wav_sha256=sha256_file(wav_path),
        wav_sample_rate_hz=wav_info.sample_rate_hz,
        wav_channels=wav_info.channels,
        effective_concurrency=effective_concurrency,
        max_observed_concurrency=max_observed,
    )
    write_report(report_path, report)
    return SessionResult(
        report=report,
        report_path=report_path,
        wav_path=wav_path,
        checkpoint_path=checkpoint_path if cfg.checkpoint_enabled else None,
        effective_concurrency=effective_concurrency,
        max_observed_concurrency=max_observed,
    )
