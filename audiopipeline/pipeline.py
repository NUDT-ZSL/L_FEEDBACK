"""End-to-end offline audio alignment and loudness pipeline."""

from __future__ import annotations

import json
import os

import numpy as np

from .builder import build_segment
from .config import CHANNELS, Config, SAMPLE_RATE
from .decoder import check_ffmpeg, decode_audio, probe_duration_seconds
from .errors import PipelineError, catalog_as_list
from .loudness import (
    gain_to_reach_target,
    integrated_loudness,
    true_peak_db,
)
from .manifest import (
    Event,
    load_manifest,
    ms_to_samples as manifest_ms,
    samples_to_ms,
)
from .mixer import place_events
from .writer import write_wav24


def _short_fallback_flag(event: Event, source_frames: int, block_ms: float) -> bool:
    block_frames = int(round(SAMPLE_RATE * block_ms / 1000.0))
    return source_frames < block_frames


class Pipeline:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.config.validate()
        self._decode_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    def _decode_cached(self, event: Event) -> np.ndarray | None:
        path = event.absolute_path
        if path in self._decode_cache:
            return self._decode_cache[path]
        try:
            audio, _ = decode_audio(path)
        except PipelineError as exc:
            event.errors.append(exc.code)
            event.warnings.append(exc.reason)
            return None
        self._decode_cache[path] = audio
        return audio

    def _process_event(self, event: Event, total_samples: int):
        if event.errors:
            return None
        source = self._decode_cached(event)
        if source is None:
            return None

        offset_frames = manifest_ms(event.offset_ms)
        if offset_frames >= source.shape[0]:
            event.errors.append("E_TIME_INVALID")
            return None

        cfg = self.config
        original_lufs = integrated_loudness(
            source,
            block_ms=cfg.loudness_block_ms,
            overlap=cfg.loudness_overlap,
            absolute_gate_lufs=cfg.absolute_gate_lufs,
        )
        if original_lufs is None:
            event.errors.append("E_LOUDNESS_FAILED")
            event.warnings.append("素材为静音或有效门限内无能量，跳过该事件")
            return None

        target = cfg.category_targets_lufs.get(
            event.category, next(iter(cfg.category_targets_lufs.values()))
        )
        gain = gain_to_reach_target(original_lufs, target, cfg.max_gain_db)
        event.original_loudness_lufs = round(float(original_lufs), 3)
        event.adjustment_gain_db = round(float(20.0 * np.log10(gain)), 3)
        event.short_fallback = _short_fallback_flag(
            event, source.shape[0], cfg.loudness_block_ms
        )
        event.source_duration_ms = samples_to_ms(source.shape[0])
        event.probe_duration_s = probe_duration_seconds(event.absolute_path)

        try:
            built = build_segment(event, source, cfg, total_samples)
        except Exception as exc:  # defensive: never crash the whole run
            event.errors.append("E_DECODE_FAILED")
            event.warnings.append(f"片段构建失败: {exc}")
            return None

        segment = built.audio * np.float32(gain)
        event.looped = built.looped
        event.trimmed = built.trimmed
        event.padded = built.padded
        event.segment_samples = segment.shape[0]
        return segment

    # ------------------------------------------------------------------ #
    def run(self, manifest_path: str, output_path: str, report_path: str) -> dict:
        check_ffmpeg()
        events, total_duration_ms, _base = load_manifest(manifest_path)
        total_samples = manifest_ms(total_duration_ms)

        placements: list[tuple[Event, np.ndarray]] = []
        for event in events:
            segment = self._process_event(event, total_samples)
            if segment is not None:
                placements.append((event, segment))

        timeline, _placed, overlap_warnings = place_events(
            placements, total_samples, self.config
        )

        master = self._master(timeline)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        write_wav24(output_path, master)

        report = self._build_report(
            events, total_duration_ms, master, overlap_warnings, manifest_path
        )
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return report

    # ------------------------------------------------------------------ #
    def _master(self, timeline: np.ndarray) -> np.ndarray:
        cfg = self.config
        mixed_lufs = integrated_loudness(
            timeline,
            block_ms=cfg.loudness_block_ms,
            overlap=cfg.loudness_overlap,
            absolute_gate_lufs=cfg.absolute_gate_lufs,
        )
        master_gain = gain_to_reach_target(
            mixed_lufs, cfg.target_integrated_lufs, cfg.max_gain_db
        )
        master = timeline * np.float32(master_gain)

        pre_limit_tp = true_peak_db(master, int(cfg.tp_upsample_factor))
        limit_linear = float(10.0 ** (cfg.true_peak_limit_dbtp / 20.0))
        safety = 1.0
        if pre_limit_tp is not None:
            tp_linear = float(10.0 ** (pre_limit_tp / 20.0))
            if tp_linear > limit_linear:
                safety = limit_linear / tp_linear
            master = master * np.float32(safety)

        final_lufs = integrated_loudness(
            master,
            block_ms=cfg.loudness_block_ms,
            overlap=cfg.loudness_overlap,
            absolute_gate_lufs=cfg.absolute_gate_lufs,
        )
        final_tp = true_peak_db(master, int(cfg.tp_upsample_factor))
        self.master_stats = {
            "mixed_loudness_lufs": _round(mixed_lufs),
            "master_gain_db": round(20.0 * np.log10(master_gain), 3),
            "peak_limiter_gain_db": round(20.0 * np.log10(safety), 3),
            "final_integrated_lufs": _round(final_lufs),
            "final_true_peak_dbtp": _round(final_tp),
            "pre_limit_true_peak_dbtp": _round(pre_limit_tp),
        }
        return master.astype(np.float32)

    # ------------------------------------------------------------------ #
    def _build_report(
        self,
        events: list[Event],
        total_duration_ms: float,
        master: np.ndarray,
        overlap_warnings: list[str],
        manifest_path: str,
    ) -> dict:
        event_reports = []
        succeeded = skipped = 0
        for event in events:
            ok = not event.errors
            if ok:
                succeeded += 1
            else:
                skipped += 1
            event_reports.append(
                {
                    "event_id": event.event_id,
                    "index": event.index,
                    "path": event.path,
                    "category": event.category,
                    "priority": event.priority,
                    "loop": bool(event.loop),
                    "start_ms": event.start_ms,
                    "offset_ms": event.offset_ms,
                    "duration_ms": event.duration_ms,
                    "original_loudness_lufs": getattr(
                        event, "original_loudness_lufs", None
                    ),
                    "adjustment_gain_db": getattr(
                        event, "adjustment_gain_db", None
                    ),
                    "short_event_fallback": getattr(event, "short_fallback", False),
                    "source_duration_ms": getattr(event, "source_duration_ms", None),
                    "probe_duration_seconds": getattr(event, "probe_duration_s", None),
                    "actual_start_ms": event.actual_start_ms,
                    "actual_end_ms": event.actual_end_ms,
                    "looped": getattr(event, "looped", False),
                    "trimmed": getattr(event, "trimmed", False),
                    "padded_silence": getattr(event, "padded", False),
                    "status": "success" if ok else "skipped",
                    "error_codes": sorted(set(event.errors)),
                    "warnings": sorted(set(event.warnings)),
                }
            )

        report = {
            "schema_version": "1.0",
            "manifest": os.path.basename(manifest_path),
            "sample_rate_hz": SAMPLE_RATE,
            "channels": CHANNELS,
            "declared_total_duration_ms": total_duration_ms,
            "actual_total_duration_ms": samples_to_ms(master.shape[0]),
            "total_samples": int(master.shape[0]),
            "config": self.config.to_dict(),
            "summary": {
                "total_events": len(events),
                "succeeded": succeeded,
                "skipped": skipped,
                "overall_loudness_lufs": self.master_stats["final_integrated_lufs"],
                "overall_true_peak_dbtp": self.master_stats["final_true_peak_dbtp"],
                "overlap_warning_count": len(overlap_warnings),
            },
            "master": self.master_stats,
            "overlap_warnings": overlap_warnings,
            "events": event_reports,
            "error_catalog": catalog_as_list(),
        }
        return report


def _round(value, digits: int = 3):
    if value is None:
        return None
    return round(float(value), digits)
