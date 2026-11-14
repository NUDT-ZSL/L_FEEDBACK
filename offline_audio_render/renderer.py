"""Deterministic streaming offline renderer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .bus_chain import BusProcessor
from .errors import E_EVENT_OVERLAP, SceneError
from .random_engine import derive_variants
from .renderer_events import EventPlayer, EventRenderRecord
from .scene import Scene
from .sources import probe_source
from .wav_output import open_float_wav

# Fixed internal timeline quantum: output is bit-independent of block_size.
TIMELINE_QUANTUM = 512


@dataclass
class RenderResult:
    output_files: dict[str, str]
    report_path: str
    report: dict


@dataclass
class _RuntimeEvent:
    player: EventPlayer | None
    record: EventRenderRecord
    start_sample: int
    end_sample: int
    skipped: bool = False


class Renderer:
    def __init__(self, scene: Scene, output_dir: str | Path) -> None:
        self.scene = scene
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.warnings: list[dict] = []
        self.records: list[EventRenderRecord] = []
        self._max_event_bytes = 0
        self._peak_working_bytes = 0

    def render(self) -> RenderResult:
        scene = self.scene
        runtime_events = self._prepare_events()
        total_frames = self._resolve_duration(runtime_events)

        bus_names = [
            name
            for name in scene.bus_order
            if scene.buses[name].route_to is None
        ]
        handles: dict[str, sf.SoundFile] = {}
        bus_paths: dict[str, Path] = {}
        for name in bus_names:
            path = self.output_dir / f"{_safe_name(name)}.wav"
            handles[name] = open_float_wav(
                path, scene.sample_rate, scene.channels
            )
            bus_paths[name] = path

        processors = {
            name: BusProcessor.build(
                scene.buses[name], scene.channels, scene.sample_rate
            )
            for name in scene.bus_order
        }
        accumulators = {
            name: np.zeros(
                (TIMELINE_QUANTUM, scene.channels), dtype=np.float64
            )
            for name in scene.bus_order
        }

        try:
            for frame_start in range(0, total_frames, TIMELINE_QUANTUM):
                frames = min(TIMELINE_QUANTUM, total_frames - frame_start)
                for buffer in accumulators.values():
                    buffer.fill(0.0)
                self._mix_events(
                    runtime_events, frame_start, frames, accumulators
                )
                self._process_buses(processors, accumulators, frames, handles)
        finally:
            for player_event in runtime_events:
                if player_event.player is not None:
                    player_event.player.close()
            for handle in handles.values():
                handle.close()

        report = self._build_report(
            runtime_events, total_frames, bus_paths
        )
        report_path = self.output_dir / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        return RenderResult(
            output_files={
                name: str(path) for name, path in bus_paths.items()
            },
            report_path=str(report_path),
            report=report,
        )

    def _prepare_events(self) -> list[_RuntimeEvent]:
        scene = self.scene
        runtime: list[_RuntimeEvent] = []
        for spec in scene.events:
            variants = derive_variants(scene.global_seed, spec.seed_offset)
            source_path = scene.scene_dir / spec.source
            info, probe_error = probe_source(source_path)
            start_sample = int(round(spec.time * scene.sample_rate))
            record = EventRenderRecord(
                event_id=spec.event_id,
                source=str(spec.source),
                start_sample=start_sample,
                end_sample=start_sample,
                applied_gain_db=float(variants.gain_jitter_db),
                seed_value=variants.seed_value,
                skipped=False,
                warnings=[],
            )
            if probe_error is not None:
                self._skip(record, probe_error)
                runtime.append(
                    _RuntimeEvent(
                        None,
                        record,
                        record.start_sample,
                        record.end_sample,
                        True,
                    )
                )
                continue
            assert info is not None
            self._track_memory(info.frames, info.channels)
            player = EventPlayer(
                spec=spec,
                source_path=source_path,
                info=info,
                variants=variants,
                scene=scene,
                record=record,
                read_block=scene.block_size,
            )
            start_sample = player.start_sample
            runtime.append(
                _RuntimeEvent(
                    player=player,
                    record=record,
                    start_sample=start_sample,
                    end_sample=start_sample + player.output_frames,
                )
            )
        self._check_runtime_voice_overlap(runtime)
        return runtime

    def _check_runtime_voice_overlap(
        self, runtime_events: list[_RuntimeEvent]
    ) -> None:
        grouped: dict[str, list[_RuntimeEvent]] = {}
        for runtime in runtime_events:
            if runtime.skipped or runtime.player is None:
                continue
            voice = runtime.player.spec.voice
            if voice is not None:
                grouped.setdefault(voice, []).append(runtime)
        for voice, items in grouped.items():
            items.sort(key=lambda item: item.start_sample)
            for previous, current in zip(items, items[1:]):
                if current.start_sample < previous.end_sample:
                    raise SceneError(
                        E_EVENT_OVERLAP,
                        (
                            f"events {previous.record.event_id!r} and "
                            f"{current.record.event_id!r} overlap on shared "
                            f"voice {voice!r}"
                        ),
                        f"$.events[voice={voice}]",
                    )

    def _mix_events(
        self,
        runtime_events: list[_RuntimeEvent],
        frame_start: int,
        frames: int,
        accumulators: dict[str, np.ndarray],
    ) -> None:
        frame_end = frame_start + frames
        for runtime in runtime_events:
            if runtime.skipped:
                continue
            player = runtime.player
            assert player is not None
            if frame_end <= runtime.start_sample or frame_start >= runtime.end_sample:
                continue
            block = player.render_range(frame_start, frames)
            if block is None:
                if not runtime.skipped:
                    runtime.skipped = True
                    runtime.record.skipped = True
                    for warning in player.collect_warnings():
                        self._append_warning(runtime.record.event_id, warning)
                continue
            for warning in player.collect_warnings():
                already = any(
                    item["event_id"] == runtime.record.event_id
                    and warning.startswith(item["code"])
                    for item in self.warnings
                )
                if not already:
                    self._append_warning(runtime.record.event_id, warning)
            self._peak_working_bytes = max(
                self._peak_working_bytes, player.working_bytes()
            )
            accumulators[player.spec.bus][: block.shape[0]] += block

    def _process_buses(
        self,
        processors: dict[str, BusProcessor],
        accumulators: dict[str, np.ndarray],
        frames: int,
        handles: dict[str, sf.SoundFile],
    ) -> None:
        scene = self.scene
        # Dependency order guarantees child buses finish before their parent.
        for name in scene.bus_order:
            processor = processors[name]
            block = processor.process(accumulators[name][:frames])
            accumulators[name][:frames] = block
            route = scene.buses[name].route_to
            if route is not None:
                accumulators[route][:frames] += block
            else:
                clipped = np.clip(block, -1.0, 1.0)
                handles[name].write(clipped.astype(np.float32))

    def _resolve_duration(self, runtime_events: list[_RuntimeEvent]) -> int:
        if self.scene.duration is not None:
            return int(round(self.scene.duration * self.scene.sample_rate))
        if not runtime_events:
            return 0
        return max(event.end_sample for event in runtime_events)

    def _skip(self, record: EventRenderRecord, warning: str) -> None:
        record.skipped = True
        record.warnings.append(warning)
        self._append_warning(record.event_id, warning)

    def _append_warning(self, event_id: str, warning: str) -> None:
        code, _, message = warning.partition(":")
        self.warnings.append(
            {
                "code": code,
                "event_id": event_id,
                "message": message.strip(),
            }
        )

    def _track_memory(self, frames: int, channels: int) -> None:
        source_bytes = frames * channels * 4  # float32 equivalent
        self._max_event_bytes = max(self._max_event_bytes, source_bytes)

    def _build_report(
        self,
        runtime_events: list[_RuntimeEvent],
        total_frames: int,
        bus_paths: dict[str, Path],
    ) -> dict:
        event_reports = []
        for runtime in runtime_events:
            record = runtime.record
            event_reports.append(
                {
                    "id": record.event_id,
                    "source": record.source,
                    "start_sample": runtime.start_sample,
                    "end_sample": runtime.end_sample,
                    "applied_gain_db": round(record.applied_gain_db, 9),
                    "seed_value": record.seed_value,
                    "skipped": record.skipped,
                    "warnings": list(record.warnings),
                }
            )
        return {
            "schema_version": 1,
            "sample_rate": self.scene.sample_rate,
            "layout": self.scene.layout,
            "channels": self.scene.channels,
            "total_frames": total_frames,
            "duration_seconds": total_frames / self.scene.sample_rate,
            "global_seed": self.scene.global_seed,
            "block_size": self.scene.block_size,
            "memory": {
                "largest_source_float32_bytes": self._max_event_bytes,
                "peak_stream_working_bytes": self._peak_working_bytes,
                "working_ratio": (
                    self._peak_working_bytes / self._max_event_bytes
                    if self._max_event_bytes
                    else 0.0
                ),
            },
            "outputs": {
                name: str(path) for name, path in bus_paths.items()
            },
            "skipped_events": [
                item["id"]
                for item in event_reports
                if item["skipped"]
            ],
            "warnings": self.warnings,
            "events": event_reports,
        }


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
