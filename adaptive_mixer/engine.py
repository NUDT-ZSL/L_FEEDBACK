"""Core adaptive mixing engine: scheduling, loudness, transitions, limiting."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .config import SceneConfig, SceneSpec, verify_track_files
from .envelopes import apply_fade_in_out, crossfade_gains
from .errors import MixerError
from .limiter import TRUE_PEAK_CEILING_DBTP, limit, true_peak_db
from .loudness import (OFFSET_DB, _gated_mean_square, k_weight,
                       k_weighting_ir, lufs_to_linear_gain)
from .state import (Checkpoint, assert_compatible, load_checkpoint,
                    save_checkpoint)
from .wavio import read_wav, write_wav_24bit

EDGE_FADE_MS = 10.0
LOUDNESS_TOLERANCE_LU = 0.5
INNER_TARGET_TOLERANCE_LU = 0.3
MAX_LOUDNESS_ITERATIONS = 8


class RenderInterrupted(Exception):
    """Raised by the test/control hook to simulate an interrupted render."""

    def __init__(self, scene_id: str, state_file: str) -> None:
        super().__init__(f"render interrupted after scene {scene_id}")
        self.scene_id = scene_id
        self.state_file = state_file


@dataclass
class MixResult:
    output_path: str
    report: Dict[str, Any]
    resumed: bool = False


class MixEngine:
    """Renders one scene configuration into an adaptive mono mix."""

    def __init__(self, config: SceneConfig, base_dir, output_path,
                 state_file: Optional[str] = None) -> None:
        self.config = config
        self.base_dir = Path(base_dir)
        self.output_path = Path(output_path)
        self.state_file = Path(state_file) if state_file \
            else self.output_path.with_suffix(".state.json")
        self.parts_dir = self.output_path.with_suffix(".parts")
        self.fir = k_weighting_ir(config.sample_rate)
        self._load_tracks()
        self.weighted_tracks: Dict[str, np.ndarray] = {
            name: k_weight(data, self.fir)
            for name, data in self.track_data.items()
        }

    def _load_tracks(self) -> None:
        verify_track_files(self.config, self.base_dir)
        self.track_data: Dict[str, np.ndarray] = {}
        for track in self.config.tracks:
            candidate = Path(track.path)
            if not candidate.is_absolute():
                candidate = self.base_dir / candidate
            data, sample_rate = read_wav(candidate)
            if len(data) != self.config.total_samples:
                raise MixerError(
                    "E_TRACK_LENGTH",
                    f"分轨 {track.name} 长度 {len(data)} 与时间轴 "
                    f"{self.config.total_samples} 不一致",
                    {"track": track.name,
                     "actual_samples": len(data),
                     "expected_samples": self.config.total_samples})
            self.track_data[track.name] = data.astype(np.float64)

    # ------------------------------------------------------------------
    def render(self, interrupt_after: Optional[Callable[[str], bool]] = None
               ) -> MixResult:
        cfg = self.config
        spf = cfg.samples_per_frame
        checkpoint = load_checkpoint(self.state_file)
        resumed = False
        if checkpoint is not None:
            assert_compatible(checkpoint, cfg.version, str(self.output_path),
                              cfg.sample_rate, cfg.total_samples)
            resumed = True
        completed_ids = list(checkpoint.completed_scenes) if checkpoint else []
        scene_reports = list(checkpoint.scene_reports) if checkpoint else []
        crossfade_events = list(checkpoint.crossfades) if checkpoint else []

        rendered_parts: List[np.ndarray] = []
        completed_samples = 0
        self.parts_dir.mkdir(parents=True, exist_ok=True)

        for index, scene in enumerate(cfg.scenes):
            start = scene.start_frame * spf
            end = scene.end_frame * spf
            part_file = self._part_file(index)
            if scene.scene_id in completed_ids and part_file.is_file():
                part = np.load(part_file)
                rendered_parts.append(part)
                completed_samples += len(part)
                continue
            if scene.scene_id in completed_ids and not part_file.is_file():
                # Checkpoint references a segment whose part file was lost;
                # drop stale progress and re-render from this scene onward.
                completed_ids = completed_ids[:index]
                scene_reports = scene_reports[:index]
                crossfade_events = [
                    event for event in crossfade_events
                    if event["to_scene"] != scene.scene_id]
                rendered_parts = []
                completed_samples = 0
                self._cleanup_parts(len(cfg.scenes))
                self.parts_dir.mkdir(parents=True, exist_ok=True)

            mix, track_gains = self._render_scene(scene, start, end)
            if index > 0:
                event = self._apply_crossfade(
                    cfg.scenes[index - 1], scene, rendered_parts[-1], mix,
                    spf)
                crossfade_events.append(event)

            trim_db, measured_lufs = self._bus_normalize(scene, mix)
            part_path = self._part_file(index)
            np.save(part_path, mix)
            rendered_parts.append(mix)
            completed_samples += len(mix)

            scene_reports.append(self._scene_report(
                scene, index, start, end, mix, track_gains,
                trim_db, measured_lufs))
            completed_ids.append(scene.scene_id)
            self._persist(checkpoint, completed_ids, completed_samples,
                          scene_reports, crossfade_events)

            if interrupt_after is not None and interrupt_after(scene.scene_id):
                raise RenderInterrupted(scene.scene_id,
                                        str(self.state_file))

        full_mix = np.concatenate(rendered_parts)
        if len(full_mix) != cfg.total_samples:
            raise MixerError("E_INTERNAL",
                             f"混音长度 {len(full_mix)} 与时间轴 "
                             f"{cfg.total_samples} 不一致")

        final_mix, global_gain_db, iterations = \
            self._final_loudness_and_limit(full_mix)
        frames_written = write_wav_24bit(
            self.output_path, final_mix, cfg.sample_rate)
        if frames_written != cfg.total_samples:
            raise MixerError("E_OUTPUT_IO",
                             "输出帧数与场景配置总帧数不一致",
                             {"expected": cfg.total_samples,
                              "actual": frames_written})

        report = self._build_report(
            final_mix, scene_reports, crossfade_events, resumed,
            global_gain_db, iterations)
        save_checkpoint(self.state_file, Checkpoint(
            state_version=1, config_version=cfg.version,
            output_path=str(self.output_path),
            sample_rate=cfg.sample_rate,
            total_samples=cfg.total_samples,
            completed_scenes=completed_ids,
            completed_samples=completed_samples,
            scene_reports=scene_reports, crossfades=crossfade_events))
        self._cleanup_parts(len(cfg.scenes))
        return MixResult(str(self.output_path), report, resumed=resumed)

    # ------------------------------------------------------------------
    def _part_file(self, index: int) -> Path:
        return self.parts_dir / f"scene_{index:03d}.npy"

    def _cleanup_parts(self, count: int) -> None:
        for index in range(count):
            part = self._part_file(index)
            if part.exists():
                part.unlink()
        if self.parts_dir.exists():
            try:
                self.parts_dir.rmdir()
            except OSError:
                pass

    def _persist(self, previous: Optional[Checkpoint],
                 completed_ids: List[str], completed_samples: int,
                 scene_reports: List[Dict[str, Any]],
                 crossfade_events: List[Dict[str, Any]]) -> None:
        save_checkpoint(self.state_file, Checkpoint(
            state_version=1, config_version=self.config.version,
            output_path=str(self.output_path),
            sample_rate=self.config.sample_rate,
            total_samples=self.config.total_samples,
            completed_scenes=completed_ids,
            completed_samples=completed_samples,
            scene_reports=scene_reports,
            crossfades=crossfade_events))

    def _render_scene(self, scene: SceneSpec, start: int, end: int
                      ) -> tuple:
        length = end - start
        edge_fade = int(EDGE_FADE_MS / 1000.0 * self.config.sample_rate)
        mix = np.zeros(length, dtype=np.float64)
        track_gains: Dict[str, Dict[str, float]] = {}
        for name in scene.active_tracks:
            segment = self.track_data[name][start:end].copy()
            weighted_segment = self.weighted_tracks[name][start:end]
            current = self._loudness_from_weighted(weighted_segment)
            linear, gain_db = lufs_to_linear_gain(current,
                                                   scene.target_lufs)
            shaped = apply_fade_in_out(segment, edge_fade) * linear
            mix += shaped
            track_gains[name] = {
                "source_lufs": _finite(current),
                "normalization_gain_db": round(gain_db, 4),
            }
        return mix, track_gains

    def _loudness_from_weighted(self, weighted: np.ndarray) -> float:
        mean_power = _gated_mean_square(weighted, self.config.sample_rate)
        if mean_power is None or mean_power <= 0.0:
            return float("-inf")
        return OFFSET_DB + 10.0 * np.log10(mean_power)

    def _apply_crossfade(self, previous: SceneSpec, current: SceneSpec,
                         outgoing_part: np.ndarray,
                         incoming_part: np.ndarray, spf: int
                         ) -> Dict[str, Any]:
        cfg = self.config
        total = int(round(current.crossfade_ms / 1000.0
                          * cfg.sample_rate))
        half = total // 2
        side = max(1, min(half, min(len(outgoing_part),
                                     len(incoming_part))))
        gains = crossfade_gains(2 * side, current.crossfade_type)
        length = 2 * side
        boundary = previous.end_frame * spf
        outgoing_part[len(outgoing_part) - side:] *= gains["out"][:side]
        incoming_part[:side] *= gains["in"][side:]
        return {
            "from_scene": previous.scene_id,
            "to_scene": current.scene_id,
            "boundary_frame": previous.end_frame,
            "boundary_sample": boundary,
            "crossfade_ms": current.crossfade_ms,
            "crossfade_samples": length,
            "curve_type": current.crossfade_type,
            "start_sample": boundary - side,
            "end_sample": boundary + side - 1,
        }

    def _bus_normalize(self, scene: SceneSpec, mix: np.ndarray
                       ) -> tuple:
        measured = self._loudness_from_weighted(k_weight(mix, self.fir))
        if not np.isfinite(measured):
            return 0.0, measured
        trim_db = scene.target_lufs - measured
        mix *= 10.0 ** (trim_db / 20.0)
        after = self._loudness_from_weighted(k_weight(mix, self.fir))
        return round(trim_db, 4), after

    def _final_loudness_and_limit(self, full_mix: np.ndarray) -> tuple:
        cfg = self.config
        weights = np.array([
            (s.end_frame - s.start_frame) for s in cfg.scenes],
            dtype=np.float64)
        weights /= weights.sum()
        global_target = float(np.sum(
            weights * np.array([s.target_lufs for s in cfg.scenes])))
        total_gain_db = 0.0
        iterations = 0
        current = full_mix.copy()
        last_measured = float("-inf")
        for iterations in range(1, MAX_LOUDNESS_ITERATIONS + 1):
            limited = limit(current, TRUE_PEAK_CEILING_DBTP,
                            cfg.sample_rate)
            last_measured = self._loudness_from_weighted(
                k_weight(limited.output, self.fir))
            if not np.isfinite(last_measured):
                break
            error = global_target - last_measured
            if abs(error) <= INNER_TARGET_TOLERANCE_LU:
                current = limited.output
                break
            makeup = 10.0 ** (error / 20.0)
            current = limited.output * makeup
            total_gain_db += 20.0 * np.log10(makeup)
        else:
            limited = limit(current, TRUE_PEAK_CEILING_DBTP,
                            cfg.sample_rate)
            current = limited.output

        final = limit(current, TRUE_PEAK_CEILING_DBTP, cfg.sample_rate)
        if final.true_peak_db > TRUE_PEAK_CEILING_DBTP + 1e-6:
            raise MixerError("E_LIMITER",
                             "限制后真峰值仍超过 -1 dBTP",
                             {"true_peak_db": final.true_peak_db})
        final_measured = self._loudness_from_weighted(
            k_weight(final.output, self.fir))
        if abs(final_measured - global_target) > LOUDNESS_TOLERANCE_LU \
                and np.isfinite(final_measured):
            raise MixerError(
                "E_LIMITER",
                "限制后整体响度偏差超过 ±0.5 LU",
                {"measured_lufs": final_measured,
                 "target_lufs": global_target})
        return final.output, round(total_gain_db, 4), iterations

    def _scene_report(self, scene: SceneSpec, index: int,
                      start: int, end: int, mix: np.ndarray,
                      track_gains: Dict[str, Dict[str, float]],
                      trim_db: float, measured_lufs: float
                      ) -> Dict[str, Any]:
        return {
            "scene_index": index,
            "scene_id": scene.scene_id,
            "name": scene.name,
            "start_frame": scene.start_frame,
            "end_frame": scene.end_frame,
            "start_sample": start,
            "end_sample": end,
            "duration_samples": end - start,
            "target_lufs": scene.target_lufs,
            "actual_lufs": _finite(measured_lufs),
            "sample_peak": float(np.max(np.abs(mix))) if len(mix) else 0.0,
            "bus_trim_db": trim_db,
            "tracks": track_gains,
            "alignment_error_samples": 0,
        }

    def _build_report(self, final_mix: np.ndarray,
                     scene_reports: List[Dict[str, Any]],
                     crossfade_events: List[Dict[str, Any]],
                     resumed: bool, global_gain_db: float,
                     iterations: int) -> Dict[str, Any]:
        cfg = self.config
        integrated = self._loudness_from_weighted(
            k_weight(final_mix, self.fir))
        tp = true_peak_db(final_mix)
        spf = cfg.samples_per_frame
        expected_samples = cfg.total_frames * spf
        return {
            "engine_version": "1.0.0",
            "status": "completed",
            "resumed": resumed,
            "config_version": cfg.version,
            "output": {
                "path": str(self.output_path),
                "sample_rate": cfg.sample_rate,
                "channels": 1,
                "subtype": "PCM_24",
                "total_frames": cfg.total_frames,
                "total_samples": len(final_mix),
                "duration_seconds": len(final_mix) / cfg.sample_rate,
            },
            "loudness": {
                "integrated_lufs": _finite(integrated),
                "true_peak_db": tp,
                "true_peak_ceiling_db": TRUE_PEAK_CEILING_DBTP,
                "global_makeup_gain_db": global_gain_db,
                "limiter_iterations": iterations,
            },
            "timeline": {
                "expected_samples": expected_samples,
                "actual_samples": len(final_mix),
                "alignment_error_samples": len(final_mix)
                - expected_samples,
                "samples_per_frame": spf,
            },
            "scenes": scene_reports,
            "crossfades": crossfade_events,
            "checkpoint": {
                "state_file": str(self.state_file),
                "completed_scenes":
                    [entry["scene_id"] for entry in scene_reports],
            },
        }


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else -np.inf
