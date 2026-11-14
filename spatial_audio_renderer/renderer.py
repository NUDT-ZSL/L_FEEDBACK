"""Offline frame rendering, resumable state, WAV writing, and manifests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

from .dsp import db_to_gain
from .errors import BusinessError, invalid_scene
from .reverb import convolve_shared_bus
from .scene import Scene, safe_track_filename
from .snapshot import LoadedSnapshot, SnapshotStore
from .spatial import SpatialPreparation, emitter_state, prepare_spatial


SUBTYPE_BY_BIT_DEPTH = {
    8: "PCM_U8",
    16: "PCM_16",
    24: "PCM_24",
    32: "PCM_32",
}


@dataclass
class RenderOptions:
    snapshot_interval: int
    max_snapshots: int = 8
    force_resume: bool = False
    simulate_crash_after_frames: Optional[int] = None


@dataclass
class RenderResult:
    mix_path: Path
    dry_track_paths: List[Path]
    manifest_path: Path
    resumed_from: Optional[Path]
    resumed_from_frame: int
    total_frames: int
    total_samples: int
    snapshots_written: int


def output_filenames(scene: Scene) -> List[str]:
    names: List[str] = []
    used: Dict[str, str] = {}
    for emitter in scene.emitters:
        base = safe_track_filename(emitter.emitter_id)
        if base in used:
            digest = hashlib.sha256(emitter.emitter_id.encode("utf-8")).hexdigest()[:10]
            base = "dry_%s_%s.wav" % (base[4:-4], digest)
        if base in used:
            raise invalid_scene("could not create unique dry-track filename",
                                emitter_id=emitter.emitter_id)
        used[base] = emitter.emitter_id
        names.append(base)
    return names


def _atomic_wav_write(path: Path, data: np.ndarray, sample_rate: int,
                      bit_depth: int) -> None:
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".%s." % path.name, suffix=".tmp",
            dir=str(path.parent), delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        sf.write(
            str(temporary_path),
            np.asarray(data, dtype=np.float64),
            int(sample_rate),
            format="WAV",
            subtype=SUBTYPE_BY_BIT_DEPTH[bit_depth],
            endian="LITTLE",
        )
        os.replace(str(temporary_path), str(path))
    except Exception as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass
        reason = getattr(exc, "strerror", None) or str(exc)
        raise BusinessError(
            "INVALID_OUTPUT_DIR",
            "WAV file cannot be written",
            {"path": str(path), "reason": reason, "exception_type": type(exc).__name__},
        ) from exc


class OfflineRenderer:
    def __init__(self, scene: Scene, output_dir: Path, snapshot_dir: Path,
                 options: RenderOptions) -> None:
        self.scene = scene
        self.output_dir = output_dir
        self.snapshot_dir = snapshot_dir
        self.options = options
        self.frame_size = scene.frame_size
        self.total_frames = int(math.ceil(scene.total_samples / float(self.frame_size)))
        self.preparation: SpatialPreparation = prepare_spatial(scene)
        self.track_filenames = output_filenames(scene)
        self.store = SnapshotStore(
            snapshot_dir,
            scene.fingerprint,
            self.frame_size,
            self.total_frames,
            scene.total_samples,
            options.max_snapshots,
        )
        self.dry_mix = np.zeros((2, scene.total_samples), dtype=np.float64)
        self.wet_lanes = np.zeros(
            (self.preparation.lane_count, scene.total_samples), dtype=np.float64
        )
        self.resumed_from: Optional[Path] = None
        self.resumed_from_frame = 0
        self.start_frame = 0
        self.snapshots_written = 0
        self.request_interrupt = False

        scan = self.store.scan(self.preparation.lane_count, options.force_resume)
        if scan.snapshot is not None:
            self._restore(scan.snapshot)

    def _restore(self, snapshot: LoadedSnapshot) -> None:
        sample_frame = snapshot.sample_frame
        self.dry_mix[:, :sample_frame] = snapshot.dry_mix
        self.wet_lanes[:, :sample_frame] = snapshot.wet_lanes
        self.start_frame = snapshot.render_frame
        self.resumed_from = snapshot.manifest_path
        self.resumed_from_frame = snapshot.render_frame

    def _emitter_states(self, sample_frame: int) -> List[Dict[str, Any]]:
        return [
            emitter_state(emitter, sample_frame)
            for emitter in self.preparation.emitters
        ]

    def _reverb_state(self) -> Dict[str, Any]:
        return {
            "precomputed": True,
            "lane_count": self.preparation.lane_count,
            "lanes": [lane.state() for lane in self.preparation.lanes],
        }

    def _save_snapshot(self, completed_frames: int, sample_frame: int) -> Path:
        path = self.store.write(
            completed_frames,
            sample_frame,
            self.dry_mix,
            self.wet_lanes,
            self._emitter_states(sample_frame),
            self._reverb_state(),
        )
        self.snapshots_written += 1
        return path

    def _render_frame(self, frame_index: int) -> None:
        start = frame_index * self.frame_size
        end = min(start + self.frame_size, self.scene.total_samples)
        if start >= end:
            return
        for emitter in self.preparation.emitters:
            chunk = emitter.dry_track[start:end]
            dry_gain = 1.0
            if emitter.region is not None:
                dry_gain = emitter.region.reverb.dry
            left = chunk * (dry_gain * emitter.pan_left)
            right = chunk * (dry_gain * emitter.pan_right)
            self.dry_mix[0, start:end] += left
            self.dry_mix[1, start:end] += right
            if emitter.send_track is not None and emitter.lane_index is not None:
                self.wet_lanes[emitter.lane_index, start:end] += (
                    emitter.send_track[start:end] * emitter.energy_weight
                )

    def _raise_if_interrupted(self, completed_frames: int, sample_frame: int) -> None:
        if not self.request_interrupt:
            return
        self._save_snapshot(completed_frames, sample_frame)
        raise BusinessError(
            "RENDER_INTERRUPTED",
            "rendering was interrupted; a resumable snapshot was written",
            {
                "render_frame": completed_frames,
                "sample_frame": sample_frame,
                "snapshot_directory": str(self.snapshot_dir),
            },
        )

    def _render_frames(self) -> None:
        interval = max(1, int(self.options.snapshot_interval))
        crash_after = self.options.simulate_crash_after_frames
        for frame_index in range(self.start_frame, self.total_frames):
            self._render_frame(frame_index)
            completed_frames = frame_index + 1
            sample_frame = min(completed_frames * self.frame_size, self.scene.total_samples)

            self._raise_if_interrupted(completed_frames, sample_frame)
            force_snapshot = (
                crash_after is not None and completed_frames >= crash_after
            )
            should_snapshot = (
                completed_frames % interval == 0
                or completed_frames == self.total_frames
                or force_snapshot
            )
            if should_snapshot:
                self._save_snapshot(completed_frames, sample_frame)
            if force_snapshot:
                # Simulates an uncatchable process crash while leaving a complete
                # snapshot for the automatic recovery path.
                os._exit(137)

    def _wet_bus(self) -> np.ndarray:
        wet = np.zeros((2, self.scene.total_samples), dtype=np.float64)
        for lane_index, lane in enumerate(self.preparation.lanes):
            left, right = convolve_shared_bus(
                self.wet_lanes[lane_index],
                lane.impulse,
                self.scene.total_samples,
                lane.wet_gain,
            )
            wet[0] += left
            wet[1] += right
        return wet

    def _write_manifest(self, mix_path: Path, dry_paths: List[Path],
                        wet_bus: np.ndarray) -> Path:
        mix_peak = float(np.max(np.abs(self.dry_mix + wet_bus))) if self.scene.total_samples else 0.0
        mix_rms = (
            float(np.sqrt(np.mean((self.dry_mix + wet_bus) ** 2)))
            if self.scene.total_samples else 0.0
        )
        payload: Dict[str, Any] = {
            "format": "spatial-audio-render-manifest",
            "version": 1,
            "status": "complete",
            "scene_path": str(self.scene.source_path),
            "scene_fingerprint": self.scene.fingerprint,
            "scene_sha256": self.scene.raw_sha256,
            "sample_rate": self.scene.sample_rate,
            "bit_depth": self.scene.bit_depth,
            "channels": 2,
            "total_samples": self.scene.total_samples,
            "duration_seconds": self.scene.total_samples / float(self.scene.sample_rate),
            "frame_size": self.frame_size,
            "total_frames": self.total_frames,
            "snapshot_interval_frames": self.options.snapshot_interval,
            "resumed_from": str(self.resumed_from) if self.resumed_from else None,
            "resumed_from_frame": self.resumed_from_frame,
            "mix_file": mix_path.name,
            "dry_tracks": [
                {
                    "emitter_id": emitter.spec.emitter_id,
                    "file": path.name,
                    "distance_gain": emitter.distance_gain,
                    "pan": emitter.pan,
                    "region_id": emitter.region.region_id if emitter.region else None,
                    "occlusion_cutoff_hz": emitter.occlusion_cutoff_hz,
                }
                for emitter, path in zip(self.preparation.emitters, dry_paths)
            ],
            "reverb": self._reverb_state(),
            "pre_master_peak": mix_peak,
            "pre_master_rms": mix_rms,
        }
        manifest_path = self.output_dir / "render_manifest.json"
        try:
            temporary_path = manifest_path.with_name(".%s.tmp" % manifest_path.name)
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(str(temporary_path), str(manifest_path))
        except OSError as exc:
            raise BusinessError(
                "INVALID_OUTPUT_DIR",
                "render manifest cannot be written",
                {"path": str(manifest_path), "reason": exc.strerror or str(exc)},
            ) from exc
        return manifest_path

    def run(self) -> RenderResult:
        self._render_frames()
        if self.request_interrupt:
            raise BusinessError(
                "RENDER_INTERRUPTED",
                "rendering was interrupted before finalization",
                {"snapshot_directory": str(self.snapshot_dir)},
            )

        wet_bus = self._wet_bus()
        master_gain = db_to_gain(self.scene.master_gain_db)
        final_mix = np.clip((self.dry_mix + wet_bus) * master_gain, -1.0, 1.0)
        mix_path = self.output_dir / "mix.wav"
        _atomic_wav_write(mix_path, final_mix.T, self.scene.sample_rate, self.scene.bit_depth)

        dry_paths: List[Path] = []
        for emitter, filename in zip(self.preparation.emitters, self.track_filenames):
            path = self.output_dir / filename
            _atomic_wav_write(
                path,
                np.clip(emitter.dry_track, -1.0, 1.0),
                self.scene.sample_rate,
                self.scene.bit_depth,
            )
            dry_paths.append(path)

        manifest_path = self._write_manifest(mix_path, dry_paths, wet_bus)
        return RenderResult(
            mix_path=mix_path,
            dry_track_paths=dry_paths,
            manifest_path=manifest_path,
            resumed_from=self.resumed_from,
            resumed_from_frame=self.resumed_from_frame,
            total_frames=self.total_frames,
            total_samples=self.scene.total_samples,
            snapshots_written=self.snapshots_written,
        )
