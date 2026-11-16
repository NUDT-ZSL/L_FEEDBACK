"""Streaming first-person spatial render pipeline."""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, List, Optional, Sequence, TextIO, Tuple

import numpy as np
import soundfile as sf

from .binaural import BinauralRenderer, CueBlock
from .compression import Compressor, CompressorConfig
from .errors import ParameterError, RenderRuntimeError, TrackLoadError
from .geometry import vectorized_spatial_cues
from .loudness import LoudnessMeter
from .occlusion import OcclusionFilter, occlusion_curves
from .resampling import SincTrackRenderer, load_audio_track
from .scene import SceneSpec, SourceSpec


@dataclass
class RenderResult:
    duration_seconds: float
    peak_dbfs: float
    integrated_lufs: Optional[float]
    normalize_gain_db: float
    skipped_tracks: Tuple[str, ...]


@dataclass
class PreparedTrack:
    spec: SourceSpec
    resampler: SincTrackRenderer
    occlusion_filter: OcclusionFilter
    binaural: BinauralRenderer
    rate_smoother: Optional["RateSmoother"]


class RateSmoother:
    """One-pole rate interpolation that preserves exact clock continuity."""

    def __init__(self, sample_rate: int, smoothing_seconds: float = 0.004) -> None:
        self.coefficient = math.exp(
            -1.0 / max(1.0, smoothing_seconds * float(sample_rate))
        )
        self.state: Optional[float] = None

    def process(self, rates: np.ndarray) -> np.ndarray:
        rates = np.asarray(rates, dtype=np.float64)
        if len(rates) == 0:
            return rates.copy()
        if self.state is None:
            self.state = float(rates[0])
        output = np.empty(len(rates), dtype=np.float64)
        state = self.state
        for index, target in enumerate(rates):
            state = float(target) + self.coefficient * (state - float(target))
            output[index] = state
        self.state = state
        return output


def radial_doppler_rates(
    source_positions: np.ndarray,
    listener_positions: np.ndarray,
    source_velocities: np.ndarray,
    listener_velocities: np.ndarray,
    speed_of_sound: float,
) -> np.ndarray:
    """Return c / (c + outward radial velocity) for every sample."""
    relative = source_positions - listener_positions
    distance = np.maximum(
        np.linalg.norm(relative, axis=1), 1.0e-12
    )
    unit = relative / distance[:, None]
    radial = np.sum(
        (source_velocities - listener_velocities) * unit, axis=1
    )
    denominator = speed_of_sound + radial
    safe_denominator = np.where(
        np.abs(denominator) < 1.0e-6,
        np.sign(denominator) * 1.0e-6 + 1.0e-12,
        denominator,
    )
    rates = speed_of_sound / safe_denominator
    return np.clip(rates, 0.25, 4.0)


class _ProgressReporter:
    def __init__(
        self,
        stream: TextIO,
        duration_samples: int,
        sample_rate: int,
    ) -> None:
        self.stream = stream
        self.duration_samples = duration_samples
        self.sample_rate = sample_rate
        self.next_second = 1
        self.last_percentage = -1.0

    def update(self, rendered_samples: int) -> None:
        duration_seconds = self.duration_samples / float(self.sample_rate)
        rendered_seconds = rendered_samples / float(self.sample_rate)
        while (
            self.next_second <= math.ceil(duration_seconds)
            and rendered_seconds + 1.0e-12 >= self.next_second
        ):
            percentage = (
                100.0
                * min(rendered_samples, self.duration_samples)
                / float(max(1, self.duration_samples))
            )
            self.stream.write("progress: %.2f%%\n" % percentage)
            self.stream.flush()
            self.last_percentage = percentage
            self.next_second += 1

    def finish(self) -> None:
        if self.last_percentage < 100.0:
            self.stream.write("progress: 100.00%\n")
            self.stream.flush()


def _infer_duration(
    scene: SceneSpec, loaded_lengths: Sequence[Tuple[SourceSpec, int]]
) -> int:
    if scene.duration_samples is not None:
        return int(scene.duration_samples)
    candidates = [scene.timeline_end_sample()]
    for source, length in loaded_lengths:
        candidates.append(source.start_sample + length)
    duration = max(candidates)
    if duration <= 0:
        raise ParameterError(
            "cannot infer duration: provide duration_samples in the scene"
        )
    return int(duration)


def _prepare_tracks(
    scene: SceneSpec, stderr: TextIO
) -> Tuple[List[PreparedTrack], Tuple[str, ...], List[Tuple[SourceSpec, int]]]:
    prepared: List[PreparedTrack] = []
    skipped: List[str] = []
    loaded_lengths: List[Tuple[SourceSpec, int]] = []
    taps = 32
    for source in scene.sources:
        try:
            samples = load_audio_track(
                source.file_path, scene.sample_rate, taps=taps
            )
        except TrackLoadError as exc:
            skipped.append(source.name)
            stderr.write(
                "warning: skipping track '%s': %s\n" % (source.name, exc)
            )
            stderr.flush()
            continue
        loaded_lengths.append((source, len(samples)))
        prepared.append(
            PreparedTrack(
                spec=source,
                resampler=SincTrackRenderer(
                    samples, loop=source.loop, taps=taps
                ),
                occlusion_filter=OcclusionFilter(scene.sample_rate),
                binaural=BinauralRenderer(
                    scene.sample_rate, head_radius=scene.head_radius
                ),
                rate_smoother=RateSmoother(scene.sample_rate),
            )
        )
    return prepared, tuple(skipped), loaded_lengths


def _estimate_disk_requirement(duration_samples: int) -> int:
    raw_float_bytes = duration_samples * 2 * 4
    final_pcm_bytes = duration_samples * 2 * 2
    margin = max(1024 * 1024, int(0.05 * (raw_float_bytes + final_pcm_bytes)))
    return raw_float_bytes + final_pcm_bytes + margin


def _open_raw_temp(output_path: Path) -> Tuple[Path, BinaryIO]:
    parent = output_path.parent
    try:
        handle = tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=".spatial-render-",
            suffix=".f32",
            dir=str(parent),
            delete=False,
        )
    except OSError as exc:
        raise RenderRuntimeError(
            "cannot create temporary render file: %s" % exc
        ) from exc
    return Path(handle.name), handle


def _cleanup(path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _render_pass(
    scene: SceneSpec,
    tracks: Sequence[PreparedTrack],
    duration_samples: int,
    raw_handle: BinaryIO,
    compressor: Optional[Compressor],
    meter: LoudnessMeter,
    stderr: TextIO,
) -> None:
    progress = _ProgressReporter(
        stderr, duration_samples, scene.sample_rate
    )
    for start in range(0, duration_samples, scene.frame_length):
        length = min(scene.frame_length, duration_samples - start)
        sample_times = np.arange(
            start, start + length, dtype=np.int64
        )
        quaternions = scene.head.rotation.sample(sample_times)
        listener_positions = scene.head.position.sample(sample_times)
        listener_velocity = (
            scene.head.position.velocity_per_sample(sample_times)
            * float(scene.sample_rate)
        )
        mix = np.zeros((length, 2), dtype=np.float64)
        for track in tracks:
            source_positions = track.spec.position.sample(sample_times)
            relative = source_positions - listener_positions
            (
                lateral,
                itd,
                left_ild,
                right_ild,
                elevation,
            ) = vectorized_spatial_cues(
                relative,
                quaternions,
                head_radius=scene.head_radius,
                speed_of_sound=scene.speed_of_sound,
            )
            cues = CueBlock(
                itd_seconds=itd,
                left_ild=left_ild,
                right_ild=right_ild,
                left_shadow=np.maximum(0.0, lateral),
                right_shadow=np.maximum(0.0, -lateral),
                elevation=elevation,
                lateral=lateral,
            )
            source_velocity = (
                track.spec.position.velocity_per_sample(sample_times)
                * float(scene.sample_rate)
            )
            raw_rates = radial_doppler_rates(
                source_positions,
                listener_positions,
                source_velocity,
                listener_velocity,
                scene.speed_of_sound,
            )
            assert track.rate_smoother is not None
            rates = track.rate_smoother.process(raw_rates)

            active_offset = max(0, track.spec.start_sample - start)
            mono = np.zeros(length, dtype=np.float64)
            if active_offset < length:
                active_rates = rates[active_offset:]
                mono[active_offset:] = track.resampler.render(active_rates)

            _, cutoffs, occlusion_gains = occlusion_curves(
                scene.occluders,
                listener_positions,
                source_positions,
                scene.sample_rate,
                scene.occlusion_base_cutoff_hz,
            )
            mono = track.occlusion_filter.process(
                mono, cutoffs, occlusion_gains
            )
            stereo = track.binaural.process(mono, cues)
            distance = np.linalg.norm(relative, axis=1)
            distance_gain = scene.reference_distance / np.maximum(
                distance, scene.reference_distance
            )
            mix += stereo * (
                distance_gain[:, None] * float(track.spec.gain)
            )
        if compressor is not None:
            mix = compressor.process(mix)
        meter.process(mix)
        raw_handle.write(
            mix.astype("<f4", copy=False).tobytes(order="C")
        )
        progress.update(start + length)
    progress.finish()


def _write_final_wav(
    raw_path: Path,
    output_path: Path,
    sample_rate: int,
    duration_samples: int,
    gain_db: float,
) -> Tuple[float, bool]:
    final_temp = output_path.parent / (
        ".spatial-render-%s.wav" % uuid.uuid4().hex
    )
    gain = 10.0 ** (gain_db / 20.0)
    peak = 0.0
    clipped = False
    try:
        with raw_path.open("rb") as raw_handle:
            with sf.SoundFile(
                str(final_temp),
                mode="w",
                samplerate=sample_rate,
                channels=2,
                format="WAV",
                subtype="PCM_16",
            ) as output_handle:
                remaining = duration_samples
                frames_per_read = 262144
                while remaining > 0:
                    frames = min(remaining, frames_per_read)
                    raw = raw_handle.read(frames * 2 * 4)
                    expected = frames * 2 * 4
                    if len(raw) != expected:
                        raise RenderRuntimeError(
                            "temporary render file ended unexpectedly"
                        )
                    data = np.frombuffer(raw, dtype="<f4").reshape(
                        frames, 2
                    ).astype(np.float64)
                    scaled = data * gain
                    if scaled.size:
                        block_peak = float(np.max(np.abs(scaled)))
                        peak = max(peak, min(1.0, block_peak))
                        clipped = clipped or block_peak > 1.0
                    pcm = np.rint(
                        np.clip(scaled, -1.0, 1.0) * 32767.0
                    ).astype("<i2")
                    output_handle.write(pcm)
                    remaining -= frames
        os.replace(str(final_temp), str(output_path))
    except RenderRuntimeError:
        _cleanup(final_temp)
        raise
    except Exception as exc:
        _cleanup(final_temp)
        raise RenderRuntimeError(
            "cannot write output WAV %s: %s" % (output_path, exc)
        ) from exc
    peak_dbfs = (
        20.0 * math.log10(peak) if peak > 1.0e-12 else -math.inf
    )
    return peak_dbfs, clipped


def render_scene(
    scene: SceneSpec,
    output_path: Path,
    normalize_lufs: Optional[float] = None,
    compressor_config: Optional[CompressorConfig] = None,
    stderr: Optional[TextIO] = None,
) -> RenderResult:
    stderr = stderr if stderr is not None else sys.stderr
    output_path = Path(output_path)
    if compressor_config is not None:
        compressor_config.validate()
    if normalize_lufs is not None and (
        not math.isfinite(normalize_lufs)
        or not -70.0 <= normalize_lufs <= 0.0
    ):
        raise ParameterError(
            "normalize loudness target must be in [-70, 0] LUFS"
        )

    tracks, skipped, loaded_lengths = _prepare_tracks(scene, stderr)
    duration_samples = _infer_duration(scene, loaded_lengths)
    if duration_samples <= 0:
        raise ParameterError("scene duration must be positive")
    parent = output_path.parent
    if not parent.exists() or not parent.is_dir():
        raise RenderRuntimeError(
            "output directory does not exist: %s" % parent
        )
    if output_path.exists() and output_path.is_dir():
        raise RenderRuntimeError(
            "output path is a directory: %s" % output_path
        )
    try:
        free_bytes = shutil.disk_usage(str(parent)).free
    except OSError as exc:
        raise RenderRuntimeError(
            "cannot inspect output disk space: %s" % exc
        ) from exc
    required = _estimate_disk_requirement(duration_samples)
    if free_bytes < required:
        raise RenderRuntimeError(
            "insufficient disk space: need %d bytes, available %d"
            % (required, free_bytes)
        )

    raw_path: Optional[Path] = None
    raw_handle: Optional[BinaryIO] = None
    try:
        raw_path, raw_handle = _open_raw_temp(output_path)
        compressor = (
            Compressor(scene.sample_rate, compressor_config)
            if compressor_config is not None
            else None
        )
        meter = LoudnessMeter(scene.sample_rate, channels=2)
        _render_pass(
            scene,
            tracks,
            duration_samples,
            raw_handle,
            compressor,
            meter,
            stderr,
        )
        raw_handle.flush()
        try:
            os.fsync(raw_handle.fileno())
        except OSError:
            pass
        raw_handle.close()
        raw_handle = None
        integrated = meter.integrated_loudness()
        gain_db = 0.0
        if normalize_lufs is not None:
            if integrated is None or not math.isfinite(integrated):
                stderr.write(
                    "warning: loudness normalization skipped because "
                    "integrated loudness is undefined\n"
                )
            else:
                gain_db = float(normalize_lufs - integrated)
        peak_dbfs, clipped = _write_final_wav(
            raw_path,
            output_path,
            scene.sample_rate,
            duration_samples,
            gain_db,
        )
        if clipped:
            stderr.write(
                "warning: output clipped at 0 dBFS during 16-bit conversion\n"
            )
        duration_seconds = duration_samples / float(scene.sample_rate)
        reported_integrated = (
            integrated + gain_db
            if integrated is not None and math.isfinite(integrated)
            else None
        )
        integrated_text = (
            "%.2f LUFS" % reported_integrated
            if reported_integrated is not None
            else "undefined"
        )
        stderr.write(
            "finished: duration=%.3f s, peak=%.2f dBFS, integrated=%s\n"
            % (duration_seconds, peak_dbfs, integrated_text)
        )
        stderr.flush()
        return RenderResult(
            duration_seconds=duration_seconds,
            peak_dbfs=peak_dbfs,
            integrated_lufs=reported_integrated,
            normalize_gain_db=gain_db,
            skipped_tracks=skipped,
        )
    except RenderRuntimeError:
        if raw_handle is not None:
            try:
                raw_handle.close()
            except OSError:
                pass
        _cleanup(raw_path)
        raise
    except (OSError, ValueError, MemoryError) as exc:
        if raw_handle is not None:
            try:
                raw_handle.close()
            except OSError:
                pass
        _cleanup(raw_path)
        raise RenderRuntimeError("render failed: %s" % exc) from exc
    finally:
        if raw_handle is not None:
            try:
                raw_handle.close()
            except OSError:
                pass
        _cleanup(raw_path)
