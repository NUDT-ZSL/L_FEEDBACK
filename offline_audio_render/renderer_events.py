"""Per-event streaming playback, parameter curves and spatial expansion."""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from .channels import adapt_block
from .curves import OnePoleLowPass, db_to_gain, lowpass_alpha
from .random_engine import EventVariants
from .scene import EventSpec, Scene
from .sources import LinearResampler, SourceInfo, open_source
from .spatial import distance_gain, pan_gains


@dataclass
class EventRenderRecord:
    event_id: str
    source: str
    start_sample: int
    end_sample: int
    applied_gain_db: float
    seed_value: int
    skipped: bool = False
    warnings: list[str] = field(default_factory=list)


class EventPlayer:
    def __init__(
        self,
        spec: EventSpec,
        source_path,
        info: SourceInfo,
        variants: EventVariants,
        scene: Scene,
        record: EventRenderRecord,
        read_block: int,
    ) -> None:
        self.spec = spec
        self._source_path = source_path
        self._info = info
        self._variants = variants
        self._scene = scene
        self._record = record
        self._read_block = read_block

        nominal_start = int(round(spec.time * scene.sample_rate))
        start_shift_frames = int(
            round(variants.start_offset_s * scene.sample_rate)
        )
        self._source_start = max(0, -start_shift_frames)
        self.start_sample = max(0, nominal_start + start_shift_frames)
        self._timeline_trim = max(0, -nominal_start - start_shift_frames)
        # Output frames cover the full source at the pitch-shifted rate.
        source_available = max(0, info.frames - self._source_start)
        rendered_frames = int(np.ceil(source_available / variants.pitch_factor))
        self.output_frames = max(0, rendered_frames - self._timeline_trim)
        self._consumed = 0
        self._opened = False
        self._resampler: LinearResampler | None = None
        self._channel_filters: list[OnePoleLowPass] | None = None
        self._spatial = self._compute_spatial()

    def _compute_spatial(self) -> np.ndarray:
        scene = self._scene
        pose = scene.trajectory.pose_at(self.spec.time)
        source_position = np.asarray(self.spec.position, dtype=np.float64)
        from .spatial import relative_azimuth_distance

        azimuth, distance = relative_azimuth_distance(source_position, pose)
        attenuation = distance_gain(distance, scene.min_distance)
        return pan_gains(azimuth, scene.layout) * attenuation

    def render_range(self, timeline_start: int, frames: int) -> np.ndarray | None:
        timeline_end = timeline_start + frames
        if timeline_end <= self.start_sample or timeline_start >= self.start_sample + self.output_frames:
            return None
        output_channels = self._scene.channels
        output = np.zeros((frames, output_channels), dtype=np.float64)
        dest_offset = max(0, self.start_sample - timeline_start)
        relative_start = max(0, timeline_start - self.start_sample)
        if relative_start >= self.output_frames:
            return None
        take = min(
            timeline_end - self.start_sample, self.output_frames
        ) - relative_start
        if take <= 0:
            return None
        if not self._opened:
            self._open()
        if self._resampler is None:
            return None
        if self._timeline_trim and self._consumed == 0:
            self._resampler.pull(self._timeline_trim)
            self._consumed += self._timeline_trim
        # This player is only touched inside its active range, so the
        # resampler position always advances sequentially block by block.
        expected = relative_start
        if self._consumed < expected:
            gap = expected - self._consumed
            self._resampler.pull(gap)
            self._consumed += gap
        block = self._resampler.pull(take)
        self._consumed += take
        processed = self._process_block(block, relative_start, take)
        output[dest_offset : dest_offset + take] = processed
        return output

    def _open(self) -> None:
        self._opened = True
        sound, _info, warning = open_source(
            self._source_path, self._scene.sample_rate
        )
        if sound is None:
            self._record.skipped = True
            if warning:
                self._record.warnings.append(warning)
            self._resampler = None
            return
        self._resampler = LinearResampler(
            sound=sound,
            channels=self._info.channels,
            rate=self._variants.pitch_factor,
            start_frame=self._source_start,
            read_block=self._read_block,
        )

    def _process_block(
        self, block: np.ndarray, event_offset: int, frames: int
    ) -> np.ndarray:
        normalised_t = (
            np.arange(event_offset, event_offset + frames, dtype=np.float64)
            / max(1, self.output_frames)
        )
        gain_db = self.spec.gain_curve.evaluate(normalised_t)
        gain_db += self._variants.gain_jitter_db
        gain = db_to_gain(gain_db)
        cutoff = self.spec.lowpass_curve.evaluate(normalised_t)
        alpha = lowpass_alpha(cutoff, self._scene.sample_rate)

        block = block * gain[:, None]
        if self._channel_filters is None:
            self._channel_filters = [
                OnePoleLowPass() for _ in range(block.shape[1])
            ]
        filtered = np.empty_like(block)
        for channel in range(block.shape[1]):
            filtered[:, channel] = self._channel_filters[channel].process(
                block[:, channel], alpha
            )

        adapted, warning = adapt_block(
            filtered,
            self._info.channels,
            self._scene.layout,
            self._spatial,
        )
        if warning is not None:
            message = f"{warning}: event {self.spec.event_id}"
            if message not in self._record.warnings:
                self._record.warnings.append(message)
        return adapted

    def close(self) -> None:
        if self._resampler is not None:
            self._resampler.close()

    def collect_warnings(self) -> list[str]:
        return list(self._record.warnings)

    def working_bytes(self) -> int:
        if self._resampler is None:
            return 0
        return self._resampler.working_bytes()
