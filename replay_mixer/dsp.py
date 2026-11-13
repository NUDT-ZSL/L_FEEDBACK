from __future__ import annotations

import bisect
import math
import sys
from array import array
from dataclasses import dataclass

from .audio import LoadedProject, LoadedTrack
from .schema import ProjectConfig
from .util import db_to_lin


PCM_SCALE = 1.0 / 32768.0


@dataclass(frozen=True)
class TimeMap:
    replay_frames: tuple[float, ...]
    source_frames: tuple[float, ...]
    slopes: tuple[float, ...]

    @classmethod
    def from_anchors(
        cls,
        anchors: tuple,
        sample_rate: int,
    ) -> "TimeMap":
        replay = tuple(anchor.replay_ms * sample_rate / 1000.0 for anchor in anchors)
        source = tuple(anchor.source_ms * sample_rate / 1000.0 for anchor in anchors)
        slopes = tuple(
            (source[index + 1] - source[index]) / (replay[index + 1] - replay[index])
            for index in range(len(replay) - 1)
        )
        return cls(replay_frames=replay, source_frames=source, slopes=slopes)

    def source_frame(self, replay_frame: float) -> float:
        last = len(self.replay_frames) - 1
        if replay_frame < self.replay_frames[0]:
            return self.source_frames[0] + (
                replay_frame - self.replay_frames[0]
            ) * self.slopes[0]
        if replay_frame >= self.replay_frames[-1]:
            return self.source_frames[-1] + (
                replay_frame - self.replay_frames[-1]
            ) * self.slopes[-1]
        index = bisect.bisect_right(self.replay_frames, replay_frame) - 1
        index = min(index, last - 1)
        return self.source_frames[index] + (
            replay_frame - self.replay_frames[index]
        ) * self.slopes[index]


@dataclass(frozen=True)
class TrimInterval:
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class PreparedCommentary:
    loaded: LoadedTrack
    time_map: TimeMap
    trim_intervals: tuple[TrimInterval, ...]
    trim_mask: bytearray
    gain_lin: float
    pan_matrix: tuple[tuple[float, ...], ...]
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class PreparedSimpleTrack:
    loaded: LoadedTrack
    gain_lin: float
    pan_matrix: tuple[tuple[float, ...], ...]
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class PreparedProject:
    config: ProjectConfig
    commentary: tuple[PreparedCommentary, ...]
    engine: PreparedSimpleTrack
    ambience: PreparedSimpleTrack


def frame_at_ms(milliseconds: float, sample_rate: int) -> int:
    return int(round(milliseconds * sample_rate / 1000.0))


def pan_matrix(
    source_channels: int,
    output_channels: int,
    pan: float,
) -> tuple[tuple[float, ...], ...]:
    """Return output-channel x input-channel pan coefficients."""
    if source_channels not in (1, 2) or output_channels not in (1, 2):
        raise ValueError("pan_matrix only supports mono/stereo")

    if output_channels == 1:
        if source_channels == 1:
            return ((1.0,),)
        if pan < 0.0:
            left_gain, right_gain = 1.0, 1.0 + pan
        else:
            left_gain, right_gain = 1.0 - pan, 1.0
        return ((0.5 * left_gain, 0.5 * right_gain),)

    if source_channels == 1:
        angle = (pan + 1.0) * math.pi / 4.0
        return ((math.cos(angle),), (math.sin(angle),))
    if pan < 0.0:
        left_gain, right_gain = 1.0, 1.0 + pan
    else:
        left_gain, right_gain = 1.0 - pan, 1.0
    return ((left_gain, 0.0), (0.0, right_gain))


def detect_silence(
    pcm: array,
    channels: int,
    sample_rate: int,
    window_ms: float,
    threshold_dbfs: float,
    min_duration_ms: float,
) -> tuple[TrimInterval, ...]:
    frame_count = len(pcm) // channels
    if frame_count == 0:
        return ()
    window_frames = max(1, int(round(window_ms * sample_rate / 1000.0)))
    threshold_power = 10.0 ** (threshold_dbfs / 10.0)
    scale = 1.0 / (32768.0 * 32768.0 * channels)

    prefix = [0.0] * (frame_count + 1)
    energy_sum = 0.0
    for frame in range(frame_count):
        offset = frame * channels
        energy = 0
        for channel in range(channels):
            sample = pcm[offset + channel]
            energy += sample * sample
        energy_sum += energy * scale
        prefix[frame + 1] = energy_sum

    low = bytearray(frame_count)
    half = window_frames // 2
    for frame in range(frame_count):
        start = max(0, frame - half)
        end = min(frame_count, start + window_frames)
        if end - start < window_frames:
            start = max(0, end - window_frames)
        width = end - start
        mean_power = (prefix[end] - prefix[start]) / width
        if mean_power < threshold_power:
            low[frame] = 1

    intervals: list[TrimInterval] = []
    index = 0
    while index < frame_count:
        if not low[index]:
            index += 1
            continue
        end = index + 1
        while end < frame_count and low[end]:
            end += 1
        duration_ms = (end - index) * 1000.0 / sample_rate
        if duration_ms > min_duration_ms:
            intervals.append(TrimInterval(index, end))
        index = end
    return tuple(intervals)


def trim_mask(frame_count: int, intervals: tuple[TrimInterval, ...]) -> bytearray:
    mask = bytearray(frame_count)
    for interval in intervals:
        mask[interval.start_frame : interval.end_frame] = b"\x01" * (
            interval.end_frame - interval.start_frame
        )
    return mask


def prepare_project(loaded: LoadedProject) -> PreparedProject:
    config = loaded.config
    total_frames = config.total_frames
    commentary: list[PreparedCommentary] = []
    for track in loaded.commentary:
        spec = track.spec
        assert spec.silence is not None
        intervals = detect_silence(
            track.pcm,
            track.channels,
            track.sample_rate,
            spec.silence.window_ms,
            spec.silence.threshold_dbfs,
            spec.silence.min_duration_ms,
        )
        commentary.append(
            PreparedCommentary(
                loaded=track,
                time_map=TimeMap.from_anchors(spec.anchors, config.sample_rate),
                trim_intervals=intervals,
                trim_mask=trim_mask(track.frames, intervals),
                gain_lin=db_to_lin(spec.gain_db),
                pan_matrix=pan_matrix(track.channels, config.output_channels, spec.pan),
                start_frame=max(
                    0,
                    min(total_frames, frame_at_ms(spec.start_ms, config.sample_rate)),
                ),
                end_frame=max(
                    0,
                    min(total_frames, frame_at_ms(spec.end_ms, config.sample_rate)),
                ),
            )
        )

    def simple(track: LoadedTrack) -> PreparedSimpleTrack:
        spec = track.spec
        start_frame = max(
            0,
            min(total_frames, frame_at_ms(spec.start_ms, config.sample_rate)),
        )
        end_frame = max(
            start_frame,
            min(total_frames, frame_at_ms(spec.end_ms, config.sample_rate)),
        )
        return PreparedSimpleTrack(
            loaded=track,
            gain_lin=db_to_lin(spec.gain_db),
            pan_matrix=pan_matrix(track.channels, config.output_channels, spec.pan),
            start_frame=start_frame,
            end_frame=end_frame,
        )

    return PreparedProject(
        config=config,
        commentary=tuple(commentary),
        engine=simple(loaded.engine),
        ambience=simple(loaded.ambience),
    )


def _add_track_sample(
    mix: list[float],
    frame: int,
    output_channels: int,
    source_values: tuple[float, ...],
    pan: tuple[tuple[float, ...], ...],
    gain: float,
) -> None:
    base = frame * output_channels
    for output_channel in range(output_channels):
        coefficients = pan[output_channel]
        value = 0.0
        for source_channel, coefficient in enumerate(coefficients):
            if coefficient:
                value += source_values[source_channel] * coefficient
        mix[base + output_channel] += value * gain


def _soft_limit(value: float, ceiling: float, knee: float) -> float:
    magnitude = abs(value)
    if magnitude <= knee:
        return value
    width = ceiling - knee
    limited = knee + width * math.tanh((magnitude - knee) / width)
    if value < 0.0:
        return -limited
    return limited


def render_premix_block(
    prepared: PreparedProject,
    start_frame: int,
    frame_count: int,
) -> array:
    config = prepared.config
    output_channels = config.output_channels
    mix = [0.0] * (frame_count * output_channels)

    for commentary in prepared.commentary:
        loaded = commentary.loaded
        source_channels = loaded.channels
        source_frames = loaded.frames
        source_pcm = loaded.pcm
        trim = commentary.trim_mask
        time_map = commentary.time_map
        active_start = max(start_frame, commentary.start_frame)
        active_end = min(start_frame + frame_count, commentary.end_frame)
        for absolute_frame in range(active_start, active_end):
            position = time_map.source_frame(float(absolute_frame))
            if position < 0.0 or position >= source_frames:
                continue
            lower = int(math.floor(position))
            if trim[lower]:
                continue
            fraction = position - lower
            if fraction == 0.0:
                values = tuple(
                    float(source_pcm[lower * source_channels + channel]) * PCM_SCALE
                    for channel in range(source_channels)
                )
            else:
                next_valid = lower + 1 < source_frames
                values_list: list[float] = []
                for channel in range(source_channels):
                    first = (
                        float(source_pcm[lower * source_channels + channel])
                        * PCM_SCALE
                    )
                    second = (
                        float(source_pcm[(lower + 1) * source_channels + channel])
                        * PCM_SCALE
                        if next_valid
                        else 0.0
                    )
                    values_list.append(first + (second - first) * fraction)
                values = tuple(values_list)
            _add_track_sample(
                mix,
                absolute_frame - start_frame,
                output_channels,
                values,
                commentary.pan_matrix,
                commentary.gain_lin,
            )

    for simple in (prepared.engine, prepared.ambience):
        loaded = simple.loaded
        source_channels = loaded.channels
        source_pcm = loaded.pcm
        source_frames = loaded.frames
        active_start = max(start_frame, simple.start_frame)
        active_end = min(start_frame + frame_count, simple.end_frame)
        for absolute_frame in range(active_start, active_end):
            source_frame = absolute_frame - simple.start_frame
            if source_frame < 0 or source_frame >= source_frames:
                continue
            offset = source_frame * source_channels
            values = tuple(
                float(source_pcm[offset + channel]) * PCM_SCALE
                for channel in range(source_channels)
            )
            _add_track_sample(
                mix,
                absolute_frame - start_frame,
                output_channels,
                values,
                simple.pan_matrix,
                simple.gain_lin,
            )

    ceiling = db_to_lin(config.mix.peak_limit_dbfs)
    knee = ceiling * config.mix.soft_knee_ratio
    limited = [
        _soft_limit(value, ceiling, knee)
        for value in mix
    ]
    return array("f", limited)


def quantize_with_gain(block: array, gain_lin: float) -> array:
    quantized = array("h")
    append = quantized.append
    for value in block:
        sample = int(round(float(value) * gain_lin * 32767.0))
        if sample > 32767:
            sample = 32767
        elif sample < -32768:
            sample = -32768
        append(sample)
    return quantized


def float_array_to_le_bytes(values: array) -> bytes:
    copied = array("f", values)
    if sys.byteorder == "big":
        copied.byteswap()
    return copied.tobytes()


def float_array_from_le_bytes(data: bytes) -> array:
    values = array("f")
    values.frombytes(data)
    if sys.byteorder == "big":
        values.byteswap()
    return values


def int16_array_to_le_bytes(values: array) -> bytes:
    copied = array("h", values)
    if sys.byteorder == "big":
        copied.byteswap()
    return copied.tobytes()


def commentary_effective_frames(
    commentary: PreparedCommentary,
    total_frames: int,
) -> tuple[int, int]:
    source_frames = commentary.loaded.frames
    active_frames = 0
    effective_frames = 0
    start = min(total_frames, commentary.start_frame)
    end = min(total_frames, commentary.end_frame)
    time_map = commentary.time_map
    mask = commentary.trim_mask
    for absolute_frame in range(start, end):
        active_frames += 1
        position = time_map.source_frame(float(absolute_frame))
        if position < 0.0 or position >= source_frames:
            continue
        if mask[int(math.floor(position))]:
            continue
        effective_frames += 1
    return active_frames, effective_frames
