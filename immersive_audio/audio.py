from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import soundfile as sf

from .constants import DEFAULT_CHUNK_SAMPLES
from .errors import (
    AUDIO_DECODE,
    AUDIO_FORMAT,
    AUDIO_IO,
    AUDIO_NOT_FOUND,
    PipelineError,
)
from .scene import AudioClip, AudioObject, Scene
from .timeline import seconds_to_frames


@dataclass(frozen=True)
class ClipAnalysis:
    clip: AudioClip
    source_frames: int
    source_sample_rate: int
    source_channels: int
    target_frames: int
    sha256: str


@dataclass(frozen=True)
class ObjectAudioAnalysis:
    object_index: int
    object_id: str
    clips: tuple[ClipAnalysis, ...]
    bus_frames: int


@dataclass(frozen=True)
class SceneAudioAnalysis:
    total_frames: int
    objects: tuple[ObjectAudioAnalysis, ...]


@dataclass(frozen=True)
class ObjectBusInfo:
    object_index: int
    object_id: str
    path: Path
    total_frames: int
    valid_start_frame: int | None
    valid_end_frame: int | None


def _target_frame_count(source_frames: int, source_rate: int, target_rate: int) -> int:
    if source_frames < 0 or source_rate <= 0 or target_rate <= 0:
        return 0
    return (2 * source_frames * target_rate + source_rate) // (2 * source_rate)


def _probe_clip(clip: AudioClip, target_rate: int) -> ClipAnalysis:
    if not clip.path.is_file():
        raise PipelineError(AUDIO_NOT_FOUND, f"audio clip does not exist: {clip.display_path}")
    try:
        info = sf.info(os.fspath(clip.path))
    except Exception as exc:
        raise PipelineError(
            AUDIO_DECODE, f"cannot decode audio clip: {clip.display_path}"
        ) from exc
    if info.samplerate <= 0 or info.frames < 0:
        raise PipelineError(AUDIO_FORMAT, f"invalid audio metadata: {clip.display_path}")
    return ClipAnalysis(
        clip=clip,
        source_frames=int(info.frames),
        source_sample_rate=int(info.samplerate),
        source_channels=int(info.channels),
        target_frames=_target_frame_count(
            int(info.frames), int(info.samplerate), target_rate
        ),
        sha256="",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(DEFAULT_CHUNK_SAMPLES)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise PipelineError(AUDIO_IO, f"cannot hash audio clip: {path.name}") from exc
    return digest.hexdigest()


def analyze_scene_audio(scene: Scene) -> SceneAudioAnalysis:
    hashes: dict[Path, str] = {}
    objects: list[ObjectAudioAnalysis] = []
    maximum_end = 0
    for obj in scene.objects:
        clips: list[ClipAnalysis] = []
        for clip in obj.clips:
            analysis = _probe_clip(clip, scene.sample_rate)
            digest = hashes.get(clip.path)
            if digest is None:
                digest = sha256_file(clip.path)
                hashes[clip.path] = digest
            clips.append(
                ClipAnalysis(
                    clip=analysis.clip,
                    source_frames=analysis.source_frames,
                    source_sample_rate=analysis.source_sample_rate,
                    source_channels=analysis.source_channels,
                    target_frames=analysis.target_frames,
                    sha256=digest,
                )
            )
            clip_end = (
                seconds_to_frames(clip.start, scene.sample_rate)
                + clip.loops * analysis.target_frames
            )
            maximum_end = max(maximum_end, clip_end)
        objects.append(
            ObjectAudioAnalysis(
                object_index=obj.index,
                object_id=obj.object_id,
                clips=tuple(clips),
                bus_frames=0,
            )
        )

    if scene.duration is None:
        total_frames = maximum_end
    else:
        total_frames = seconds_to_frames(scene.duration, scene.sample_rate)
    return SceneAudioAnalysis(
        total_frames=total_frames,
        objects=tuple(
            ObjectAudioAnalysis(
                object_index=item.object_index,
                object_id=item.object_id,
                clips=item.clips,
                bus_frames=total_frames,
            )
            for item in objects
        ),
    )


def _mono_source_chunk(
    handle: sf.SoundFile,
    source_frames: int,
    source_rate: int,
    target_rate: int,
    out_start: int,
    out_count: int,
) -> np.ndarray:
    positions = (
        np.arange(out_start, out_start + out_count, dtype=np.float64)
        * source_rate
        / target_rate
    )
    if source_frames <= 0:
        return np.zeros(out_count, dtype=np.float32)
    positions = np.minimum(positions, source_frames - 1)
    indices = np.floor(positions).astype(np.int64)
    fractions = (positions - indices).astype(np.float32)
    first_index = int(indices[0])
    last_index = int(indices[-1]) + 1
    last_index = min(last_index, source_frames - 1)
    read_count = last_index - first_index + 1
    if read_count <= 0:
        return np.zeros(out_count, dtype=np.float32)
    handle.seek(first_index)
    try:
        data = handle.read(
            read_count, dtype="float32", always_2d=True
        )
    except Exception as exc:
        raise PipelineError(AUDIO_DECODE, "audio read failed during decoding") from exc
    if data.shape[0] < read_count:
        raise PipelineError(AUDIO_DECODE, "audio clip ended during decoding")
    mono = np.mean(data, axis=1, dtype=np.float32)
    local_indices = indices - first_index
    left = mono[local_indices]
    right_indices = np.minimum(local_indices + 1, read_count - 1)
    right = mono[right_indices]
    return (left + (right - left) * fractions).astype(np.float32, copy=False)


def _mix_analysis_into_bus(
    bus_path: Path,
    object_analysis: ObjectAudioAnalysis,
    total_frames: int,
    sample_rate: int,
    chunk_samples: int = DEFAULT_CHUNK_SAMPLES,
) -> tuple[int | None, int | None]:
    valid_start: int | None = None
    valid_end: int | None = None
    try:
        bus_handle = bus_path.open("w+b")
    except OSError as exc:
        raise PipelineError(AUDIO_IO, "cannot create object bus temporary file") from exc
    try:
        with bus_handle:
            bus_handle.truncate(total_frames * np.dtype("<f4").itemsize)
            for analysis in object_analysis.clips:
                clip = analysis.clip
                target_frames = analysis.target_frames
                if target_frames <= 0:
                    continue
                start_frame = seconds_to_frames(clip.start, sample_rate)
                if start_frame >= total_frames:
                    continue
                try:
                    source_handle = sf.SoundFile(
                        os.fspath(clip.path), mode="r"
                    )
                except Exception as exc:
                    raise PipelineError(
                        AUDIO_DECODE, f"cannot open audio clip: {clip.display_path}"
                    ) from exc
                with source_handle:
                    if source_handle.channels <= 0:
                        raise PipelineError(
                            AUDIO_FORMAT,
                            f"audio clip has no channels: {clip.display_path}",
                        )
                    for out_start in range(0, target_frames, chunk_samples):
                        out_count = min(chunk_samples, target_frames - out_start)
                        signal = _mono_source_chunk(
                            source_handle,
                            analysis.source_frames,
                            analysis.source_sample_rate,
                            sample_rate,
                            out_start,
                            out_count,
                        )
                        for loop_index in range(clip.loops):
                            destination = (
                                start_frame
                                + loop_index * target_frames
                                + out_start
                            )
                            if destination >= total_frames:
                                break
                            copy_count = min(out_count, total_frames - destination)
                            if copy_count <= 0:
                                continue
                            bus_handle.seek(destination * 4)
                            raw = bus_handle.read(copy_count * 4)
                            if len(raw) != copy_count * 4:
                                raise PipelineError(
                                    AUDIO_IO,
                                    "object bus temporary file is truncated",
                                )
                            existing = np.frombuffer(raw, dtype="<f4")
                            mixed = existing + signal[:copy_count]
                            bus_handle.seek(destination * 4)
                            bus_handle.write(
                                mixed.astype("<f4", copy=False).tobytes()
                            )
                            nonzero = bool(np.any(mixed))
                            if nonzero:
                                segment_end = destination + copy_count
                                if valid_start is None:
                                    valid_start = destination
                                valid_end = segment_end
            bus_handle.flush()
    except PipelineError:
        raise
    except OSError as exc:
        raise PipelineError(AUDIO_IO, "object bus write failed") from exc
    return valid_start, valid_end


def build_object_bus(
    scene: Scene,
    obj: AudioObject,
    analysis: SceneAudioAnalysis,
    output_path: Path,
    chunk_samples: int = DEFAULT_CHUNK_SAMPLES,
) -> ObjectBusInfo:
    object_analysis = analysis.objects[obj.index]
    valid_start, valid_end = _mix_analysis_into_bus(
        output_path,
        object_analysis,
        analysis.total_frames,
        scene.sample_rate,
        chunk_samples=chunk_samples,
    )
    return ObjectBusInfo(
        object_index=obj.index,
        object_id=obj.object_id,
        path=output_path,
        total_frames=analysis.total_frames,
        valid_start_frame=valid_start,
        valid_end_frame=valid_end,
    )


def build_scene_object_buses(
    scene: Scene,
    analysis: SceneAudioAnalysis,
    temp_dir: str | Path,
) -> tuple[ObjectBusInfo, ...]:
    directory = Path(temp_dir)
    results: list[ObjectBusInfo] = []
    for obj in scene.objects:
        bus_path = directory / f"object-{obj.index:02d}.f32"
        results.append(build_object_bus(scene, obj, analysis, bus_path))
    return tuple(results)


def read_object_bus(info: ObjectBusInfo) -> np.ndarray:
    try:
        raw = info.path.read_bytes()
    except OSError as exc:
        raise PipelineError(AUDIO_IO, "cannot read object bus temporary file") from exc
    expected = info.total_frames * 4
    if len(raw) != expected:
        raise PipelineError(AUDIO_IO, "object bus temporary file has an invalid size")
    return np.frombuffer(raw, dtype="<f4").copy()


def effective_range(audio: np.ndarray, chunk_samples: int = DEFAULT_CHUNK_SAMPLES):
    first: int | None = None
    last: int | None = None
    for start in range(0, audio.shape[0], chunk_samples):
        block = audio[start : start + chunk_samples]
        indices = np.flatnonzero(block)
        if indices.size:
            if first is None:
                first = start + int(indices[0])
            last = start + int(indices[-1]) + 1
    return first, last


def mix_object_bus(scene: Scene, obj: AudioObject, total_frames: int) -> np.ndarray:
    analysis = analyze_scene_audio(scene)
    with tempfile.TemporaryDirectory(prefix="iapano-bus-") as temporary:
        path = Path(temporary) / "bus.f32"
        info = build_object_bus(scene, obj, analysis, path)
        return read_object_bus(info)[: total_frames if total_frames >= 0 else None]


def mix_scene_to_buses(scene: Scene) -> tuple[dict[str, np.ndarray], dict[str, tuple[int | None, int | None]]]:
    analysis = analyze_scene_audio(scene)
    audio: dict[str, np.ndarray] = {}
    ranges: dict[str, tuple[int | None, int | None]] = {}
    with tempfile.TemporaryDirectory(prefix="iapano-buses-") as temporary:
        buses = build_scene_object_buses(scene, analysis, temporary)
        for obj, info in zip(scene.objects, buses):
            bus = read_object_bus(info)
            audio[obj.object_id] = bus
            ranges[obj.object_id] = effective_range(bus)
    return audio, ranges
