from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .config import BuildConfig, VariantConfig, load_config
from .errors import ConfigError, ConsistencyError, OverwriteError
from .hls import (
    MasterEntry,
    build_master_playlist,
    build_media_playlist,
    parse_master_playlist,
    parse_media_playlist,
)
from .media import (
    AudioProbe,
    count_adts_frames,
    decode_to_pcm,
    encode_aac_segment,
    find_binary,
    probe_audio,
    probe_encoded_segment,
    strip_aac_priming,
)
from .timeline import (
    TimelineSegment,
    align_pcm,
    plan_timeline,
    relative_shift_frames,
)


@dataclass(frozen=True)
class EncodedFile:
    path: Path
    uri: str
    size_bytes: int
    sha256: str
    frame_count: int
    duration: Decimal


@dataclass
class VariantBuild:
    config: VariantConfig
    probe: AudioProbe
    source_duration: Decimal
    shift_frames: int
    files: List[EncodedFile]


@dataclass
class BuildResult:
    output_dir: Path
    sample_rate: int
    segment_count: int
    variants: List[VariantBuild]
    report_path: Path


def _output_targets(config: BuildConfig) -> List[Path]:
    targets = [
        config.output_dir / "master.m3u8",
        config.output_dir / "build_report.json",
    ]
    targets.extend(
        config.output_dir / variant.media_dir for variant in config.variants
    )
    return targets


def _check_overwrite_conflicts(config: BuildConfig) -> None:
    if config.overwrite:
        return
    existing = [path for path in _output_targets(config) if path.exists()]
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise OverwriteError(
            "output already exists; set overwrite to true to replace it: "
            f"{paths}"
        )


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _clean_generated_outputs(config: BuildConfig) -> None:
    for path in _output_targets(config):
        _remove_path(path)
    if config.output_dir.is_dir():
        for child in config.output_dir.iterdir():
            if child.is_dir() and re.fullmatch(r"[0-9]+k", child.name):
                shutil.rmtree(child)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seconds(samples: int, sample_rate: int) -> Decimal:
    return Decimal(samples) / Decimal(sample_rate)


def _ffmpeg_time(samples: int, sample_rate: int) -> str:
    return f"{_seconds(samples, sample_rate):.12f}"


def _json_number(value: Decimal, places: str = "0.000000001") -> float:
    return float(value.quantize(Decimal(places)))


def _validate_inputs(config: BuildConfig, ffprobe: str) -> List[AudioProbe]:
    probes: List[AudioProbe] = []
    for variant in config.variants:
        probe = probe_audio(ffprobe, variant.input_path)
        probes.append(probe)
    reference_rate = probes[0].sample_rate
    mismatched = [
        f"{variant.bitrate_kbps}k={probe.sample_rate}Hz"
        for variant, probe in zip(config.variants, probes)
        if probe.sample_rate != reference_rate
    ]
    if mismatched:
        raise ConfigError(
            "input WAV sample rates must match; "
            f"reference={reference_rate}Hz, found {', '.join(mismatched)}"
        )
    return probes


def _build_variant(
    ffmpeg: str,
    config: BuildConfig,
    variant: VariantConfig,
    probe: AudioProbe,
    reference_probe: AudioProbe,
    segments: Sequence[TimelineSegment],
    work_dir: Path,
    staging_dir: Path,
) -> VariantBuild:
    sample_rate = reference_probe.sample_rate
    channels = probe.channels
    frame_bytes = channels * 2
    decoded_pcm = work_dir / f"{variant.bitrate_kbps}.decoded.pcm"
    aligned_pcm = work_dir / f"{variant.bitrate_kbps}.aligned.pcm"
    decode_to_pcm(
        ffmpeg,
        variant.input_path,
        decoded_pcm,
        sample_rate,
        channels,
    )

    total_frames = segments[-1].end_sample
    shift_frames = relative_shift_frames(
        variant_offset_ms=variant.offset_ms,
        variant_start_time=probe.start_time,
        reference_offset_ms=config.variants[0].offset_ms,
        reference_start_time=reference_probe.start_time,
        sample_rate=sample_rate,
    )
    align_pcm(
        decoded_pcm,
        aligned_pcm,
        target_frames=total_frames,
        frame_bytes=frame_bytes,
        shift_frames=shift_frames,
    )

    variant_dir = staging_dir / variant.media_dir
    variant_dir.mkdir(parents=True, exist_ok=False)
    files: List[EncodedFile] = []
    for segment in segments:
        uri = f"{variant.media_dir}/segment_{segment.index:05d}.aac"
        output_path = staging_dir / uri
        partial_path = output_path.with_suffix(".aac.part")
        encode_aac_segment(
            ffmpeg=ffmpeg,
            aligned_pcm=aligned_pcm,
            destination=partial_path,
            sample_rate=sample_rate,
            channels=channels,
            bitrate_kbps=variant.bitrate_kbps,
            start_seconds=_ffmpeg_time(segment.start_sample, sample_rate),
            duration_seconds=_ffmpeg_time(segment.sample_count, sample_rate),
        )
        frame_count = strip_aac_priming(partial_path, output_path)
        partial_path.unlink()
        files.append(
            EncodedFile(
                path=output_path,
                uri=uri,
                size_bytes=output_path.stat().st_size,
                sha256=_sha256(output_path),
                frame_count=frame_count,
                duration=_seconds(frame_count * 1024, sample_rate),
            )
        )

    playlist = build_media_playlist(segments, sample_rate)
    (variant_dir / "index.m3u8").write_text(playlist, encoding="utf-8")

    source_frames = decoded_pcm.stat().st_size // frame_bytes
    source_duration = _seconds(source_frames, sample_rate)
    if probe.duration is not None:
        tolerance = max(Decimal("0.001"), Decimal(1) / Decimal(sample_rate))
        if abs(probe.duration - source_duration) > tolerance:
            raise ConfigError(
                f"ffprobe duration for {variant.input_path} does not match "
                f"decoded PCM: {probe.duration}s vs {source_duration}s"
            )
    return VariantBuild(
        config=variant,
        probe=probe,
        source_duration=source_duration,
        shift_frames=shift_frames,
        files=files,
    )


def _playlist_boundaries(
    playlist_path: Path,
    expected_count: int,
    bitrate_kbps: int,
) -> Tuple[Decimal, ...]:
    info = parse_media_playlist(playlist_path)
    if len(info.durations) != expected_count:
        raise ConsistencyError(
            f"{bitrate_kbps}k playlist segment count {len(info.durations)} "
            f"does not match expected {expected_count}; first differing "
            f"segment index is {min(len(info.durations), expected_count)}"
        )
    running = Decimal("0")
    boundaries = [running]
    for index, duration in enumerate(info.durations):
        if duration <= 0:
            raise ConsistencyError(
                f"{bitrate_kbps}k segment {index} has non-positive EXTINF"
            )
        running += duration
        boundaries.append(running)
    return tuple(boundaries)


def _validate_uri(uri: str, bitrate_kbps: int, index: int) -> None:
    if uri.startswith("/") or "://" in uri or ".." in Path(uri).parts:
        raise ConsistencyError(
            f"{bitrate_kbps}k segment {index} has invalid relative URI: {uri}"
        )


def _verify_build(
    config: BuildConfig,
    sample_rate: int,
    segments: Sequence[TimelineSegment],
    variants: Sequence[VariantBuild],
    staging_dir: Path,
    ffprobe: str,
) -> None:
    if not variants:
        raise ConsistencyError("no variants were built")
    expected_count = len(segments)
    max_frame_duration = Decimal(1024) / Decimal(sample_rate)
    all_boundaries: List[Tuple[Decimal, ...]] = []

    for variant in variants:
        playlist_path = staging_dir / variant.config.media_dir / "index.m3u8"
        boundaries = _playlist_boundaries(
            playlist_path, expected_count, variant.config.bitrate_kbps
        )
        all_boundaries.append(boundaries)
        info = parse_media_playlist(playlist_path)
        if len(variant.files) != expected_count:
            raise ConsistencyError(
                f"{variant.config.bitrate_kbps}k encoded segment count "
                f"{len(variant.files)} does not match expected {expected_count}"
            )
        for index, (segment, encoded, uri) in enumerate(
            zip(segments, variant.files, info.uris)
        ):
            _validate_uri(uri, variant.config.bitrate_kbps, index)
            expected_output = playlist_path.parent / uri
            if expected_output.resolve() != encoded.path.resolve():
                raise ConsistencyError(
                    f"{variant.config.bitrate_kbps}k segment {index} URI "
                    f"does not match output file: {uri}"
                )
            if not expected_output.is_file():
                raise ConsistencyError(
                    f"{variant.config.bitrate_kbps}k segment {index} is missing: {uri}"
                )

            parsed_frames = count_adts_frames(encoded.path)
            if parsed_frames != encoded.frame_count:
                raise ConsistencyError(
                    f"{variant.config.bitrate_kbps}k segment {index} ADTS "
                    f"frame count changed: {parsed_frames}"
                )
            encoded_probe = probe_encoded_segment(ffprobe, encoded.path)
            if encoded_probe.packet_count != encoded.frame_count:
                raise ConsistencyError(
                    f"{variant.config.bitrate_kbps}k segment {index} ffprobe "
                    f"frame count {encoded_probe.packet_count} differs from "
                    f"ADTS count {encoded.frame_count}"
                )
            if encoded_probe.sample_rate != sample_rate:
                raise ConsistencyError(
                    f"{variant.config.bitrate_kbps}k segment {index} sample "
                    f"rate {encoded_probe.sample_rate} differs from {sample_rate}"
                )

            planned_duration = _seconds(segment.sample_count, sample_rate)
            if abs(encoded.duration - planned_duration) > max_frame_duration:
                raise ConsistencyError(
                    f"{variant.config.bitrate_kbps}k segment {index} duration "
                    f"{encoded.duration}s exceeds one AAC frame tolerance from "
                    f"planned {planned_duration}s"
                )

    reference_boundaries = all_boundaries[0]
    for variant, boundaries in zip(variants[1:], all_boundaries[1:]):
        if boundaries != reference_boundaries:
            for index in range(expected_count + 1):
                if boundaries[index] != reference_boundaries[index]:
                    raise ConsistencyError(
                        f"{variant.config.bitrate_kbps}k segment {index} "
                        f"playlist boundary {boundaries[index]} differs from "
                        f"reference {reference_boundaries[index]}"
                    )

    for index, segment in enumerate(segments):
        frame_counts = {variant.files[index].frame_count for variant in variants}
        if len(frame_counts) != 1:
            values = ", ".join(
                f"{variant.config.bitrate_kbps}k={variant.files[index].frame_count}"
                for variant in variants
            )
            raise ConsistencyError(
                f"segment {index} AAC frame counts differ across variants: {values}"
            )
        expected_frames = int(
            (Decimal(segment.sample_count) / Decimal(1024)).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
        if next(iter(frame_counts)) != expected_frames:
            variant = next(
                item
                for item in variants
                if item.files[index].frame_count != expected_frames
            )
            raise ConsistencyError(
                f"{variant.config.bitrate_kbps}k segment {index} has "
                f"{variant.files[index].frame_count} AAC frames, expected "
                f"{expected_frames}"
            )

    master_path = staging_dir / "master.m3u8"
    master_entries = parse_master_playlist(master_path)
    if len(master_entries) != len(variants):
        raise ConsistencyError(
            f"master playlist has {len(master_entries)} renditions, expected "
            f"{len(variants)}"
        )
    expected_uris = [f"{variant.config.media_dir}/index.m3u8" for variant in variants]
    if [entry.uri for entry in master_entries] != expected_uris:
        raise ConsistencyError("master playlist rendition URIs do not match outputs")
    for entry in master_entries:
        if entry.bandwidth < entry.average_bandwidth:
            raise ConsistencyError(
                f"master playlist BANDWIDTH is below AVERAGE-BANDWIDTH for {entry.uri}"
            )


def _variant_report(
    config: BuildConfig,
    variant: VariantBuild,
    sample_rate: int,
    total_duration: Decimal,
) -> Dict[str, object]:
    total_bytes = sum(item.size_bytes for item in variant.files)
    encoded_duration = sum(item.duration for item in variant.files)
    average_bps = (
        Decimal(total_bytes * 8) / encoded_duration
        if encoded_duration > 0
        else Decimal(0)
    )
    source_frames = int(
        (variant.source_duration * Decimal(sample_rate)).to_integral_value()
    )
    target_frames = int(
        (total_duration * Decimal(sample_rate)).to_integral_value()
    )
    if variant.shift_frames >= 0:
        lead_silence = 0
        effective_samples = min(
            max(source_frames - variant.shift_frames, 0), target_frames
        )
    else:
        lead_silence = min(-variant.shift_frames, target_frames)
        effective_samples = min(
            source_frames, target_frames - lead_silence
        )
    tail_silence = target_frames - lead_silence - effective_samples
    try:
        input_name = variant.config.input_path.relative_to(
            config.config_path.parent
        ).as_posix()
    except ValueError:
        input_name = str(variant.config.input_path)
    return {
        "bitrate_kbps": variant.config.bitrate_kbps,
        "input_file": input_name,
        "media_playlist": f"{variant.config.media_dir}/index.m3u8",
        "segment_count": len(variant.files),
        "total_duration_seconds": _json_number(total_duration),
        "source_duration_seconds": _json_number(variant.source_duration),
        "source_probe_duration_seconds": (
            None
            if variant.probe.duration is None
            else _json_number(variant.probe.duration)
        ),
        "configured_offset_ms": _json_number(variant.config.offset_ms),
        "alignment_offset_ms": _json_number(
            Decimal(variant.shift_frames * 1000) / Decimal(sample_rate)
        ),
        "trimmed_samples": max(0, variant.shift_frames),
        "padded_samples": max(0, -variant.shift_frames),
        "lead_silence_samples": lead_silence,
        "tail_silence_samples": tail_silence,
        "total_silence_padding_samples": lead_silence + tail_silence,
        "effective_audio_samples": effective_samples,
        "source_sample_rate_hz": variant.probe.sample_rate,
        "source_channels": variant.probe.channels,
        "source_start_time_seconds": _json_number(variant.probe.start_time),
        "target_bitrate_bps": variant.config.bitrate_kbps * 1000,
        "encoded_average_bitrate_bps": int(
            average_bps.to_integral_value(rounding=ROUND_CEILING)
        ),
        "total_file_size_bytes": total_bytes,
    }


def _build_report(
    config: BuildConfig,
    sample_rate: int,
    segments: Sequence[TimelineSegment],
    variants: Sequence[VariantBuild],
) -> Dict[str, object]:
    total_duration = _seconds(segments[-1].end_sample, sample_rate)
    segment_reports = []
    for segment in segments:
        files = {}
        for variant in variants:
            encoded = variant.files[segment.index]
            files[f"{variant.config.bitrate_kbps}k"] = {
                "uri": encoded.uri,
                "size_bytes": encoded.size_bytes,
                "sha256": encoded.sha256,
                "aac_frame_count": encoded.frame_count,
                "encoded_duration_seconds": _json_number(encoded.duration),
            }
        segment_reports.append(
            {
                "index": segment.index,
                "start_sample": segment.start_sample,
                "end_sample": segment.end_sample,
                "start_time_seconds": _json_number(
                    _seconds(segment.start_sample, sample_rate)
                ),
                "duration_seconds": _json_number(
                    _seconds(segment.sample_count, sample_rate)
                ),
                "files": files,
            }
        )

    return {
        "schema_version": 1,
        "tool": "hls_audio_builder",
        "sample_rate_hz": sample_rate,
        "segment_duration_seconds": _json_number(config.segment_duration),
        "segment_count": len(segments),
        "total_duration_seconds": _json_number(total_duration),
        "sample_boundaries": [
            segments[0].start_sample,
            *[segment.end_sample for segment in segments],
        ],
        "variants": [
            _variant_report(config, variant, sample_rate, total_duration)
            for variant in variants
        ],
        "segments": segment_reports,
    }


def _master_entries(variants: Sequence[VariantBuild]) -> List[MasterEntry]:
    entries = []
    for variant in variants:
        if not variant.files:
            raise ConsistencyError(
                f"{variant.config.bitrate_kbps}k has no encoded segments"
            )
        average_bps = Decimal(sum(item.size_bytes for item in variant.files) * 8) / sum(
            item.duration for item in variant.files
        )
        peak_bps = max(
            Decimal(item.size_bytes * 8) / item.duration for item in variant.files
        )
        average_bandwidth = max(
            1, int(average_bps.to_integral_value(rounding=ROUND_CEILING))
        )
        bandwidth = max(
            variant.config.bitrate_kbps * 1000,
            int(peak_bps.to_integral_value(rounding=ROUND_CEILING)),
        )
        entries.append(
            MasterEntry(
                bandwidth=bandwidth,
                average_bandwidth=average_bandwidth,
                uri=f"{variant.config.media_dir}/index.m3u8",
            )
        )
    return entries


def build(config_path: Path) -> BuildResult:
    config = load_config(config_path)
    ffmpeg = find_binary("ffmpeg")
    ffprobe = find_binary("ffprobe")
    probes = _validate_inputs(config, ffprobe)
    _check_overwrite_conflicts(config)

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{config.output_dir.name}.build-",
            dir=str(config.output_dir.parent),
        )
    )
    work_dir = Path(tempfile.mkdtemp(prefix="hls-audio-work-"))
    try:
        reference_probe = probes[0]
        with tempfile.TemporaryDirectory(prefix="hls-reference-") as reference_temp:
            reference_pcm = Path(reference_temp) / "reference.pcm"
            decode_to_pcm(
                ffmpeg,
                config.variants[0].input_path,
                reference_pcm,
                reference_probe.sample_rate,
                reference_probe.channels,
            )
            frame_bytes = reference_probe.channels * 2
            size = reference_pcm.stat().st_size
            if size % frame_bytes != 0:
                raise ConfigError("reference input decoded to a partial sample frame")
            total_samples = size // frame_bytes

        segments = plan_timeline(
            total_samples, reference_probe.sample_rate, config.segment_duration
        )
        variants = [
            _build_variant(
                ffmpeg=ffmpeg,
                config=config,
                variant=variant,
                probe=probe,
                reference_probe=reference_probe,
                segments=segments,
                work_dir=work_dir,
                staging_dir=staging_dir,
            )
            for variant, probe in zip(config.variants, probes)
        ]

        master_text = build_master_playlist(_master_entries(variants))
        (staging_dir / "master.m3u8").write_text(master_text, encoding="utf-8")
        report = _build_report(
            config, reference_probe.sample_rate, segments, variants
        )
        (staging_dir / "build_report.json").write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        _verify_build(
            config=config,
            sample_rate=reference_probe.sample_rate,
            segments=segments,
            variants=variants,
            staging_dir=staging_dir,
            ffprobe=ffprobe,
        )

        if config.output_dir.exists() and not config.output_dir.is_dir():
            raise OverwriteError(
                f"output path exists and is not a directory: {config.output_dir}"
            )
        _check_overwrite_conflicts(config)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if config.overwrite:
            _clean_generated_outputs(config)
        for child in sorted(staging_dir.iterdir(), key=lambda item: item.name):
            destination = config.output_dir / child.name
            if destination.exists():
                _remove_path(destination)
            shutil.move(str(child), str(destination))

        return BuildResult(
            output_dir=config.output_dir,
            sample_rate=reference_probe.sample_rate,
            segment_count=len(segments),
            variants=variants,
            report_path=config.output_dir / "build_report.json",
        )
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        shutil.rmtree(work_dir, ignore_errors=True)

