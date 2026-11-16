"""Offline processing engine and report generation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import MAX_DURATION_SECONDS, MIN_RATE, MAX_RATE
from .dynamics import Compressor, LookaheadLimiter
from .errors import MixerError
from .loudness import SignalAnalyzer, safe_db
from .memory import current_working_set_bytes
from .resample import SincResampler


def _round_half_up(value):
    return int(math.floor(float(value) + 0.5))


def _warn(code, message, track=None, **details):
    result = {"code": code, "message": message}
    if track is not None:
        result["track"] = track
    result.update(details)
    return result


@dataclass
class InputInfo:
    path: Path
    sample_rate: int
    channels: int
    frames: int
    subtype: str
    format: str
    file_bytes: int


@dataclass
class Alignment:
    resampled_frames: int
    destination_start: int
    source_start: int
    active_frames: int

    @property
    def source_stop(self):
        return self.source_start + self.active_frames


class _Accumulator:
    def __init__(self, channels):
        self.channels = int(channels)
        self.frames = 0
        self.square_sum = 0.0
        self.peak = 0.0

    def add(self, block, global_start, active_start, active_stop):
        if block.shape[0] == 0:
            return
        local_start = max(global_start, active_start) - global_start
        local_stop = min(global_start + block.shape[0], active_stop) - global_start
        if local_stop <= local_start:
            return
        part = block[local_start:local_stop]
        self.frames += part.shape[0]
        if part.size:
            self.peak = max(self.peak, float(np.max(np.abs(part))))
            self.square_sum += float(np.sum(part.astype(np.float64) ** 2))

    def result(self):
        if self.frames <= 0:
            return {
                "frames": 0,
                "peak_linear": 0.0,
                "peak_dbfs": None,
                "rms_linear": 0.0,
                "rms_dbfs": None,
            }
        rms = math.sqrt(self.square_sum / (self.frames * self.channels))
        return {
            "frames": self.frames,
            "peak_linear": self.peak,
            "peak_dbfs": safe_db(self.peak),
            "rms_linear": rms,
            "rms_dbfs": safe_db(rms),
        }


class _GainReduction:
    def __init__(self):
        self.frames = 0
        self.maximum = 0.0
        self.total = 0.0

    def add(self, reduction, global_start, active_start, active_stop):
        if reduction.size == 0:
            return
        local_start = max(global_start, active_start) - global_start
        local_stop = min(global_start + reduction.size, active_stop) - global_start
        if local_stop <= local_start:
            return
        part = reduction[local_start:local_stop]
        self.frames += part.size
        self.maximum = max(self.maximum, float(np.max(part)))
        self.total += float(np.sum(part))

    def result(self):
        mean = self.total / self.frames if self.frames else 0.0
        return {
            "frames": self.frames,
            "max_db": float(self.maximum),
            "mean_db": float(mean),
        }


def _format_for(frames, channels, bytes_per_sample):
    estimated = int(frames) * int(channels) * int(bytes_per_sample)
    return "RF64" if estimated >= 3_900_000_000 else "WAV"


def _subtype_bytes(subtype):
    return {
        "PCM_16": 2,
        "PCM_24": 3,
        "PCM_32": 4,
        "FLOAT": 4,
        "DOUBLE": 8,
    }[subtype]


def inspect_inputs(tracks):
    infos = []
    for track in tracks:
        path = Path(track.input_path)
        if not path.exists() or not path.is_file():
            raise MixerError(
                "E_INPUT_MISSING",
                "input audio file does not exist",
                {"track": track.track_id, "path": str(path)},
                exit_code=3,
            )
        if path.suffix.lower() != ".wav":
            raise MixerError(
                "E_UNSUPPORTED_FORMAT",
                "only WAV input files are supported",
                {"track": track.track_id, "path": str(path)},
                exit_code=3,
            )
        try:
            info = sf.info(str(path))
        except Exception as exc:
            raise MixerError(
                "E_UNSUPPORTED_FORMAT",
                "unable to decode input as a supported WAV file",
                {"track": track.track_id, "path": str(path), "reason": str(exc)},
                exit_code=3,
            )
        file_format = str(info.format).upper()
        if file_format not in {"WAV", "WAVEX", "RF64"}:
            raise MixerError(
                "E_UNSUPPORTED_FORMAT",
                "input container is not WAV",
                {"track": track.track_id, "format": info.format},
                exit_code=3,
            )
        if not MIN_RATE <= int(info.samplerate) <= MAX_RATE:
            raise MixerError(
                "E_UNSUPPORTED_RATE",
                "input sample rate must be in [8000, 192000]",
                {"track": track.track_id, "sample_rate": int(info.samplerate)},
                exit_code=3,
            )
        if int(info.frames) <= 0:
            raise MixerError(
                "E_EMPTY_INPUT",
                "input audio contains no frames",
                {"track": track.track_id, "path": str(path)},
                exit_code=3,
            )
        if int(info.channels) <= 0 or int(info.channels) > 64:
            raise MixerError(
                "E_UNSUPPORTED_CHANNELS",
                "input channel count is not supported",
                {"track": track.track_id, "channels": int(info.channels)},
                exit_code=3,
            )
        if track.channel_map is not None:
            for index in track.channel_map:
                if index is not None and index >= int(info.channels):
                    raise MixerError(
                        "E_INVALID_TRACK",
                        "channel_map references a channel outside the input file",
                        {
                            "track": track.track_id,
                            "input_channels": int(info.channels),
                            "channel_index": index,
                        },
                        exit_code=3,
                    )
        infos.append(
            InputInfo(
                path=path,
                sample_rate=int(info.samplerate),
                channels=int(info.channels),
                frames=int(info.frames),
                subtype=str(info.subtype),
                format=file_format,
                file_bytes=int(os.path.getsize(path)),
            )
        )
    return infos


def align_tracks(config, infos):
    output_rate = config.output.sample_rate
    duration_frames = None
    if config.output.duration_seconds is not None:
        duration_frames = _round_half_up(config.output.duration_seconds * output_rate)
        if duration_frames <= 0:
            raise MixerError(
                "E_INVALID_OUTPUT",
                "output duration quantizes to zero frames",
                {"duration_seconds": config.output.duration_seconds},
            )
    alignments = []
    warnings = []
    inferred_end = 0
    for track, info in zip(config.tracks, infos):
        resampled_frames = _round_half_up(info.frames * output_rate / info.sample_rate)
        raw_start = _round_half_up(track.offset_ms * output_rate / 1000.0)
        source_start = max(0, -raw_start)
        destination_start = max(0, raw_start)
        if raw_start < 0:
            warnings.append(
                _warn(
                    "W_OFFSET_NEGATIVE_TRIMMED",
                    "negative offset trimmed audio from the track start",
                    track.track_id,
                    requested_offset_ms=track.offset_ms,
                    trimmed_frames=source_start,
                    trimmed_ms=source_start * 1000.0 / output_rate,
                )
            )
        available = max(0, resampled_frames - source_start)
        if duration_frames is None:
            active = available
            inferred_end = max(inferred_end, destination_start + active)
        else:
            if destination_start >= duration_frames:
                active = 0
                warnings.append(
                    _warn(
                        "W_OFFSET_BEYOND_OUTPUT",
                        "track starts at or beyond the output duration",
                        track.track_id,
                        requested_offset_ms=track.offset_ms,
                        output_duration_seconds=duration_frames / output_rate,
                    )
                )
            else:
                active = min(available, duration_frames - destination_start)
                if active < available:
                    warnings.append(
                        _warn(
                            "W_TRACK_END_TRIMMED",
                            "track tail extends beyond the output duration",
                            track.track_id,
                            trimmed_frames=available - active,
                            trimmed_ms=(available - active) * 1000.0 / output_rate,
                        )
                    )
            inferred_end = max(inferred_end, duration_frames)
        if source_start >= resampled_frames and available == 0:
            warnings.append(
                _warn(
                    "W_TRACK_OUTSIDE_OUTPUT",
                    "the whole track falls outside the output timeline",
                    track.track_id,
                    requested_offset_ms=track.offset_ms,
                )
            )
        alignments.append(
            Alignment(
                resampled_frames=resampled_frames,
                destination_start=destination_start,
                source_start=source_start,
                active_frames=active,
            )
        )
    if duration_frames is None:
        duration_frames = inferred_end
        if duration_frames <= 0:
            raise MixerError(
                "E_EMPTY_TIMELINE",
                "no track contributes audio to the output timeline",
            )
    if duration_frames / output_rate > MAX_DURATION_SECONDS + 1e-9:
        raise MixerError(
            "E_DURATION_TOO_LONG",
            "output duration exceeds 3600 seconds",
            {"duration_seconds": duration_frames / output_rate},
        )
    return alignments, int(duration_frames), warnings


def _map_channels(block, input_channels, output_channels, channel_map):
    if channel_map is not None:
        output = np.zeros((block.shape[0], output_channels), dtype=np.float64)
        for target, source in enumerate(channel_map):
            if source is not None:
                output[:, target] = block[:, source]
        return output
    if input_channels == output_channels:
        return block
    if input_channels == 1:
        return np.repeat(block, output_channels, axis=1)
    if output_channels == 1:
        return np.mean(block, axis=1, keepdims=True)
    output = np.zeros((block.shape[0], output_channels), dtype=np.float64)
    copies = min(input_channels, output_channels)
    output[:, :copies] = block[:, :copies]
    if output_channels > input_channels:
        output[:, input_channels:] = block[:, -1:]
    return output


def _process_track(job):
    config = job["config"]
    track = job["track"]
    info = job["info"]
    alignment = job["alignment"]
    temp_path = job["temp_path"]
    apply_limiter = job["apply_limiter"]
    output_rate = config.output.sample_rate
    output_channels = config.output.channels
    active_start = alignment.source_start
    active_stop = alignment.source_stop
    before = _Accumulator(output_channels)
    after = _Accumulator(output_channels)
    compressor_stats = _GainReduction()
    limiter_stats = _GainReduction()
    gain = 10.0 ** (track.gain_db / 20.0)
    compressor = (
        Compressor(output_rate, track.compressor) if track.compressor is not None else None
    )
    limiter = (
        LookaheadLimiter(output_rate, output_channels, track.limiter)
        if apply_limiter and track.limiter is not None
        else None
    )
    resampler = SincResampler(
        info.sample_rate,
        output_rate,
        info.channels,
        alignment.resampled_frames,
    )
    temp_path = Path(temp_path)
    writer = None
    resampled_position = 0

    def consume_output(start, data, reduction):
        if data.shape[0] == 0:
            return
        if reduction is not None and reduction.size:
            limiter_stats.add(reduction, start, active_start, active_stop)
        after.add(data, start, active_start, active_stop)
        write_start = max(start, active_start)
        write_stop = min(start + data.shape[0], active_stop)
        if write_stop > write_start:
            writer.write(data[write_start - start:write_stop - start])

    try:
        writer = sf.SoundFile(
            str(temp_path),
            mode="w",
            samplerate=output_rate,
            channels=output_channels,
            subtype="FLOAT",
            format=_format_for(alignment.active_frames, output_channels, 4),
        )
        with sf.SoundFile(str(info.path), mode="r") as reader:
            while True:
                source = reader.read(
                    job["chunk_frames"],
                    dtype="float64",
                    always_2d=True,
                )
                if source.shape[0] == 0:
                    break
                resampled = resampler.feed(source)
                if resampled.shape[0] == 0:
                    continue
                mapped = _map_channels(
                    resampled, info.channels, output_channels, track.channel_map
                )
                start = resampled_position
                before.add(mapped, start, active_start, active_stop)
                processed = mapped * gain
                if compressor is not None:
                    processed, reduction = compressor.process(processed)
                    compressor_stats.add(reduction, start, active_start, active_stop)
                if limiter is not None:
                    limit_start, limited, reduction = limiter.feed(processed)
                    consume_output(limit_start, limited, reduction)
                else:
                    consume_output(start, processed, None)
                resampled_position += resampled.shape[0]

            tail = resampler.finish()
            if tail.shape[0]:
                mapped = _map_channels(tail, info.channels, output_channels, track.channel_map)
                start = resampled_position
                before.add(mapped, start, active_start, active_stop)
                processed = mapped * gain
                if compressor is not None:
                    processed, reduction = compressor.process(processed)
                    compressor_stats.add(reduction, start, active_start, active_stop)
                if limiter is not None:
                    limit_start, limited, reduction = limiter.feed(processed)
                    consume_output(limit_start, limited, reduction)
                else:
                    consume_output(start, processed, None)
                resampled_position += tail.shape[0]
            if limiter is not None:
                limit_start, limited, reduction = limiter.flush()
                consume_output(limit_start, limited, reduction)
        writer.close()
        writer = None
        if resampled_position != alignment.resampled_frames:
            raise RuntimeError(
                "track {} resampled {} of {} frames".format(
                    track.track_id,
                    resampled_position,
                    alignment.resampled_frames,
                )
            )
        if before.frames != alignment.active_frames:
            raise RuntimeError(
                "track {} measured {} of {} active input frames".format(
                    track.track_id, before.frames, alignment.active_frames
                )
            )
        if after.frames != alignment.active_frames:
            raise RuntimeError(
                "track {} emitted {} of {} active output frames".format(
                    track.track_id, after.frames, alignment.active_frames
                )
            )
    finally:
        if writer is not None:
            writer.close()

    compressor_report = None
    if track.compressor is not None:
        compressor_report = asdict(track.compressor)
        compressor_report["applied"] = True
        compressor_report["gain_reduction"] = compressor_stats.result()
    limiter_report = None
    if track.limiter is not None:
        limiter_report = asdict(track.limiter)
        limiter_report["applied"] = bool(apply_limiter)
        limiter_report["gain_reduction"] = (
            limiter_stats.result() if apply_limiter else {
                "frames": 0,
                "max_db": 0.0,
                "mean_db": 0.0,
            }
        )
    return {
        "id": track.track_id,
        "input": str(track.input_path),
        "original_sample_rate": info.sample_rate,
        "target_sample_rate": output_rate,
        "original_channels": info.channels,
        "output_channels": output_channels,
        "original_frames": info.frames,
        "resampled_frames": alignment.resampled_frames,
        "offset_ms_requested": track.offset_ms,
        "offset_ms_effective": alignment.destination_start * 1000.0 / output_rate,
        "destination_start_sample": alignment.destination_start,
        "source_trim_start_sample": alignment.source_start,
        "active_frames": alignment.active_frames,
        "active_duration_seconds": alignment.active_frames / output_rate,
        "gain_db": track.gain_db,
        "channel_map": list(track.channel_map) if track.channel_map is not None else None,
        "compressor": compressor_report,
        "limiter": limiter_report,
        "stats": {
            "before_processing": before.result(),
            "after_processing": after.result(),
        },
        "temp_path": str(temp_path),
        "mix_start_sample": alignment.destination_start,
        "mix_frames": alignment.active_frames,
    }


def _mix_bus(config, results, output_temp, duration_frames):
    buses = []
    writer = None
    try:
        for result in results:
            if result["mix_frames"] <= 0:
                continue
            path = result["temp_path"]
            handle = sf.SoundFile(path, mode="r")
            if handle.samplerate != config.output.sample_rate:
                handle.close()
                raise RuntimeError("temporary track sample rate mismatch")
            if handle.channels != config.output.channels:
                handle.close()
                raise RuntimeError("temporary track channel count mismatch")
            buses.append(
                {
                    "handle": handle,
                    "start": int(result["mix_start_sample"]),
                    "frames": int(result["mix_frames"]),
                }
            )
        samples_per_frame = _subtype_bytes(config.output.subtype)
        writer = sf.SoundFile(
            str(output_temp),
            mode="w",
            samplerate=config.output.sample_rate,
            channels=config.output.channels,
            subtype=config.output.subtype,
            format=_format_for(duration_frames, config.output.channels, samples_per_frame),
        )
    except Exception:
        for bus in buses:
            bus["handle"].close()
        if writer is not None:
            writer.close()
        raise
    before_meter = SignalAnalyzer(config.output.sample_rate, config.output.channels)
    after_meter = SignalAnalyzer(config.output.sample_rate, config.output.channels)
    bus_limiter = (
        LookaheadLimiter(
            config.output.sample_rate,
            config.output.channels,
            config.output.bus_limiter,
        )
        if config.output.limiter_placement in {"bus", "both"}
        else None
    )
    bus_limiter_stats = _GainReduction()
    block_frames = config.chunk_frames
    position = 0

    def consume_final(start, data, reduction):
        if reduction is not None and reduction.size:
            bus_limiter_stats.add(reduction, start, 0, duration_frames)
        if data.shape[0]:
            writer.write(data)
            after_meter.feed(data)

    try:
        while position < duration_frames:
            count = min(block_frames, duration_frames - position)
            mixed = np.zeros((count, config.output.channels), dtype=np.float64)
            for bus in buses:
                overlap_start = max(position, bus["start"])
                overlap_stop = min(position + count, bus["start"] + bus["frames"])
                if overlap_stop <= overlap_start:
                    continue
                bus["handle"].seek(overlap_start - bus["start"])
                data = bus["handle"].read(
                    overlap_stop - overlap_start,
                    dtype="float64",
                    always_2d=True,
                )
                mixed[
                    overlap_start - position:overlap_stop - position
                ] += data
            before_meter.feed(mixed)
            if bus_limiter is not None:
                start, limited, reduction = bus_limiter.feed(mixed)
                consume_final(start, limited, reduction)
            else:
                consume_final(position, mixed, None)
            position += count
        if bus_limiter is not None:
            start, limited, reduction = bus_limiter.flush()
            consume_final(start, limited, reduction)
    finally:
        if writer is not None:
            writer.close()
        for bus in buses:
            bus["handle"].close()
    if writer.frames != duration_frames:
        raise RuntimeError(
            "output writer received {} of {} frames".format(
                writer.frames, duration_frames
            )
        )
    return {
        "before_mix": before_meter.result(),
        "after_mix": after_meter.result(),
        "bus_limiter": {
            "applied": bus_limiter is not None,
            "parameters": asdict(config.output.bus_limiter),
            "gain_reduction": bus_limiter_stats.result(),
        },
    }


def _sha256(path, chunk_size=1048576):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        return temp_path
    except Exception:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise


def run_mix(config):
    started = datetime.now(timezone.utc)
    baseline_current, baseline_peak = current_working_set_bytes()
    infos = inspect_inputs(config.tracks)
    alignments, duration_frames, warnings = align_tracks(config, infos)
    for track, info, alignment in zip(config.tracks, infos, alignments):
        info_dict = {
            "original_sample_rate": info.sample_rate,
            "target_sample_rate": config.output.sample_rate,
            "resampled_frames": alignment.resampled_frames,
        }
        for warning in warnings:
            if warning.get("track") == track.track_id:
                warning.setdefault("input", info_dict)

    output_path = Path(config.output.path)
    report_path = Path(config.output.report_path)
    if output_path == report_path:
        raise MixerError(
            "E_INVALID_OUTPUT",
            "output and report paths must be different",
        )
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MixerError(
            "E_OUTPUT_WRITE",
            "unable to create output directory",
            {"reason": str(exc)},
            exit_code=5,
        )

    try:
        temp_dir_context = tempfile.TemporaryDirectory(
            prefix=".offline_mixer_",
            dir=str(output_path.parent),
        )
    except OSError as exc:
        raise MixerError(
            "E_OUTPUT_WRITE",
            "unable to create temporary workspace",
            {"reason": str(exc)},
            exit_code=5,
        )

    with temp_dir_context as temp_dir:
        final_temp = Path(temp_dir) / "final_output.wav"
        report_temp = None
        jobs = []
        for index, (track, info, alignment) in enumerate(
            zip(config.tracks, infos, alignments)
        ):
            jobs.append(
                {
                    "config": config,
                    "track": track,
                    "info": info,
                    "alignment": alignment,
                    "temp_path": Path(temp_dir) / "track_{:04d}.wav".format(index),
                    "apply_limiter": config.output.limiter_placement
                    in {"per_track", "both"},
                    "chunk_frames": config.chunk_frames,
                }
            )

        if config.workers == 1:
            results = []
            for job in jobs:
                try:
                    results.append(_process_track(job))
                except MixerError:
                    raise
                except Exception as exc:
                    raise MixerError(
                        "E_PROCESSING",
                        "track processing failed",
                        {"track": job["track"].track_id, "reason": str(exc)},
                        exit_code=4,
                    )
        else:
            results = [None] * len(jobs)
            with ThreadPoolExecutor(max_workers=config.workers) as executor:
                future_map = {
                    executor.submit(_process_track, job): (index, job["track"].track_id)
                    for index, job in enumerate(jobs)
                }
                for future, (index, track_id) in future_map.items():
                    try:
                        results[index] = future.result()
                    except MixerError:
                        for pending in future_map:
                            pending.cancel()
                        raise
                    except Exception as exc:
                        for pending in future_map:
                            pending.cancel()
                        raise MixerError(
                            "E_WORKER",
                            "parallel track processing failed",
                            {"track": track_id, "reason": str(exc)},
                            exit_code=4,
                        )

        try:
            bus_report = _mix_bus(config, results, final_temp, duration_frames)
        except MixerError:
            raise
        except Exception as exc:
            raise MixerError(
                "E_PROCESSING",
                "bus mixing or limiting failed",
                {"reason": str(exc)},
                exit_code=4,
            )

        output_analysis = bus_report["after_mix"]
        ceiling = config.output.bus_limiter.ceiling_dbfs
        true_peak = output_analysis.get("true_peak_dbtp")
        if true_peak is not None and true_peak > ceiling + 1e-6:
            warnings.append(
                _warn(
                    "W_TRUE_PEAK_ABOVE_CEILING",
                    "measured true peak exceeds the configured ceiling",
                    None,
                    true_peak_dbtp=true_peak,
                    ceiling_dbfs=ceiling,
                    limiter_placement=config.output.limiter_placement,
                )
            )

        current, peak = current_working_set_bytes()
        peak = peak or current or 0
        input_total_bytes = sum(info.file_bytes for info in infos)
        decoded_input_bytes = sum(
            info.frames * info.channels * 4 for info in infos
        )
        memory_ratio = (
            float(peak) / float(decoded_input_bytes) if decoded_input_bytes else None
        )
        finished = datetime.now(timezone.utc)
        elapsed = (finished - started).total_seconds()
        output_hash = _sha256(final_temp)
        output_frames = duration_frames
        report = {
            "schema_version": "1.0",
            "status": "ok",
            "created_utc": finished.isoformat(),
            "source_config": str(config.source_path),
            "output": {
                "path": str(output_path),
                "sample_rate": config.output.sample_rate,
                "channels": config.output.channels,
                "subtype": config.output.subtype,
                "duration_seconds": output_frames / config.output.sample_rate,
                "frames": output_frames,
                "sha256": output_hash,
            },
            "processing": {
                "workers": config.workers,
                "parallel_mode": "thread",
                "chunk_frames": config.chunk_frames,
                "limiter_placement": config.output.limiter_placement,
                "deterministic_order": "parallel per-track files, ordered serial bus sum",
                "resampler": {
                    "method": "windowed-sinc",
                    "taps": 64,
                    "window": "Kaiser beta=12",
                    "cutoff": 0.97,
                },
                "elapsed_seconds": elapsed,
            },
            "loudness": {
                "standard": "ITU-R BS.1770-4",
                "before_mix": bus_report["before_mix"],
                "after_mix": output_analysis,
                "integrated_lufs": output_analysis.get("integrated_lufs"),
                "integrated_lufs_before_mix": bus_report["before_mix"].get(
                    "integrated_lufs"
                ),
                "integrated_lufs_after_mix": output_analysis.get("integrated_lufs"),
                "short_term_max_lufs": output_analysis.get("short_term_max_lufs"),
                "true_peak_dbtp": output_analysis.get("true_peak_dbtp"),
            },
            "integrated_lufs": output_analysis.get("integrated_lufs"),
            "short_term_max_lufs": output_analysis.get("short_term_max_lufs"),
            "true_peak_dbtp": output_analysis.get("true_peak_dbtp"),
            "memory_peak_bytes": int(peak),
            "limiter": bus_report["bus_limiter"],
            "tracks": [],
            "track_count": len(results),
            "warnings": warnings,
            "memory": {
                "measurement": "OS process resident working set (peak)",
                "baseline_bytes": baseline_current,
                "baseline_peak_bytes": baseline_peak,
                "current_bytes": current,
                "peak_working_set_bytes": int(peak),
                "input_file_bytes": input_total_bytes,
                "decoded_input_float32_bytes": decoded_input_bytes,
                "peak_to_decoded_input_ratio": (
                    round(memory_ratio, 6) if memory_ratio is not None else None
                ),
                "streaming_temp_files": True,
            },
        }
        for result in results:
            public = dict(result)
            public.pop("temp_path", None)
            report["tracks"].append(public)

        current, peak = current_working_set_bytes()
        peak = peak or current or report["memory"]["peak_working_set_bytes"]
        report["memory"]["current_bytes"] = current
        report["memory"]["peak_working_set_bytes"] = int(peak)
        report["memory"]["peak_to_decoded_input_ratio"] = (
            round(float(peak) / float(decoded_input_bytes), 6)
            if decoded_input_bytes
            else None
        )
        report["memory_peak_bytes"] = int(peak)

        try:
            report_temp = _atomic_json(report_path, report)
        except Exception as exc:
            raise MixerError(
                "E_OUTPUT_WRITE",
                "unable to write report",
                {"path": str(report_path), "reason": str(exc)},
                exit_code=5,
            )
        try:
            try:
                os.replace(str(final_temp), str(output_path))
            except OSError as exc:
                raise MixerError(
                    "E_OUTPUT_WRITE",
                    "unable to publish output WAV",
                    {"path": str(output_path), "reason": str(exc)},
                    exit_code=5,
                )
            try:
                os.replace(str(report_temp), str(report_path))
            except OSError as exc:
                try:
                    os.unlink(str(output_path))
                except OSError:
                    pass
                raise MixerError(
                    "E_OUTPUT_WRITE",
                    "unable to publish JSON report",
                    {"path": str(report_path), "reason": str(exc)},
                    exit_code=5,
                )
        finally:
            if report_temp is not None:
                try:
                    Path(report_temp).unlink()
                except OSError:
                    pass
        report["output"]["path"] = str(output_path)
        return report
