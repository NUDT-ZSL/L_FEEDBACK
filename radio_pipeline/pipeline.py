from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from . import __version__
from .audio import (
    generate_source_pcm,
    loudness_lufs_approx,
    pcm_square_sum,
    sha256_file,
)
from .config import PipelineConfig
from .errors import ProcessingError, VerificationFailure
from .ffmpeg_io import (
    decode_audio_int16,
    embedded_metadata,
    encode_with_ffmpeg,
    ffmpeg_version,
    stream_summary,
)
from .packets import generate_network_events, read_packet_jsonl, write_packet_jsonl
from .plc import render_plc
from .receiver import (
    enrich_events_with_frames,
    receive_packets,
    write_accepted_packets,
    write_jsonl,
)
from .stats import LOG_SCHEMA, build_report, write_report


def _missing_ranges(
    frames: list[Any], total_packets: int, frame_samples: int
) -> list[dict[str, Any]]:
    sequences = [index for index, frame in enumerate(frames) if frame.packet is None]
    ranges: list[dict[str, Any]] = []
    index = 0
    while index < len(sequences):
        start = sequences[index]
        end = start
        while index + 1 < len(sequences) and sequences[index + 1] == end + 1:
            index += 1
            end = sequences[index]
        ranges.append(
            {
                "start_sequence": start,
                "end_sequence": end,
                "count": end - start + 1,
                "start_sample": start * frame_samples,
                "end_sample": (end + 1) * frame_samples,
            }
        )
        index += 1
    if total_packets == 0 and not ranges:
        return []
    return ranges


def _metadata_payload(
    config: PipelineConfig,
    missing_ranges: list[dict[str, Any]],
    missing_count: int,
) -> dict[str, Any]:
    return {
        "sample_rate": config.audio.sample_rate,
        "channels": config.audio.channels,
        "duration_ms": config.audio.duration_ms,
        "total_samples": config.audio.total_samples,
        "frame_ms": config.audio.frame_ms,
        "packet_count": config.audio.total_packets,
        "missing_packet_count": missing_count,
        "missing_packet_ranges": missing_ranges,
        "timeline_policy": "fixed_960_samples_per_packet; missing=PLC_or_mute",
    }


def run_pipeline(
    config: PipelineConfig,
    output_dir: Path,
    input_path: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    source_frames = generate_source_pcm(config.audio, config.seed)
    source_pcm_bytes = source_frames.astype("<i2").tobytes(order="C")
    source_pcm_sha = hashlib.sha256(source_pcm_bytes).hexdigest()
    generated_input_path = logs_dir / "input_packets.jsonl"

    if input_path is None:
        events = generate_network_events(source_frames, config.network, config.seed)
        input_sha = write_packet_jsonl(generated_input_path, events)
        input_file = generated_input_path
        input_mode = "generated"
    else:
        input_file = input_path.resolve()
        try:
            input_sha = sha256_file(input_file)
        except OSError as exc:
            raise ProcessingError(f"cannot read input packet sequence: {exc}") from exc
        input_mode = "provided"
        if input_file != generated_input_path:
            try:
                shutil.copyfile(input_file, generated_input_path)
            except OSError as exc:
                raise ProcessingError(
                    f"cannot copy input packet sequence into run directory: {exc}"
                ) from exc

    parsed = read_packet_jsonl(input_file, config.audio.frame_samples * 2)
    receiver_result = receive_packets(
        parsed,
        config.audio.total_packets,
        config.jitter_buffer,
        config.plc,
        config.audio.frame_ms,
    )
    plc_result = render_plc(
        receiver_result.frames, config.audio, config.plc, config.seed
    )
    enrich_events_with_frames(receiver_result.event_logs, receiver_result.frames)

    expected_samples = config.audio.total_samples
    if plc_result.pcm.size != expected_samples:
        raise ProcessingError(
            f"PLC timeline produced {plc_result.pcm.size} samples, expected {expected_samples}"
        )
    if len(receiver_result.frames) != config.audio.total_packets:
        raise ProcessingError("frame timeline length changed during processing")
    timeline_monotonic = all(
        frame.timestamp_ms == index * config.audio.frame_ms
        for index, frame in enumerate(receiver_result.frames)
    )
    if not timeline_monotonic:
        raise ProcessingError("frame timeline is not monotonically increasing")

    missing_frames = [frame for frame in receiver_result.frames if frame.packet is None]
    missing_ranges = _missing_ranges(
        receiver_result.frames, config.audio.total_packets, config.audio.frame_samples
    )
    metadata = _metadata_payload(config, missing_ranges, len(missing_frames))
    encode_result = encode_with_ffmpeg(
        plc_result.pcm,
        output_dir,
        metadata,
        config.output.wav_name,
        config.output.ogg_name,
        config.output.opus_bitrate_bps,
    )

    decoded_wav = decode_audio_int16(encode_result.wav_path)
    if decoded_wav.size != plc_result.pcm.size:
        raise ProcessingError(
            f"encoded WAV decoded to {decoded_wav.size} samples, expected {plc_result.pcm.size}"
        )
    decoded_ogg = decode_audio_int16(encode_result.ogg_path)
    if abs(decoded_ogg.size - plc_result.pcm.size) > config.output.duration_tolerance_samples:
        raise ProcessingError(
            f"encoded Ogg decoded to {decoded_ogg.size} samples, expected "
            f"{plc_result.pcm.size} +/- {config.output.duration_tolerance_samples}"
        )
    wav_metadata = embedded_metadata(encode_result.wav_probe)
    ogg_metadata = embedded_metadata(encode_result.ogg_probe)
    if wav_metadata != metadata or ogg_metadata != metadata:
        raise ProcessingError("FFmpeg output metadata was not embedded as expected")

    square_sum = pcm_square_sum(plc_result.pcm)
    loudness = loudness_lufs_approx(square_sum, int(plc_result.pcm.size))
    output_info = {
        "sample_count": int(plc_result.pcm.size),
        "duration_ms": config.audio.duration_ms,
        "sum_squares": square_sum,
        "loudness_lufs_approx": loudness,
        "pcm_sha256": sha256_file(encode_result.pcm_path),
        "wav_sha256": sha256_file(encode_result.wav_path),
        "ogg_sha256": sha256_file(encode_result.ogg_path),
        "wav_stream": stream_summary(encode_result.wav_probe),
        "ogg_stream": stream_summary(encode_result.ogg_probe),
        "embedded_metadata": metadata,
    }

    frame_rows: list[dict[str, Any]] = []
    for frame in receiver_result.frames:
        row = frame.to_log()
        row["output_start_sample"] = frame.sequence * config.audio.frame_samples
        row["output_end_sample"] = (frame.sequence + 1) * config.audio.frame_samples
        frame_rows.append(row)

    log: dict[str, Any] = {
        "schema": LOG_SCHEMA,
        "pipeline_version": __version__,
        "config": config.normalized,
        "config_sha256": config.sha256(),
        "input_mode": input_mode,
        "input_sha256": input_sha,
        "input_file": input_file.name,
        "source": {
            "sample_rate": config.audio.sample_rate,
            "channels": config.audio.channels,
            "frame_ms": config.audio.frame_ms,
            "frame_samples": config.audio.frame_samples,
            "duration_ms": config.audio.duration_ms,
            "total_samples": config.audio.total_samples,
            "packets": config.audio.total_packets,
            "source_pcm_sha256": source_pcm_sha,
        },
        "ffmpeg": {
            "version": ffmpeg_version(),
            "commands": encode_result.commands,
            "wav_probe": encode_result.wav_probe,
            "ogg_probe": encode_result.ogg_probe,
        },
        "events": receiver_result.event_logs,
        "frames": frame_rows,
        "diagnostics": receiver_result.diagnostics,
        "jitter_buffer": receiver_result.jitter_log,
        "plc": {
            "pitch_frames": plc_result.pitch_frames,
            "comfort_frames": plc_result.comfort_frames,
            "muted_frames": plc_result.muted_frames,
            "missing_packet_count": len(missing_frames),
            "missing_packet_ranges": missing_ranges,
        },
        "output": output_info,
    }

    write_jsonl(logs_dir / "packet_events.jsonl", receiver_result.event_logs)
    write_jsonl(logs_dir / "frame_log.jsonl", frame_rows)
    write_accepted_packets(
        logs_dir / "accepted_packets.jsonl", receiver_result.accepted_packets
    )
    manifest = {
        "schema": "radio-pipeline-input-manifest-v1",
        "seed": config.seed,
        "input_mode": input_mode,
        "input_file": input_file.name,
        "input_sha256": input_sha,
        "source_pcm_sha256": source_pcm_sha,
        "total_packets": config.audio.total_packets,
        "packet_payload_bytes": config.audio.frame_samples * 2,
        "network": config.normalized["network"],
    }
    (logs_dir / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (logs_dir / "processing_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "audio_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    try:
        report = build_report(log)
    except VerificationFailure as exc:
        raise ProcessingError(f"internal log consistency check failed: {exc}") from exc
    report_path = output_dir / "report.json"
    write_report(report, report_path)
    run_summary = {
        "schema": "radio-pipeline-run-summary-v1",
        "output_dir": str(output_dir),
        "wav": str(encode_result.wav_path),
        "ogg": str(encode_result.ogg_path),
        "pcm": str(encode_result.pcm_path),
        "report": str(report_path),
        "processing_log": str(logs_dir / "processing_log.json"),
        "counts": report["counts"],
        "rates": report["rates"],
        "buffer": report["buffer"],
        "degradation": report["degradation"],
        "output": report["output"],
        "input_sha256": input_sha,
        "config_sha256": config.sha256(),
        "return_code": 0,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_summary
