from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any

import numpy as np

from .config import ExportConfig
from .dsp import (
    apply_silence_gate,
    db_to_linear,
    detect_silence,
    fit_channels,
    fit_frames,
    gated_loudness,
    linear_to_db,
    loudness_gain,
    mix_tracks,
    repair_clipping,
    resample_narration,
)
from .errors import (
    BUSINESS_CODES,
    CancelledError,
    EventValidationError,
    PipelineError,
)
from .events import parse_events
from .fingerprint import sha256_json, write_json_atomic
from .resources import CancellationToken
from .state import (
    STATE_SCHEMA_VERSION,
    fsync_parent,
    load_state,
    partial_path_for,
    remove_if_exists,
    state_path_for,
    write_state,
)
from .timeline import Timeline, build_timeline
from .wav import (
    AudioData,
    initialize_partial_wav,
    read_wav,
    sha256_file as wav_sha256_file,
    validate_wav_header,
    write_wav_chunk,
)


PIPELINE_VERSION = "1.0.0"
ALGORITHM_VERSION = "dsp-v1"
MANIFEST_SCHEMA_VERSION = "gamepost-manifest-v1"


@dataclass
class ExportRequest:
    game: str
    mic: str
    events: str
    output: str
    manifest: str | None = None
    state: str | None = None
    config: ExportConfig = field(default_factory=ExportConfig)
    labels: dict[str, str] = field(default_factory=dict)

    def manifest_path(self) -> Path:
        if self.manifest:
            return Path(self.manifest)
        output = Path(self.output)
        return output.with_name(output.name + ".manifest.json")

    def state_path(self) -> Path:
        if self.state:
            return Path(self.state)
        return state_path_for(self.output)


def _chunk_layout(output_frames: int, chunk_frames: int) -> list[tuple[int, int]]:
    return [
        (start, min(output_frames, start + chunk_frames))
        for start in range(0, output_frames, chunk_frames)
    ]


def _chunk_sha_from_file(path: Path, start_frame: int, frames: int, channels: int) -> str:
    offset = 44 + start_frame * channels * 2
    length = frames * channels * 2
    with open(path, "rb") as handle:
        handle.seek(offset)
        payload = handle.read(length)
    return hashlib.sha256(payload).hexdigest()


def _state_payload(
    request: ExportRequest,
    timeline: Timeline,
    input_hashes: dict[str, str],
    input_metadata: dict[str, dict],
    parameter_fingerprint: str,
) -> dict:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "parameter_fingerprint": parameter_fingerprint,
        "timeline_fingerprint": timeline.fingerprint,
        "input_fingerprints": input_hashes,
        "input_metadata": input_metadata,
        "parameters": request.config.to_dict(),
        "timeline": timeline.to_dict(),
        "status": "in_progress",
        "analysis": None,
        "chunks": [],
        "output": None,
        "diagnostics": {
            "partial_path": str(partial_path_for(request.output)),
            "state_path": str(request.state_path()),
        },
    }


def _write_invalid_event_report(
    request: ExportRequest,
    error: EventValidationError,
    input_hashes: dict[str, str] | None = None,
) -> None:
    report = {
        "schema": MANIFEST_SCHEMA_VERSION,
        "status": "invalid_event_table",
        "business_codes": BUSINESS_CODES,
        "exit_code": error.exit_code,
        "primary_error": error.as_dict(),
        "errors": error.details.get("errors", []),
        "inputs": {
            "event_table": {
                "path": str(request.events),
                "sha256": (input_hashes or {}).get("events"),
            },
            "game": {
                "path": str(request.game),
                "sha256": (input_hashes or {}).get("game"),
            },
            "microphone": {
                "path": str(request.mic),
                "sha256": (input_hashes or {}).get("mic"),
            },
        },
        "parameters": request.config.to_dict(),
        "diagnostics": {
            "message": "The event table was rejected before timeline construction.",
        },
    }
    write_json_atomic(request.manifest_path(), report)


def _write_status_report(request: ExportRequest, status: str, error: PipelineError) -> None:
    write_json_atomic(
        request.manifest_path(),
        {
            "schema": MANIFEST_SCHEMA_VERSION,
            "status": status,
            "business_codes": BUSINESS_CODES,
            "exit_code": error.exit_code,
            "error": error.as_dict(),
            "parameters": request.config.to_dict(),
            "diagnostics": {
                "state_path": str(request.state_path()),
                "partial_path": str(partial_path_for(request.output)),
                "resume_rule": (
                    "Completed chunks remain in state; rerunning with the same "
                    "parameter fingerprint resumes at the next incomplete chunk."
                ),
            },
        },
    )


def _parameter_fingerprint(
    input_hashes: dict[str, str],
    timeline: Timeline,
    config: ExportConfig,
) -> str:
    return sha256_json(
        {
            "pipeline_version": PIPELINE_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "inputs": input_hashes,
            "timeline_fingerprint": timeline.fingerprint,
            "parameters": config.to_dict(),
        }
    )


def _verify_records(
    state: dict,
    partial_path: Path,
    timeline: Timeline,
    channels: int,
    chunk_frames: int,
) -> list[dict]:
    records = state.get("chunks")
    if not isinstance(records, list):
        return []
    verified: list[dict] = []
    expected_start = 0
    for expected_index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("index") != expected_index:
            break
        start = record.get("start_frame")
        frames = record.get("frames")
        checksum = record.get("sha256")
        if (
            not isinstance(start, int)
            or not isinstance(frames, int)
            or not isinstance(checksum, str)
            or start != expected_start
            or frames <= 0
            or frames > chunk_frames
            or start + frames > timeline.output_frames
        ):
            break
        actual = _chunk_sha_from_file(partial_path, start, frames, channels)
        if actual != checksum:
            break
        verified.append(record)
        expected_start += frames
    return verified


def _analysis_payload(
    clip_stats: dict,
    silence_segments: list[dict],
    game_loudness: dict,
    mic_loudness: dict,
    game_gain: dict,
    mic_gain: dict,
    mix_stats: dict,
) -> dict:
    return {
        "clipping": clip_stats,
        "silence": {
            "segments": silence_segments,
            "segment_count": len(silence_segments),
            "samples_gated": int(
                sum(segment["samples"] for segment in silence_segments)
            ),
        },
        "loudness": {
            "game": {"measured": game_loudness, "gain": game_gain},
            "microphone": {"measured": mic_loudness, "gain": mic_gain},
        },
        "mix": mix_stats,
    }


def _prepare_processed_audio(
    game: AudioData,
    mic: AudioData,
    timeline: Timeline,
    config: ExportConfig,
    analysis_override: dict | None,
) -> tuple[np.ndarray, np.ndarray, dict, list[dict]]:
    if analysis_override is None:
        clip = repair_clipping(mic.pcm, config)
        clip_stats = clip.stats
    else:
        clip = repair_clipping(mic.pcm, config)
        clip_stats = analysis_override["clipping"]
    mic_float = clip.pcm.astype(np.float64) / 32768.0
    if analysis_override is None:
        silence_segments = detect_silence(mic_float, config.sample_rate, config)
    else:
        silence_segments = analysis_override["silence"]["segments"]
    gated = apply_silence_gate(mic_float, silence_segments)
    narration = resample_narration(gated, timeline, config)
    return game.as_float(), narration, clip_stats, silence_segments


def _mix_chunk(
    game_fit: np.ndarray,
    narration: np.ndarray,
    start: int,
    end: int,
    game_gain_db: float,
    mic_gain_db: float,
    peak_scale: float,
) -> np.ndarray:
    game_gain = db_to_linear(game_gain_db)
    mic_gain = db_to_linear(mic_gain_db)
    raw = (
        game_fit[start:end] * game_gain + narration[start:end] * mic_gain
    )
    return raw * peak_scale


def _build_manifest(
    request: ExportRequest,
    state: dict,
    input_hashes: dict[str, str],
    input_metadata: dict[str, dict],
    timeline: Timeline,
    channels: int,
    final_stats: dict | None,
    resume_decision: str,
) -> dict:
    output = state.get("output") or {}
    analysis = state.get("analysis") or {}
    return {
        "schema": MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "pipeline_version": PIPELINE_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "business_codes": BUSINESS_CODES,
        "inputs": {
            "game": {
                **input_metadata.get("game", {}),
                "sha256": input_hashes["game"],
            },
            "microphone": {
                **input_metadata.get("mic", {}),
                "sha256": input_hashes["mic"],
            },
            "event_table": {
                "path": str(request.events),
                "sha256": input_hashes["events"],
            },
        },
        "timeline": timeline.to_dict(),
        "parameter_fingerprint": state["parameter_fingerprint"],
        "timeline_fingerprint": timeline.fingerprint,
        "parameters": request.config.to_dict(),
        "stages": {
            "clipping": analysis.get("clipping", {}),
            "silence_gate": analysis.get("silence", {}),
            "loudness_gating": analysis.get("loudness", {}),
            "mix_and_peak_protection": analysis.get("mix", {}),
            "final_measurement": final_stats or {},
        },
        "output": {
            "path": str(request.output),
            "file_name": Path(request.output).name,
            "sample_rate": request.config.sample_rate,
            "channels": channels,
            "bit_depth": 16,
            "frames": timeline.output_frames,
            "samples": timeline.output_frames * channels,
            "byte_length": output.get("byte_length"),
            "sha256": output.get("sha256"),
            "sample_count_matches_timeline": bool(
                output.get("frames") == timeline.output_frames
            ),
        },
        "chunks": state.get("chunks", []),
        "chunk_completion_table": [
            {
                "index": chunk["index"],
                "start_frame": chunk["start_frame"],
                "frames": chunk["frames"],
                "sha256": chunk["sha256"],
                "completed": True,
            }
            for chunk in state.get("chunks", [])
        ],
        "diagnostics": {
            "resume_decision": resume_decision,
            "state_path": str(request.state_path()),
            "partial_path": str(partial_path_for(request.output)),
            "determinism": (
                "Float64 arrays, fixed tap-order interpolation, fixed PCM "
                "rounding, and canonical chunk checksums."
            ),
        },
    }


def run_export(
    request: ExportRequest,
    token: CancellationToken | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    token = token or CancellationToken()
    request.manifest_path().parent.mkdir(parents=True, exist_ok=True)
    input_hashes: dict[str, str] = {}
    try:
        input_hashes["game"] = wav_sha256_file(request.game)
        input_hashes["mic"] = wav_sha256_file(request.mic)
        input_hashes["events"] = wav_sha256_file(request.events)
        game = read_wav(request.game, request.config.sample_rate)
        mic = read_wav(request.mic, request.config.sample_rate)
    except PipelineError as error:
        _write_status_report(request, "input_error", error)
        raise
    except OSError as exc:
        error = PipelineError(
            "An input file is missing or unreadable.",
            code="INPUT_WAV_INVALID",
            details={"reason": str(exc)},
        )
        _write_status_report(request, "input_error", error)
        raise error from exc

    try:
        events = parse_events(request.events, mic.frames, request.config)
    except EventValidationError as error:
        _write_invalid_event_report(request, error, input_hashes)
        raise
    timeline = build_timeline(mic.frames, events, request.config)
    channels = max(game.channels, mic.channels)
    parameter_fingerprint = _parameter_fingerprint(
        input_hashes, timeline, request.config
    )
    state_path = request.state_path()
    partial_path = partial_path_for(request.output)
    output_path = Path(request.output)
    input_metadata = {"game": game.describe(), "mic": mic.describe()}
    expected_state = _state_payload(
        request,
        timeline,
        input_hashes,
        input_metadata,
        parameter_fingerprint,
    )

    resumed_state = load_state(state_path)
    resume_decision = "fresh"
    state: dict
    if (
        resumed_state
        and resumed_state.get("schema_version") == STATE_SCHEMA_VERSION
        and resumed_state.get("parameter_fingerprint") == parameter_fingerprint
        and resumed_state.get("timeline_fingerprint") == timeline.fingerprint
    ):
        state = resumed_state
        resume_decision = "resume_fingerprint_match"
        if not validate_wav_header(
            partial_path,
            timeline.output_frames,
            request.config.sample_rate,
            channels,
        ):
            if (
                state.get("status") in {"complete", "finalizing"}
                and output_path.exists()
            ):
                partial_path = output_path
            else:
                state = expected_state
                resume_decision = "reset_inconsistent_state"
                remove_if_exists(state_path)
                remove_if_exists(partial_path)
                initialize_partial_wav(
                    partial_path,
                    timeline.output_frames,
                    request.config.sample_rate,
                    channels,
                )
                write_state(state_path, state)
    else:
        if resumed_state:
            resume_decision = "reset_fingerprint_mismatch"
        remove_if_exists(state_path)
        remove_if_exists(partial_path)
        remove_if_exists(Path(request.output))
        remove_if_exists(request.manifest_path())
        initialize_partial_wav(
            partial_path,
            timeline.output_frames,
            request.config.sample_rate,
            channels,
        )
        state = expected_state
        write_state(state_path, state)

    records = _verify_records(
        state,
        partial_path,
        timeline,
        channels,
        request.config.chunk_frames,
    )
    if len(records) != len(state.get("chunks", [])):
        state["chunks"] = records
        state["status"] = "in_progress"
        write_state(state_path, state)
        resume_decision = "resume_after_chunk_checksum_repair"

    layout = _chunk_layout(timeline.output_frames, request.config.chunk_frames)
    all_complete = (
        len(records) == len(layout)
        and state.get("analysis") is not None
    )

    if all_complete:
        if output_path.exists() and not partial_path.exists():
            candidate_path = output_path
        elif partial_path.exists():
            candidate_path = partial_path
        else:
            candidate_path = None
        if candidate_path is None:
            raise PipelineError(
                "All chunks are recorded but no output file exists.",
                code="STATE_INVALID",
                details={"state_path": str(state_path)},
            )
        output_hash = wav_sha256_file(candidate_path)
        output = state.get("output") or {}
        if output.get("sha256") and output.get("sha256") != output_hash:
            # The chunks individually verify, so finalize rather than trust a
            # stale whole-file hash written by an interrupted finalization.
            state["output"] = {
                **output,
                "sha256": output_hash,
                "byte_length": candidate_path.stat().st_size,
                "frames": timeline.output_frames,
            }
        else:
            state["output"] = {
                **output,
                "sha256": output_hash,
                "byte_length": candidate_path.stat().st_size,
                "frames": timeline.output_frames,
            }
        if candidate_path == partial_path and partial_path != output_path:
            os.replace(partial_path, output_path)
            fsync_parent(output_path)
        state["status"] = "complete"
        write_state(state_path, state)
        final_measurement = state.get("analysis", {}).get("final_measurement", {})
        manifest = _build_manifest(
            request,
            state,
            input_hashes,
            input_metadata,
            timeline,
            channels,
            final_measurement,
            resume_decision,
        )
        write_json_atomic(request.manifest_path(), manifest)
        if progress:
            progress(
                {
                    "status": "complete",
                    "completed_chunks": len(layout),
                    "total_chunks": len(layout),
                }
            )
        return manifest

    token.raise_if_cancelled()
    analysis_override = state.get("analysis")
    game_float, narration, clip_stats, silence_segments = _prepare_processed_audio(
        game,
        mic,
        timeline,
        request.config,
        analysis_override,
    )

    if analysis_override is None:
        game_loudness = gated_loudness(
            fit_channels(game_float, channels), request.config.sample_rate
        )
        mic_loudness = gated_loudness(
            fit_channels(narration, channels), request.config.sample_rate
        )
        game_gain = loudness_gain(
            game_loudness["integrated_lufs"],
            request.config.effective_game_target(),
            request.config,
        )
        mic_gain = loudness_gain(
            mic_loudness["integrated_lufs"],
            request.config.effective_mic_target(),
            request.config,
        )
        raw_mix, mix_stats = mix_tracks(
            game_float,
            narration,
            game_gain["gain_db"],
            mic_gain["gain_db"],
            request.config.peak_limit_dbfs,
        )
        analysis = _analysis_payload(
            clip_stats,
            silence_segments,
            game_loudness,
            mic_loudness,
            game_gain,
            mic_gain,
            mix_stats,
        )
        final_loudness = gated_loudness(raw_mix, request.config.sample_rate)
        final_peak = (
            float(np.max(np.abs(raw_mix))) if raw_mix.size else 0.0
        )
        analysis["final_measurement"] = {
            "integrated_lufs": final_loudness["integrated_lufs"],
            "peak_linear": final_peak,
            "peak_dbfs": (
                None if final_peak == 0.0 else round(linear_to_db(final_peak), 9)
            ),
            "loudness": final_loudness,
        }
        state["analysis"] = analysis
        write_state(state_path, state)
    else:
        analysis = analysis_override

    game_gain_db = float(analysis["loudness"]["game"]["gain"]["gain_db"])
    mic_gain_db = float(analysis["loudness"]["microphone"]["gain"]["gain_db"])
    peak_scale = float(analysis["mix"]["peak_protection_scale"])
    game_fit = fit_frames(game_float, timeline.output_frames, channels)
    state["chunks"] = records
    write_state(state_path, state)

    completed = len(records)
    for chunk_index, (start, end) in enumerate(layout):
        token.raise_if_cancelled()
        if chunk_index < completed:
            continue
        chunk = _mix_chunk(
            game_fit,
            narration,
            start,
            end,
            game_gain_db,
            mic_gain_db,
            peak_scale,
        )
        byte_length = (end - start) * channels * 2
        data = write_wav_chunk(
            partial_path,
            start,
            chunk,
            channels=channels,
        )
        if len(data) != byte_length:
            raise PipelineError(
                "A chunk was not written completely.",
                code="STATE_INVALID",
                details={"chunk_index": chunk_index},
            )
        record = {
            "index": chunk_index,
            "start_frame": int(start),
            "frames": int(end - start),
            "byte_length": byte_length,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        state["chunks"].append(record)
        write_state(state_path, state)
        completed += 1
        if progress:
            progress(
                {
                    "status": "in_progress",
                    "completed_chunks": completed,
                    "total_chunks": len(layout),
                    "last_chunk": record,
                }
            )

    output_hash = wav_sha256_file(partial_path)
    state["output"] = {
        "sha256": output_hash,
        "byte_length": partial_path.stat().st_size,
        "frames": timeline.output_frames,
    }
    state["status"] = "finalizing"
    write_state(state_path, state)
    os.replace(partial_path, output_path)
    fsync_parent(output_path)
    state["status"] = "complete"
    write_state(state_path, state)
    manifest = _build_manifest(
        request,
        state,
        input_hashes,
        input_metadata,
        timeline,
        channels,
        analysis.get("final_measurement", {}),
        resume_decision,
    )
    write_json_atomic(request.manifest_path(), manifest)
    if progress:
        progress(
            {
                "status": "complete",
                "completed_chunks": len(layout),
                "total_chunks": len(layout),
            }
        )
    return manifest
