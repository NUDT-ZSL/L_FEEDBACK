"""End-to-end full and incremental mixing pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import ffmpeg_utils, state
from .audio import decode_track, mix_tracks, total_video_samples
from .errors import FFmpegError, ProcessError
from .ffmpeg_utils import encode_intermediate_video, mux_final
from .hashing import canonical_json_hash, sha256_file
from .project import Project
from .video import process_video


def log_event(event: str, **fields) -> None:
    payload = {"event": event, **fields}
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _load_all_tracks(project: Project, cache_dir: Path) -> list:
    decoded = []
    for track in project.tracks:
        decoded.append(decode_track(track, cache_dir))
    return decoded


def _finish_audio_and_mux(project: Project, intermediate: Path, cache_dir: Path) -> None:
    video_info = ffmpeg_utils.probe_video(project.video_path)
    total_samples = total_video_samples(
        video_info["frame_count"], video_info["frame_rate"]
    )
    decoded = _load_all_tracks(project, cache_dir)
    mix, scale = mix_tracks(project.tracks, decoded, total_samples)
    if scale != 1.0:
        log_event("peak_normalization", scale=scale, peak_target=0.99)
    else:
        log_event("peak_normalization", scale=1.0, applied=False)
    mux_final(intermediate, mix.tobytes(), project.output_path)


def run_full(project: Project, video_info: dict, video_digest: str) -> None:
    output_path = project.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate = state.intermediate_video_path(output_path)
    cache_dir = state.audio_cache_dir(output_path)

    encoder = encode_intermediate_video(
        video_info["width"],
        video_info["height"],
        video_info["frame_rate"],
        video_info["frame_count"],
        intermediate,
    )
    try:
        timestamps = process_video(
            project.video_path,
            video_info["width"],
            video_info["height"],
            video_info["frame_rate"],
            video_info["frame_count"],
            project.markers,
            encoder,
        )
        encoder_stderr = (
            encoder.stderr.read() if encoder.stderr is not None else b""
        )
        return_code = encoder.wait()
    except BaseException:
        encoder.kill()
        encoder.wait()
        _safe_unlink(intermediate)
        raise
    if return_code != 0:
        _safe_unlink(intermediate)
        raise FFmpegError(
            "intermediate video encoder failed: "
            + encoder_stderr.decode("utf-8", "replace").strip()[-500:]
        )
    if not intermediate.is_file() or intermediate.stat().st_size == 0:
        raise ProcessError("intermediate video file was not produced")

    log_event(
        "video_processed",
        frames=len(timestamps),
        first_timestamp_us=timestamps[0] if timestamps else 0,
        last_timestamp_us=timestamps[-1] if timestamps else 0,
        markers=len(project.markers),
    )

    _finish_audio_and_mux(project, intermediate, cache_dir)

    track_digests = [sha256_file(track.path) for track in project.tracks]
    intermediate_digest = sha256_file(intermediate)
    payload = state.build_state(
        project,
        video_info,
        video_digest,
        intermediate,
        intermediate_digest,
        track_digests,
    )
    state.atomic_write_json(state.state_file_path(output_path), payload)
    log_event("state_written", path=str(state.state_file_path(output_path)))


def run_incremental(project: Project, decision, intermediate: Path) -> None:
    cache_dir = state.audio_cache_dir(project.output_path)
    log_event(
        "incremental_reuse",
        intermediate_video=str(intermediate),
        changed_tracks=decision.changed_tracks,
        reason=decision.reason,
    )
    _finish_audio_and_mux(project, intermediate, cache_dir)

    # Refresh the state file so envelope/offset hashes match the new project.
    state_path = state.state_file_path(project.output_path)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    for track, saved in zip(project.tracks, payload["audio_tracks"]):
        saved["offset_ms"] = track.offset_ms
        saved["envelope_hash"] = canonical_json_hash(list(track.envelope))
    state.atomic_write_json(state_path, payload)
    log_event("state_updated", path=str(state_path))


def execute(project: Project) -> None:
    state.cleanup_stale_temps(project.output_path)
    log_event("project_loaded", video=str(project.video_path), tracks=len(project.tracks))

    video_info = ffmpeg_utils.probe_video(project.video_path)
    video_digest = sha256_file(project.video_path)
    log_event(
        "video_probed",
        width=video_info["width"],
        height=video_info["height"],
        frame_rate=str(video_info["frame_rate"]),
        frame_count=video_info["frame_count"],
    )

    decision = state.evaluate(project, video_info, video_digest)
    if decision.reuse_video:
        saved_state = json.loads(
            state.state_file_path(project.output_path).read_text(encoding="utf-8")
        )
        intermediate = Path(
            saved_state.get("intermediate_video", {}).get("path", "")
        )
        run_incremental(project, decision, intermediate)
        log_event("run_completed", mode="incremental", output=str(project.output_path))
    else:
        log_event("full_rebuild", reason=decision.reason)
        run_full(project, video_info, video_digest)
        log_event("run_completed", mode="full", output=str(project.output_path))


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass
