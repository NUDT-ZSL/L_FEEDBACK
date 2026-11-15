from __future__ import annotations

import bisect
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .alignment import estimate_offset
from .dvb import (
    PAGE_ID,
    build_clear_segments,
    build_show_segments,
    build_subtitle_pes,
    make_region,
)
from .errors import BusinessError
from .ffmpeg_tools import FFmpegTools, resolve_ffmpeg, transcode_base_ts
from .mpegts import (
    StreamInfo,
    audio_pes_points,
    build_pmt_section,
    choose_free_pid,
    find_pmt,
    iter_ts_packets,
)
from .mux import AuxAction, inject_actions
from .recovery import load_state, save_state
from .resample import sinc_resample
from .samplegen import SampleSpec, generate_sample, generate_samples
from .scte35 import (
    PTS_MASK,
    build_event_sections,
    descriptor_summary,
    parse_command_table,
    parse_splice_info,
)
from .subtitles import (
    Cue,
    apply_global_offset,
    correct_timeline,
    parse_subtitles,
    timeline_metrics,
    write_srt,
)
from .verify import verify_output
from .wavio import read_wav16, write_wav16


@dataclass
class RunConfig:
    audio: Path
    subtitles: Path
    scte35: Path
    output_dir: Path
    overlap_policy: str = "truncate"
    ffmpeg: str | None = None
    ffprobe: str | None = None
    force: bool = False


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(config: RunConfig) -> dict[str, str]:
    return {
        "audio_sha256": sha256_file(config.audio),
        "subtitles_sha256": sha256_file(config.subtitles),
        "scte35_sha256": sha256_file(config.scte35),
        "overlap_policy": config.overlap_policy,
        "tool_version": __version__,
    }


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _output_paths(config: RunConfig) -> dict[str, Path]:
    stem = config.audio.stem
    work = config.output_dir / ".work"
    return {
        "work": work,
        "base_ts": work / "base_audio_video.ts",
        "resampled_wav": work / "resampled_48k.wav",
        "partial_ts": work / "final.with_scte35.ts.partial",
        "final_ts": config.output_dir / f"{stem}.with_scte35.ts",
        "corrected_srt": config.output_dir / f"{stem}.corrected.srt",
        "report": config.output_dir / "alignment_report.json",
        "corrections": config.output_dir / "subtitle_corrections.json",
        "scte_log": config.output_dir / "scte35_log.json",
        "verification": config.output_dir / "verification_summary.json",
        "state": work / "recovery.json",
        "render": work / "render",
    }


def _reset_outputs(paths: dict[str, Path]) -> None:
    for key in ("base_ts", "resampled_wav", "partial_ts", "final_ts", "state"):
        path = paths[key]
        if path.is_file():
            path.unlink()


def _state_base(config: RunConfig, fingerprint: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "tool_version": __version__,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "prepared",
        "failed_stage": None,
        "error": None,
        "fingerprint": fingerprint,
        "completed_transcode_segments": [],
        "completed_event_ids": [],
        "completed_aux_actions": [],
        "next_base_index": 0,
        "next_action_offset": 0,
        "partial_size": 0,
        "continuity_counters": {},
    }


def _prepare_actions(
    base_ts_path: Path,
    corrected_cues: list[Cue],
    scte_events: list[Any],
    subtitle_pid: int,
    scte_pid: int,
    render_root: Path,
    ffmpeg: str,
) -> tuple[list[AuxAction], dict[str, int], dict[str, int]]:
    data = base_ts_path.read_bytes()
    pmt = find_pmt(data)
    audio_stream = next(
        (
            stream
            for stream in pmt.streams
            if stream.stream_type in (0x0F, 0x03, 0x04)
        ),
        None,
    )
    if audio_stream is None:
        raise BusinessError("E_TRANSCODE", "transport_analysis", "Encoded TS has no audio stream")
    audio_points = audio_pes_points(data, audio_stream.pid)
    if not audio_points:
        raise BusinessError("E_TRANSCODE", "transport_analysis", "Encoded TS has no audio PES packets")
    origin_pts = int(audio_points[0]["pts"])
    pts_values = [int(point["pts"]) for point in audio_points]
    base_packet_count = sum(1 for _ in iter_ts_packets(data))

    def base_index_for_pts(target_pts: int) -> int:
        position = bisect.bisect_left(pts_values, target_pts & PTS_MASK)
        if position < len(audio_points):
            return int(audio_points[position]["packet_index"])
        return base_packet_count

    actions: list[AuxAction] = []
    render_cache: dict[str, Any] = {}
    render_methods: dict[str, int] = {}
    clear_count = 0
    suppressed_clears = 0
    for cue_index, cue in enumerate(corrected_cues):
        region = render_cache.get(cue.text)
        if region is None:
            region = make_region(cue.text, ffmpeg, render_root)
            render_cache[cue.text] = region
        render_methods[region.render_method] = render_methods.get(region.render_method, 0) + 1
        show_pts = (origin_pts + int(round(cue.start_ms * 90))) & PTS_MASK
        clear_pts = (origin_pts + int(round(cue.end_ms * 90))) & PTS_MASK
        show_pes = build_subtitle_pes(build_show_segments(region, cue.index & 0x0F), show_pts)
        clear_pes = build_subtitle_pes(build_clear_segments((cue.index + 1) & 0x0F), clear_pts)
        actions.append(
            AuxAction(
                action_id=f"dvb:show:{cue.index}",
                base_packet_index=base_index_for_pts(show_pts),
                target_ms=cue.start_ms,
                kind="pes",
                pid=subtitle_pid,
                payload=show_pes,
                detail={
                    "cue_index": cue.index,
                    "pts_90k": show_pts,
                    "region_width": region.width,
                    "region_height": region.height,
                    "region_x": region.x,
                    "region_y": region.y,
                    "render_method": region.render_method,
                },
            )
        )
        next_cue = corrected_cues[cue_index + 1] if cue_index + 1 < len(corrected_cues) else None
        if next_cue is not None and abs(next_cue.start_ms - cue.end_ms) < 0.5:
            suppressed_clears += 1
        else:
            actions.append(
                AuxAction(
                    action_id=f"dvb:clear:{cue.index}",
                    base_packet_index=base_index_for_pts(clear_pts),
                    target_ms=cue.end_ms,
                    kind="pes",
                    pid=subtitle_pid,
                    payload=clear_pes,
                    detail={"cue_index": cue.index, "pts_90k": clear_pts},
                )
            )
            clear_count += 1

    for event in scte_events:
        event.target_pts = (origin_pts + int(round(event.offset_sample * 90000 / 48000))) & PTS_MASK
        event.section = build_event_sections([event], origin_pts)[0].section
        actions.append(
            AuxAction(
                action_id=f"scte:{event.event_id}",
                base_packet_index=base_index_for_pts(event.target_pts),
                target_ms=event.offset_ms,
                kind="section",
                pid=scte_pid,
                payload=event.section or b"",
                event_id=event.event_id,
                detail={
                    "event_id": event.event_id,
                    "target_pts_90k": event.target_pts,
                    "offset_sample_48k": event.offset_sample,
                },
            )
        )
    actions.sort(key=lambda item: (item.base_packet_index, item.target_ms, item.action_id))
    render_methods["show_sets"] = len(corrected_cues)
    render_methods["clear_sets"] = clear_count
    render_methods["suppressed_abutting_clears"] = suppressed_clears
    return actions, {"origin_pts": origin_pts, "audio_pid": audio_stream.pid}, render_methods


def run_pipeline(config: RunConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(config)
    paths["work"].mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(config)
    state = load_state(paths["state"])
    if config.force:
        _reset_outputs(paths)
        state = None
    if state is not None and state.get("fingerprint") != fingerprint:
        raise BusinessError(
            "E_INPUT",
            "recovery",
            "Recovery state belongs to different inputs; use --force to start over",
            {"state": str(paths["state"])},
        )
    if state is None:
        state = _state_base(config, fingerprint)
        save_state(paths["state"], state)

    try:
        tools = resolve_ffmpeg(config.ffmpeg, config.ffprobe)
        wav = read_wav16(config.audio)
        internal = sinc_resample(wav.samples, wav.sample_rate, 48000)
        duration_ms = internal.shape[0] * 1000.0 / 48000.0
        raw_cues, subtitle_format = parse_subtitles(config.subtitles)
        raw_metrics = timeline_metrics(raw_cues)
        corrected, corrections = correct_timeline(raw_cues, config.overlap_policy)
        structural_metrics = timeline_metrics(corrected)
        alignment = estimate_offset(internal, corrected, 48000, 2000, 10)
        aligned_cues, clamp_corrections = apply_global_offset(corrected, alignment.offset_ms, duration_ms)
        corrections.extend(clamp_corrections)
        after_metrics = timeline_metrics(aligned_cues)
        write_srt(paths["corrected_srt"], aligned_cues)
        write_wav16(paths["resampled_wav"], internal, 48000)
        _write_json(
            paths["corrections"],
            {
                "policy": config.overlap_policy,
                "correction_count": len(corrections),
                "corrections": [item.as_dict() for item in corrections],
            },
        )

        events = parse_command_table(config.scte35, duration_ms, 48000)
        _write_json(
            paths["report"],
            {
                "schema_version": 1,
                "tool_version": __version__,
                "status": "aligned_pre_transcode",
                "inputs": {
                    "audio": str(config.audio),
                    "subtitles": str(config.subtitles),
                    "scte35": str(config.scte35),
                    "fingerprint": fingerprint,
                },
                "audio": {
                    "source_sample_rate": wav.sample_rate,
                    "source_channels": wav.channels,
                    "channel_layout": wav.channel_layout,
                    "source_format": wav.source_format,
                    "source_frames": wav.nframes,
                    "internal_sample_rate": 48000,
                    "internal_frames": int(internal.shape[0]),
                    "internal_duration_ms": round(duration_ms, 6),
                    "resampler": "windowed-sinc, 24 zero crossings, anti-alias low-pass",
                    "resampled_wav": str(paths["resampled_wav"]),
                },
                "subtitles": {
                    "input_format": subtitle_format,
                    "raw_cue_count": len(raw_cues),
                    "corrected_cue_count": len(aligned_cues),
                    "overlap_policy": config.overlap_policy,
                    "raw_metrics": raw_metrics.as_dict(),
                    "after_structural_correction": structural_metrics.as_dict(),
                    "after_global_offset": after_metrics.as_dict(),
                    "correction_log": str(paths["corrections"]),
                    "corrected_srt": str(paths["corrected_srt"]),
                    "corrected_starts_ms": [round(cue.start_ms, 3) for cue in aligned_cues],
                },
                "alignment": {
                    **alignment.as_dict(),
                    "offset_before_metrics": structural_metrics.as_dict(),
                    "offset_after_metrics": after_metrics.as_dict(),
                },
                "scte35": {"event_count": len(events)},
                "outputs": {
                    "mpeg_ts": str(paths["final_ts"]),
                    "alignment_report": str(paths["report"]),
                    "scte35_log": str(paths["scte_log"]),
                    "verification": str(paths["verification"]),
                },
            },
        )
        base_ready = False
        if paths["base_ts"].is_file() and state.get("completed_transcode_segments"):
            try:
                find_pmt(paths["base_ts"].read_bytes())
                base_ready = True
            except (ValueError, OSError):
                base_ready = False
        if not base_ready:
            transcode_base_ts(
                tools.ffmpeg,
                paths["resampled_wav"],
                paths["base_ts"],
                duration_ms / 1000.0,
                wav.channels,
            )
            base_ready = True
            state.update(
                {
                    "stage": "audio_encoded",
                    "completed_transcode_segments": [
                        {
                            "name": "base_audio_video_mpegts",
                            "path": str(paths["base_ts"]),
                            "audio_codec": "aac_lc",
                            "video_codec": "mpeg2video_black",
                        }
                    ],
                }
            )
            save_state(paths["state"], state)
        if not base_ready:
            raise BusinessError("E_STATE", "transcode", "Base TS was not produced")

        base_data = paths["base_ts"].read_bytes()
        original_pmt = find_pmt(base_data)
        used_pids = {stream.pid for stream in original_pmt.streams}
        used_pids.add(original_pmt.pmt_pid)
        used_pids.add(0)
        used_pids.add(0x11)
        scte_pid = choose_free_pid(used_pids, 0x0100)
        used_pids.add(scte_pid)
        subtitle_pid = choose_free_pid(used_pids, 0x0120)
        subtitle_descriptor = b"\x59\x08" + b"eng" + b"\x10" + PAGE_ID.to_bytes(2, "big") * 2
        extra_streams = [
            StreamInfo(0x06, subtitle_pid, subtitle_descriptor),
            StreamInfo(0x86, scte_pid, b""),
        ]
        new_pmt = build_pmt_section(original_pmt, extra_streams)
        actions, transport_info, render_methods = _prepare_actions(
            paths["base_ts"],
            aligned_cues,
            events,
            subtitle_pid,
            scte_pid,
            paths["render"],
            tools.ffmpeg,
        )

        expected_ids = {event.event_id for event in events}
        existing_nonempty_actions = bool(actions)
        state.update(
            {
                "stage": "subtitles_rendered",
                "scte35_pid": scte_pid,
                "subtitle_pid": subtitle_pid,
                "pmt_pid": original_pmt.pmt_pid,
                "audio_pid": transport_info["audio_pid"],
                "origin_pts": transport_info["origin_pts"],
                "expected_event_ids": sorted(expected_ids),
                "expected_action_ids": sorted(action.action_id for action in actions),
                "render_methods": render_methods,
            }
        )
        save_state(paths["state"], state)

        scte_packets: list[dict[str, Any]] = []
        for event in events:
            section = event.section
            if section is None:
                continue
            parsed = parse_splice_info(section)
            matching_action = next(
                action for action in actions if action.action_id == f"scte:{event.event_id}"
            )
            scte_packets.append(
                {
                    "event_id": event.event_id,
                    "wire_event_id": parsed.get("event_id"),
                    "type": event.event_type,
                    "offset_ms": event.offset_ms,
                    "offset_sample_48k": event.offset_sample,
                    "target_pts_90k": event.target_pts,
                    "pts_adjustment": parsed["pts_adjustment"],
                    "effective_splice_pts_90k": parsed.get(
                        "effective_splice_pts", parsed["pts_adjustment"]
                    ),
                    "duration_90k": int(round(event.duration_ms * 90)),
                    "section_length": parsed["section_length"],
                    "section_crc32": parsed["crc32"],
                    "section_hex": section.hex(),
                    "pid": scte_pid,
                    "transport_insertion_packet_index": matching_action.base_packet_index,
                    "transport_insertion_pts_90k": next(
                        (
                            int(point["pts"])
                            for point in audio_pes_points(base_data, transport_info["audio_pid"])
                            if int(point["packet_index"]) >= matching_action.base_packet_index
                        ),
                        event.target_pts,
                    ),
                    "segmentation_descriptor": descriptor_summary(event),
                    "carriage": "MPEG-2 private section (SCTE-35 table_id 0xFC)",
                }
            )
        scte_log = {
            "schema_version": 1,
            "transport": "MPEG-TS",
            "scte35_pid": scte_pid,
            "audio_pid": transport_info["audio_pid"],
            "origin_pts_90k": transport_info["origin_pts"],
            "event_count": len(scte_packets),
            "events": scte_packets,
        }
        _write_json(paths["scte_log"], scte_log)
        _write_json(
            paths["corrections"],
            {
                "policy": config.overlap_policy,
                "correction_count": len(corrections),
                "corrections": [item.as_dict() for item in corrections],
            },
        )
        report = {
            "schema_version": 1,
            "tool_version": __version__,
            "status": "prepared",
            "inputs": {
                "audio": str(config.audio),
                "subtitles": str(config.subtitles),
                "scte35": str(config.scte35),
                "fingerprint": fingerprint,
            },
            "audio": {
                "source_sample_rate": wav.sample_rate,
                "source_channels": wav.channels,
                "channel_layout": wav.channel_layout,
                "source_format": wav.source_format,
                "source_frames": wav.nframes,
                "internal_sample_rate": 48000,
                "internal_frames": int(internal.shape[0]),
                "internal_duration_ms": round(duration_ms, 6),
                "resampler": "windowed-sinc, 24 zero crossings, anti-alias low-pass",
                "resampled_wav": str(paths["resampled_wav"]),
            },
            "subtitles": {
                "input_format": subtitle_format,
                "raw_cue_count": len(raw_cues),
                "corrected_cue_count": len(aligned_cues),
                "overlap_policy": config.overlap_policy,
                "raw_metrics": raw_metrics.as_dict(),
                "after_structural_correction": structural_metrics.as_dict(),
                "after_global_offset": after_metrics.as_dict(),
                "correction_log": str(paths["corrections"]),
                "corrected_srt": str(paths["corrected_srt"]),
                "corrected_starts_ms": [round(cue.start_ms, 3) for cue in aligned_cues],
            },
            "alignment": {
                **alignment.as_dict(),
                "offset_before_metrics": structural_metrics.as_dict(),
                "offset_after_metrics": after_metrics.as_dict(),
            },
            "scte35": {
                "event_count": len(events),
                "log": str(paths["scte_log"]),
                "pid": scte_pid,
            },
            "embedding": {
                "format": "DVB subtitles (EN 300 743)",
                "pid": subtitle_pid,
                "page_id": PAGE_ID,
                "render_methods": {
                    key: value
                    for key, value in render_methods.items()
                    if key not in {"show_sets", "clear_sets", "suppressed_abutting_clears"}
                },
                "display_sets": len(aligned_cues),
                "clear_sets": render_methods["clear_sets"],
                "suppressed_abutting_clears": render_methods["suppressed_abutting_clears"],
            },
            "transport": {
                "pmt_pid": original_pmt.pmt_pid,
                "audio_pid": transport_info["audio_pid"],
                "origin_pts_90k": transport_info["origin_pts"],
                "aux_action_count": len(actions),
            },
            "outputs": {
                "mpeg_ts": str(paths["final_ts"]),
                "alignment_report": str(paths["report"]),
                "scte35_log": str(paths["scte_log"]),
                "verification": str(paths["verification"]),
            },
        }
        _write_json(paths["report"], report)

        final_candidate = paths["final_ts"]
        should_inject = not (
            final_candidate.is_file()
            and state.get("stage") in {"injected", "verified", "complete"}
        )
        if should_inject:
            inject_actions(
                paths["base_ts"],
                paths["partial_ts"],
                paths["final_ts"],
                actions,
                original_pmt.pmt_pid,
                new_pmt,
                state,
                lambda value: save_state(paths["state"], value),
            )

        verification = verify_output(
            paths["final_ts"],
            tools.ffprobe,
            tools.ffmpeg,
            scte_log,
            [cue.start_ms for cue in aligned_cues],
            render_methods["clear_sets"],
        )
        _write_json(paths["verification"], verification)
        report["status"] = verification["status"]
        report["verification"] = verification
        _write_json(paths["report"], report)
        state.update(
            {
                "stage": "complete",
                "failed_stage": None,
                "error": None,
                "completed_event_ids": sorted(expected_ids),
                "verification": verification,
            }
        )
        save_state(paths["state"], state)
        return {
            "status": "ok",
            "report": report,
            "scte35_log": scte_log,
            "verification": verification,
            "outputs": report["outputs"],
        }
    except BusinessError as exc:
        state["stage"] = "failed"
        state["failed_stage"] = exc.stage
        state["error"] = {"code": exc.code, "stage": exc.stage, "message": exc.message, "details": exc.details}
        state["failed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_state(paths["state"], state)
        raise
    except KeyboardInterrupt:
        state["stage"] = "interrupted"
        state["interrupted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_state(paths["state"], state)
        raise


def generate_regression_samples(output_dir: str | Path) -> list[dict[str, object]]:
    return generate_samples(output_dir)


def run_regression(
    output_dir: str | Path,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    samples = generate_samples(root / "samples")
    results = []
    for item in samples:
        sample_dir = Path(str(item["audio"])).parent
        run_dir = root / "runs" / str(item["name"])
        try:
            result = run_pipeline(
                RunConfig(
                    audio=Path(str(item["audio"])),
                    subtitles=Path(str(item["subtitles"])),
                    scte35=Path(str(item["scte35"])),
                    output_dir=run_dir,
                    overlap_policy="truncate",
                    ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                )
            )
            ground = json.loads((sample_dir / "ground_truth.json").read_text(encoding="utf-8"))
            estimate = result["report"]["alignment"]["estimated_offset_ms"]
            error_ms = abs(float(estimate) - float(ground["delay_ms"]))
            results.append(
                {
                    "name": item["name"],
                    "status": "ok",
                    "expected_offset_ms": ground["delay_ms"],
                    "estimated_offset_ms": estimate,
                    "offset_error_ms": round(error_ms, 3),
                    "confidence": result["report"]["alignment"]["confidence"],
                    "ts": result["outputs"]["mpeg_ts"],
                    "verification": result["verification"],
                }
            )
        except BusinessError as exc:
            results.append(
                {
                    "name": item["name"],
                    "status": "error",
                    "error": json.loads(exc.as_json())["error"],
                }
            )
    summary = {
        "status": "ok" if all(item["status"] == "ok" and item["offset_error_ms"] <= 20 for item in results) else "failed",
        "sample_count": len(results),
        "results": results,
    }
    _write_json(root / "regression_summary.json", summary)
    return summary
