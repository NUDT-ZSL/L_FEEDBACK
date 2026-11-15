"""End-to-end deterministic mastering pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import __version__
from .config import load_config, normalize_config
from .dsp import (
    active_mask_from_segments,
    apply_gain_db,
    apply_lookahead_limiter,
    compressor_gain_db,
    db_to_linear,
    detect_dialogue_activity,
    estimate_true_peak,
    integrated_loudness,
    resample_audio,
)
from .errors import MasterError, internal_error
from .layouts import (
    LAYOUTS,
    format_mask,
    mapping_matrix,
    mask_for_roles,
)
from .media import package_mp4
from .wavio import (
    quantize_pcm,
    read_audio,
    resolve_source_roles,
    write_wav,
)


def _round(value: float, digits: int = 6) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    rounded = round(float(value), digits)
    return 0.0 if rounded == 0.0 else rounded


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _layout_name_from_roles(roles: Sequence[Optional[str]]) -> str:
    normalized = list(roles)
    for name, layout_roles in LAYOUTS.items():
        if normalized == layout_roles:
            return name
    return "custom"


def _load_tracks(
    config: Dict[str, Any], target_frames_hint: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], int]:
    target_rate = int(config["target_sample_rate"])
    prepared: List[Dict[str, Any]] = []
    maximum_frames = 0
    for index, track in enumerate(config["tracks"]):
        path = track["absolute_path"]
        if not os.path.isfile(path):
            raise MasterError(
                "E_INPUT_MISSING",
                "Track audio file does not exist",
                {"track_index": index, "file": track["file"], "path": path},
            )
        audio, info = read_audio(path)
        resolved_mask, source_roles, role_source = resolve_source_roles(
            info, track["input_layout"], track["channel_map"]
        )
        before_rate = int(info["sample_rate"])
        resampled, resampled_frames = resample_audio(
            audio, before_rate, target_rate
        )
        if track["gain_db"] != 0.0:
            resampled = apply_gain_db(resampled, track["gain_db"])
        maximum_frames = max(maximum_frames, resampled_frames)
        prepared.append(
            {
                "index": index,
                "track": track,
                "info": info,
                "audio": resampled,
                "resampled_frames": resampled_frames,
                "source_roles": source_roles,
                "resolved_mask": resolved_mask,
                "role_source": role_source,
            }
        )
    if target_frames_hint is not None:
        maximum_frames = max(maximum_frames, target_frames_hint)
    if maximum_frames <= 0:
        raise MasterError("E_INPUT_MISSING", "Input tracks contain no audio frames")
    return prepared, maximum_frames


def _render_stems(
    prepared: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    total_frames: int,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    output_roles = list(config["output_roles"])
    output_channels = len(output_roles)
    layout_name = config["output_layout_name"]
    stems: Dict[str, np.ndarray] = {}
    track_reports: List[Dict[str, Any]] = []

    for item in prepared:
        track = item["track"]
        info = item["info"]
        resampled = item["audio"]
        source_roles = item["source_roles"]
        matrix, mapping_details = mapping_matrix(
            source_roles, output_roles, layout_name
        )
        mapped = resampled @ matrix
        padded = np.zeros((total_frames, output_channels), dtype=np.float64)
        padded[: mapped.shape[0], :] = mapped
        group = track["group"]
        if group not in stems:
            stems[group] = np.zeros((total_frames, output_channels), dtype=np.float64)
        stems[group] += padded

        track_reports.append(
            {
                "index": item["index"],
                "file": track["file"],
                "group": group,
                "sample_rate_before": int(info["sample_rate"]),
                "sample_rate_after": int(config["target_sample_rate"]),
                "resampled": int(info["sample_rate"])
                != int(config["target_sample_rate"]),
                "resampler": "fft_bandlimited_brickwall",
                "anti_alias_cutoff_hz": min(
                    int(info["sample_rate"]), int(config["target_sample_rate"])
                )
                / 2.0,
                "input_frames": int(info["data_frames"]),
                "resampled_frames": int(item["resampled_frames"]),
                "output_timeline_frames": int(total_frames),
                "padded_frames": int(total_frames - item["resampled_frames"]),
                "input_duration_sec": _round(
                    info["data_frames"] / float(info["sample_rate"]), 9
                ),
                "resampled_duration_sec": _round(
                    item["resampled_frames"] / float(config["target_sample_rate"]), 9
                ),
                "input_channels": int(info["channels"]),
                "input_format": info["format_name"],
                "input_channel_mask": info["channel_mask_hex"],
                "resolved_channel_mask": format_mask(item["resolved_mask"]),
                "channel_mask_source": item["role_source"],
                "resolved_input_layout": _layout_name_from_roles(source_roles),
                "channel_roles": list(source_roles),
                "channel_mapping": mapping_details,
                "track_gain_db": track["gain_db"],
            }
        )
    return stems, track_reports


def _group_loudness(
    stems: Dict[str, np.ndarray],
    sample_rate: int,
    roles: Sequence[str],
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for group, stem in stems.items():
        result[group] = integrated_loudness(stem, sample_rate, roles)
    return result


def _relative_loudness(
    group_loudness: Dict[str, float], dialogue_lufs: float
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for group, value in group_loudness.items():
        if group == "dialogue" or not math.isfinite(value):
            continue
        result[group] = value - dialogue_lufs
    return result


def _cap_relative_levels(
    stems: Dict[str, np.ndarray],
    sample_rate: int,
    roles: Sequence[str],
    dialogue_lufs: float,
    limit_db: float,
    configured_limit_db: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    adjusted = {group: stem.copy() for group, stem in stems.items()}
    corrections: Dict[str, float] = {}
    notes: List[str] = []
    for _ in range(4):
        changed = False
        measured = _group_loudness(adjusted, sample_rate, roles)
        for group in list(adjusted):
            if group == "dialogue" or not math.isfinite(measured.get(group, float("-inf"))):
                continue
            offset = measured[group] - dialogue_lufs
            if offset > limit_db + 1.0e-6:
                delta = limit_db - offset
                adjusted[group] = apply_gain_db(adjusted[group], delta)
                corrections[group] = corrections.get(group, 0.0) + delta
                changed = True
        if not changed:
            break

    non_dialogue = [group for group in adjusted if group != "dialogue"]
    aggregate_correction = 0.0
    if non_dialogue:
        aggregate = np.sum(
            [adjusted[group] for group in non_dialogue], axis=0
        )
        aggregate_lufs = integrated_loudness(aggregate, sample_rate, roles)
        if math.isfinite(aggregate_lufs):
            offset = aggregate_lufs - dialogue_lufs
            if offset > limit_db + 1.0e-6:
                aggregate_correction = limit_db - offset
                for group in non_dialogue:
                    adjusted[group] = apply_gain_db(
                        adjusted[group], aggregate_correction
                    )
                notes.append(
                    "Combined music/effects bus exceeded the dialogue-relative limit; "
                    "all non-dialogue groups were attenuated together."
                )
    final_loudness = _group_loudness(adjusted, sample_rate, roles)
    final_offsets = _relative_loudness(final_loudness, dialogue_lufs)
    if any(offset > limit_db + 0.05 for offset in final_offsets.values()):
        notes.append(
            "Relative loudness gating changed after attenuation; final offsets are reported."
        )
    return adjusted, {
        "configured_limit_db": configured_limit_db,
        "effective_limit_db": limit_db,
        "hard_rule": "non_dialogue_groups_must_not_exceed_dialogue_loudness",
        "individual_corrections_db": {
            group: _round(value) for group, value in corrections.items()
        },
        "aggregate_correction_db": _round(aggregate_correction),
        "final_group_lufs": {
            group: _round(value) for group, value in final_loudness.items()
        },
        "final_offsets_db": {
            group: _round(value) for group, value in final_offsets.items()
        },
        "cap_satisfied": all(
            offset <= limit_db + 0.05 for offset in final_offsets.values()
        ),
        "notes": notes,
    }


def _normalization_and_limiting(
    stems: Dict[str, np.ndarray],
    config: Dict[str, Any],
    dialogue_lufs: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    sample_rate = int(config["target_sample_rate"])
    roles = list(config["output_roles"])
    target_lufs = float(config["target_lufs"])
    tolerance = float(config["loudness_tolerance_lu"])
    limiter_config = config["limiter"]
    normalization_limit = int(config["normalization_iteration_limit"])
    iteration_limit = min(
        normalization_limit, int(limiter_config["iteration_limit"])
    )
    max_gain = config.get("max_normalization_gain_db")

    program = np.sum(list(stems.values()), axis=0)
    initial_lufs = integrated_loudness(program, sample_rate, roles)
    if not math.isfinite(initial_lufs):
        raise MasterError(
            "E_NO_DIALOGUE", "Program mix is silent and cannot be normalized"
        )
    dialogue_driven_gain = target_lufs - dialogue_lufs
    initial_gain = dialogue_driven_gain
    gain_limited = False
    if max_gain is not None and initial_gain > max_gain:
        initial_gain = float(max_gain)
        gain_limited = True

    current_gain = initial_gain
    iterations: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for iteration_index in range(1, iteration_limit + 1):
        gain_linear = db_to_linear(current_gain)
        pre_stems = {group: stem * gain_linear for group, stem in stems.items()}
        pre_mix = np.sum(list(pre_stems.values()), axis=0)
        limited_mix, limiter_gain, limiter_stats = apply_lookahead_limiter(
            pre_mix,
            sample_rate,
            float(config["true_peak_limit_dbtp"]),
            float(limiter_config["lookahead_ms"]),
            float(limiter_config["release_ms"]),
            int(limiter_config["oversample"]),
            float(limiter_config["safety_margin_db"]),
        )
        final_mix = pre_mix * limiter_gain[:, np.newaxis]
        lufs = integrated_loudness(final_mix, sample_rate, roles)
        peak_linear, peak_db = estimate_true_peak(
            final_mix, sample_rate, int(limiter_config["oversample"])
        )
        tp_pass = peak_db <= float(config["true_peak_limit_dbtp"]) + 1.0e-7
        tolerance_pass = abs(lufs - target_lufs) <= tolerance
        record = {
            "iteration": iteration_index,
            "pre_limiter_gain_db": _round(current_gain),
            "program_lufs": _round(lufs),
            "true_peak_dbtp": _round(peak_db),
            "true_peak_pass": bool(tp_pass),
            "loudness_pass": bool(tolerance_pass),
            "limiter": limiter_stats,
        }
        iterations.append(record)

        score = abs(lufs - target_lufs)
        candidate = {
            "score": score,
            "gain_db": current_gain,
            "lufs": lufs,
            "true_peak_dbtp": peak_db,
            "limiter_gain": limiter_gain,
            "record": record,
            "tp_pass": tp_pass,
        }
        if tp_pass and (
            best is None or candidate["score"] < best["score"]
        ):
            best = candidate
        if tp_pass and tolerance_pass:
            best = candidate
            break

        if not tp_pass:
            current_gain -= (peak_db - float(config["true_peak_limit_dbtp"])) + 0.1
        else:
            current_gain += target_lufs - lufs
        if max_gain is not None and current_gain > max_gain:
            current_gain = float(max_gain)

    if best is None:
        raise MasterError(
            "E_INTERNAL",
            "Unable to find a true-peak-compliant normalization pass",
            {"iterations": iterations},
        )

    gain_linear = db_to_linear(best["gain_db"])
    final_stems = {
        group: stem * gain_linear * best["limiter_gain"][:, np.newaxis]
        for group, stem in stems.items()
    }
    final_mix = np.sum(list(final_stems.values()), axis=0)
    final_group_lufs = _group_loudness(final_stems, sample_rate, roles)
    final_offsets = _relative_loudness(final_group_lufs, dialogue_lufs)
    others_not_above_dialogue = all(
        offset <= 0.05 for offset in final_offsets.values()
    )
    within_tolerance = abs(best["lufs"] - target_lufs) <= tolerance
    tradeoffs: List[Dict[str, Any]] = []
    if not within_tolerance:
        tradeoffs.append(
            {
                "code": "TRUE_PEAK_OR_ITERATION_LIMIT",
                "message": (
                    "The true-peak ceiling and iteration limit prevented the final "
                    "program loudness from reaching the target tolerance."
                ),
                "final_program_lufs": _round(best["lufs"]),
                "target_lufs": target_lufs,
            }
        )
    if gain_limited:
        tradeoffs.append(
            {
                "code": "MAX_NORMALIZATION_GAIN",
                "message": (
                    "The configured maximum normalization gain was reached; the "
                    "program target therefore has priority over a larger gain increase."
                ),
                "max_gain_db": max_gain,
            }
        )
    final_dialogue_lufs = final_group_lufs.get("dialogue", float("-inf"))
    normalization = {
        "target_lufs": target_lufs,
        "tolerance_lu": tolerance,
        "initial_program_lufs": _round(initial_lufs),
        "dialogue_driven_initial_gain_db": _round(dialogue_driven_gain),
        "initial_global_gain_db": _round(initial_gain),
        "final_global_gain_db": _round(best["gain_db"]),
        "program_lufs": _round(best["lufs"]),
        "target_error_lu": _round(best["lufs"] - target_lufs),
        "within_tolerance": bool(within_tolerance),
        "dialogue_lufs": _round(final_dialogue_lufs),
        "dialogue_target_error_lu": _round(final_dialogue_lufs - target_lufs),
        "other_groups_lufs": {
            group: _round(value)
            for group, value in final_group_lufs.items()
            if group != "dialogue"
        },
        "relative_offsets_db": {
            group: _round(value) for group, value in final_offsets.items()
        },
        "dialogue_priority": {
            "hard_rule": "music_and_effects_must_not_exceed_dialogue_loudness",
            "configured_relative_gain_limit_db": config["max_relative_gain_db"],
            "all_other_groups_at_or_below_dialogue": bool(
                others_not_above_dialogue
            ),
        },
        "iterations": iterations,
        "iteration_count": len(iterations),
        "tradeoffs": tradeoffs,
        "dialogue_priority_note": (
            "The global gain is initialized from the dialogue loudness, then "
            "iterated so the integrated program loudness reaches the configured "
            "target while the true-peak ceiling remains satisfied."
        ),
    }
    limiter_summary = {
        "lookahead_ms": limiter_config["lookahead_ms"],
        "release_ms": limiter_config["release_ms"],
        "oversample": limiter_config["oversample"],
        "iteration_count": len(iterations),
        "iteration_normalization_limit": normalization_limit,
        "iteration_limit": iteration_limit,
        "final": best["record"]["limiter"],
    }
    return final_mix, final_stems, {
        "normalization": normalization,
        "limiter": limiter_summary,
    }


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    text = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _build_report(
    input_dir: str,
    output_dir: str,
    config_path: str,
    config_sha256: str,
    config: Dict[str, Any],
    total_frames: int,
    tracks: Sequence[Dict[str, Any]],
    activity: Dict[str, Any],
    compression: Dict[str, Any],
    loudness: Dict[str, Any],
    wav_info: Dict[str, Any],
    media_info: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "ok",
        "tool": {
            "name": "cinema-audio-master",
            "version": __version__,
            "implementation": "numpy-offline",
        },
        "input": {
            "directory": os.path.abspath(input_dir),
            "config": os.path.abspath(config_path),
            "config_sha256": config_sha256,
            "track_count": len(config["tracks"]),
        },
        "parameters": {
            "target_sample_rate": config["target_sample_rate"],
            "output_layout": config["output_layout_name"],
            "output_channels": config["output_roles"],
            "output_channel_mask": format_mask(config["output_channel_mask"]),
            "target_lufs": config["target_lufs"],
            "true_peak_limit_dbtp": config["true_peak_limit_dbtp"],
            "max_relative_gain_db": config["max_relative_gain_db"],
            "compression": config["compression"],
            "dialogue_detection": config["dialogue_detection"],
            "limiter": config["limiter"],
            "normalization_iteration_limit": config[
                "normalization_iteration_limit"
            ],
            "output": config["output"],
        },
        "tracks": list(tracks),
        "dialogue_activity": activity,
        "compression": compression,
        "loudness": loudness["normalization"],
        "limiter": loudness["limiter"],
        "true_peak": {
            "limit_dbtp": config["true_peak_limit_dbtp"],
            "measured_dbtp": wav_info["true_peak_dbtp"],
            "method": "{}x_fft_oversampling".format(
                config["limiter"]["oversample"]
            ),
            "pass": bool(
                wav_info["true_peak_dbtp"]
                <= config["true_peak_limit_dbtp"] + 1.0e-7
            ),
        },
        "output": {
            "wav": wav_info,
            "container": media_info,
        },
        "determinism": {
            "wav_sha256": wav_info["sha256"],
            "report_policy": (
                "Sorted JSON without wall-clock timestamps; WAV PCM payload and "
                "report are deterministic for identical inputs and configuration."
            ),
        },
    }


def run_mastering(
    input_dir: str,
    output_dir: str,
    explicit_config: Optional[str] = None,
) -> Dict[str, Any]:
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    if not os.path.isdir(input_dir):
        raise MasterError(
            "E_INPUT_MISSING", "Input directory does not exist", {"input_dir": input_dir}
        )
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        raise MasterError(
            "E_OUTPUT_DIR",
            "Unable to create the output directory",
            {"output_dir": output_dir, "error": str(exc)},
        )

    stage = tempfile.mkdtemp(prefix=".cinema-master-", dir=output_dir)
    try:
        raw_config, config, config_path, config_sha256 = load_config(
            input_dir, explicit_config
        )
        prepared, total_frames = _load_tracks(config)
        if total_frames <= 0 or len(prepared) == 0:
            raise MasterError(
                "E_NO_DIALOGUE", "No usable audio frames were found"
            )
        stems, track_reports = _render_stems(prepared, config, total_frames)
        if "dialogue" not in stems:
            raise MasterError(
                "E_CONFIG_SCHEMA", "At least one dialogue track is required"
            )
        sample_rate = int(config["target_sample_rate"])
        roles = list(config["output_roles"])

        program_before_compression = np.sum(list(stems.values()), axis=0)
        pre_compression_lufs = integrated_loudness(
            program_before_compression, sample_rate, roles
        )
        detection_config = config["dialogue_detection"]
        detection = detect_dialogue_activity(
            stems["dialogue"],
            sample_rate,
            float(detection_config["frame_ms"]),
            float(detection_config["overlap"]),
            float(detection_config["threshold_dbfs"]),
            float(detection_config["min_segment_ms"]),
        )
        if not detection["segments"]:
            raise MasterError(
                "E_NO_DIALOGUE",
                "No dialogue activity passed the configured detector threshold",
                {
                    "threshold_dbfs": detection_config["threshold_dbfs"],
                    "frame_ms": detection_config["frame_ms"],
                    "overlap": detection_config["overlap"],
                    "frame_count": detection["frame_count"],
                },
            )
        active_mask = active_mask_from_segments(detection["segments"], total_frames)
        compression_config = config["compression"]
        compression_gain, compression_stats = compressor_gain_db(
            stems["dialogue"],
            active_mask,
            sample_rate,
            float(compression_config["threshold_dbfs"]),
            float(compression_config["ratio"]),
            float(compression_config["attack_ms"]),
            float(compression_config["release_ms"]),
            float(compression_config["knee_db"]),
            bool(compression_config["enabled"]),
        )
        compression_linear = np.power(10.0, compression_gain / 20.0)
        compressed_stems = {
            group: stem * compression_linear[:, np.newaxis]
            for group, stem in stems.items()
        }
        program_after_compression = np.sum(list(compressed_stems.values()), axis=0)
        post_compression_lufs = integrated_loudness(
            program_after_compression, sample_rate, roles
        )

        group_loudness = _group_loudness(compressed_stems, sample_rate, roles)
        dialogue_lufs = group_loudness.get("dialogue", float("-inf"))
        if not math.isfinite(dialogue_lufs):
            raise MasterError(
                "E_NO_DIALOGUE",
                "Dialogue stem integrates below the loudness noise floor",
            )
        adjusted_stems, relative_cap = _cap_relative_levels(
            compressed_stems,
            sample_rate,
            roles,
            dialogue_lufs,
            min(0.0, float(config["max_relative_gain_db"])),
            float(config["max_relative_gain_db"]),
        )
        final_mix_float, final_stems_float, loudness_results = (
            _normalization_and_limiting(adjusted_stems, config, dialogue_lufs)
        )
        if not relative_cap["cap_satisfied"] or not loudness_results[
            "normalization"
        ]["dialogue_priority"]["all_other_groups_at_or_below_dialogue"]:
            loudness_results["normalization"]["tradeoffs"].append(
                {
                    "code": "RELATIVE_LOUDNESS_CONSTRAINT",
                    "message": (
                        "Final integrated-loudness gating prevented a non-dialogue "
                        "group from being fully aligned below the dialogue level."
                    ),
                    "final_offsets_db": loudness_results["normalization"][
                        "relative_offsets_db"
                    ],
                }
            )

        bits = int(config["output"]["bits_per_sample"])
        quantized_mix = quantize_pcm(final_mix_float, bits)
        peak_linear, peak_db = estimate_true_peak(
            quantized_mix,
            sample_rate,
            int(config["limiter"]["oversample"]),
        )
        if peak_db > float(config["true_peak_limit_dbtp"]) + 1.0e-7:
            ceiling_linear = db_to_linear(float(config["true_peak_limit_dbtp"]))
            trim = ceiling_linear * 0.999999 / max(peak_linear, 1.0e-20)
            quantized_mix = quantize_pcm(quantized_mix * trim, bits)
            peak_linear, peak_db = estimate_true_peak(
                quantized_mix,
                sample_rate,
                int(config["limiter"]["oversample"]),
            )
            loudness_results["limiter"]["quantization_safety_trim_db"] = _round(
                -20.0 * math.log10(trim)
            )
        final_lufs = integrated_loudness(quantized_mix, sample_rate, roles)
        loudness_results["normalization"]["program_lufs"] = _round(final_lufs)
        loudness_results["normalization"]["target_error_lu"] = _round(
            final_lufs - float(config["target_lufs"])
        )
        loudness_results["normalization"]["within_tolerance"] = bool(
            abs(final_lufs - float(config["target_lufs"]))
            <= float(config["loudness_tolerance_lu"])
        )

        basename = config["output"]["basename"]
        wav_name = basename + ".wav"
        mp4_name = basename + ".mp4"
        report_name = "report.json"
        stage_wav = os.path.join(stage, wav_name)
        stage_mp4 = os.path.join(stage, mp4_name)
        stage_report = os.path.join(stage, report_name)
        write_wav(
            stage_wav,
            quantized_mix,
            sample_rate,
            int(config["output_channel_mask"]),
            bits_per_sample=bits,
            extensible=True,
        )
        wav_sha256 = _sha256_file(stage_wav)
        duration = total_frames / float(sample_rate)
        media_info = package_mp4(
            stage_wav,
            stage_mp4,
            sample_rate,
            duration,
            config["output_layout_name"],
            float(config["target_lufs"]),
            float(final_lufs),
            float(peak_db),
            config,
        )

        activity_report = {
            "method": "framed_mean_square_with_configured_overlap",
            "frame_ms": detection_config["frame_ms"],
            "overlap": detection_config["overlap"],
            "frame_samples": detection["frame_samples"],
            "hop_samples": detection["hop_samples"],
            "frame_count": detection["frame_count"],
            "active_frame_count": detection["active_frame_count"],
            "threshold_dbfs": detection_config["threshold_dbfs"],
            "min_segment_ms": detection_config["min_segment_ms"],
            "segment_count": len(detection["segments"]),
            "total_active_sec": _round(
                sum(
                    segment["duration_sec"]
                    for segment in detection["segments"]
                ),
                9,
            ),
            "segments": detection["segments"],
        }
        compression_report = {
            **compression_stats,
            "threshold_dbfs": compression_config["threshold_dbfs"],
            "ratio": compression_config["ratio"],
            "attack_ms": compression_config["attack_ms"],
            "release_ms": compression_config["release_ms"],
            "knee_db": compression_config["knee_db"],
            "pre_compression_program_lufs": _round(pre_compression_lufs),
            "post_compression_program_lufs": _round(post_compression_lufs),
            "program_gain_change_db": _round(
                post_compression_lufs - pre_compression_lufs
            ),
            "active_samples": int(active_mask.sum()),
            "relative_level_cap": relative_cap,
        }
        wav_info = {
            "path": os.path.join(output_dir, wav_name),
            "format": "WAVE_FORMAT_EXTENSIBLE",
            "sample_rate": sample_rate,
            "channels": len(roles),
            "channel_layout": config["output_layout_name"],
            "channel_roles": roles,
            "channel_mask": format_mask(config["output_channel_mask"]),
            "bits_per_sample": bits,
            "frames": total_frames,
            "duration_sec": _round(duration, 9),
            "size_bytes": os.path.getsize(stage_wav),
            "sha256": wav_sha256,
            "program_lufs": _round(final_lufs),
            "true_peak_dbtp": _round(peak_db),
        }
        media_info["path"] = os.path.join(output_dir, mp4_name)
        report = _build_report(
            input_dir,
            output_dir,
            config_path,
            config_sha256,
            config,
            total_frames,
            track_reports,
            activity_report,
            compression_report,
            loudness_results,
            wav_info,
            media_info,
        )
        report["output"]["report_path"] = os.path.join(output_dir, report_name)
        _write_json(stage_report, report)

        final_wav = os.path.join(output_dir, wav_name)
        final_mp4 = os.path.join(output_dir, mp4_name)
        final_report = os.path.join(output_dir, report_name)
        try:
            os.replace(stage_wav, final_wav)
            os.replace(stage_mp4, final_mp4)
            os.replace(stage_report, final_report)
        except OSError as exc:
            raise MasterError(
                "E_OUTPUT_WRITE",
                "Unable to publish output artifacts",
                {"output_dir": output_dir, "error": str(exc)},
            )
        return report
    except MasterError:
        raise
    except Exception as exc:
        raise internal_error(exc)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def write_error_report(
    output_dir: str, error: MasterError, input_dir: Optional[str] = None
) -> Optional[str]:
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(os.path.abspath(output_dir), "report.json")
        payload = {
            "schema_version": 1,
            "status": "error",
            "tool": {
                "name": "cinema-audio-master",
                "version": __version__,
            },
            "input_directory": os.path.abspath(input_dir) if input_dir else None,
            "error": error.to_dict(),
        }
        _write_json(path, payload)
        return path
    except Exception:
        return None

