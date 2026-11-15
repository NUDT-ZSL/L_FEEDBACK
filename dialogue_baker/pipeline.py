from __future__ import annotations

import io
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np

from . import __version__
from .audio import (
    clipped_sample_count,
    decode_dialogue,
    db_to_linear,
    ffprobe_format,
    linear_to_db,
    lookahead_limiter_gain,
    loudness_parameters,
    measure_integrated_loudness,
    read_pcm16_wav,
    true_peak_dbtp,
    write_pcm16_wav,
)
from .errors import BakerError
from .schema import ProjectSpec, load_project
from .snapshot import STAGE_ORDER, STAGE_VERSIONS, SnapshotStore
from .spatial import spatialize_all
from .util import (
    atomic_write,
    json_safe,
    pretty_json_bytes,
    sha256_bytes,
    sha256_file,
)


WAV_FILE_NAME = "dialogue.wav"
SUBTITLE_FILE_NAME = "subtitles_frames.json"
REPORT_FILE_NAME = "bake_report.json"
TRUE_PEAK_SAFETY_DB = 0.05


@dataclass
class Artifact:
    stage: str
    input_digest: str
    snapshot_digest: str
    meta: Dict[str, Any]
    files: Dict[str, bytes]
    reused: bool


@dataclass
class BakeResult:
    status: str
    completed_stage: str
    report: Optional[Dict[str, Any]]
    report_path: Optional[Path]
    stage_actions: Dict[str, str]


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _load_npy(files: Mapping[str, bytes], name: str) -> np.ndarray:
    try:
        data = files[name]
    except KeyError as exc:
        raise BakerError(
            "E_SNAPSHOT_STATE",
            "A required snapshot array is missing.",
            {"file": name},
        ) from exc
    try:
        array = np.load(io.BytesIO(data), allow_pickle=False)
    except (ValueError, OSError) as exc:
        raise BakerError(
            "E_SNAPSHOT_STATE",
            "A required snapshot array is corrupted.",
            {"file": name, "reason": str(exc)},
        ) from exc
    return np.asarray(array)


def _execute_parse(project: ProjectSpec) -> tuple[Dict[str, Any], Dict[str, bytes]]:
    data_files: Dict[str, bytes] = {}
    dialogue_meta: List[Dict[str, Any]] = []
    for index, dialogue in enumerate(project.dialogues):
        decoded = decode_dialogue(dialogue.audio_path, dialogue.end_sample - dialogue.start_sample)
        source_file = f"line_{index:03d}_source.npy"
        data_files[source_file] = _npy_bytes(decoded.samples)
        dialogue_meta.append(
            {
                "id": dialogue.id,
                "speaker": dialogue.speaker,
                "start_frame": dialogue.start_frame,
                "end_frame": dialogue.end_frame,
                "start_sample": dialogue.start_sample,
                "end_sample": dialogue.end_sample,
                "text": dialogue.text,
                "audio": {
                    "path": dialogue.audio_logical,
                    "sha256": dialogue.audio_sha256,
                    "source_sample_rate": decoded.source_rate,
                    "source_channels": decoded.source_channels,
                    "source_frames": decoded.source_frames,
                    "resampled": decoded.resampled,
                    "resampled_frames_raw": decoded.resampled_frames_raw,
                    "resampled_frames_expected": decoded.resampled_frames_expected,
                    "resample_length_error_raw_samples": decoded.resample_length_error_raw,
                    "resample_length_error_final_samples": 0,
                    "aligned_frames": decoded.target_frames,
                    "timeline_length_adjust_samples": decoded.length_adjust_samples,
                    "source_file": source_file,
                },
            }
        )
    track_meta: List[Dict[str, Any]] = []
    for index, speaker in enumerate(sorted(project.tracks)):
        track = project.tracks[speaker]
        position_file = f"track_{index:03d}_position.npy"
        forward_file = f"track_{index:03d}_forward.npy"
        data_files[position_file] = _npy_bytes(track.positions)
        data_files[forward_file] = _npy_bytes(track.forwards)
        track_meta.append(
            {
                "speaker": speaker,
                "frame_count": track.frame_count,
                "positions_file": position_file,
                "forwards_file": forward_file,
            }
        )
    meta = {
        "timebase": project.timebase.as_report(project.duration_frames),
        "config": {
            "distance_reference_m": project.distance_reference_m,
            "fade_ms": project.fade_ms,
            "limiter": project.limiter,
            "pan_strength": project.pan_strength,
            "sample_rate": project.sample_rate,
            "speed_of_sound_mps": project.speed_of_sound_mps,
            "target_lufs": project.target_lufs,
            "true_peak_dbtp": project.true_peak_dbtp,
        },
        "camera": project.camera,
        "dialogues": dialogue_meta,
        "tracks": track_meta,
    }
    return meta, data_files


def _execute_spatialize(
    project: ProjectSpec,
    parse_artifact: Artifact,
) -> tuple[Dict[str, Any], Dict[str, bytes]]:
    sources: Dict[str, np.ndarray] = {}
    parse_dialogues = {item["id"]: item for item in parse_artifact.meta["dialogues"]}
    for index, dialogue in enumerate(project.dialogues):
        source_file = parse_dialogues[dialogue.id]["audio"]["source_file"]
        sources[dialogue.id] = _load_npy(parse_artifact.files, source_file)
    spatial_lines = spatialize_all(project, sources)
    data_files: Dict[str, bytes] = {}
    dialogue_meta: List[Dict[str, Any]] = []
    for index, (dialogue, result) in enumerate(zip(project.dialogues, spatial_lines)):
        stereo_file = f"line_{index:03d}_stereo.npy"
        frame_file = f"line_{index:03d}_frames.npy"
        data_files[stereo_file] = _npy_bytes(result.stereo)
        data_files[frame_file] = _npy_bytes(result.frame_table)
        dialogue_meta.append(
            {
                "id": dialogue.id,
                "stereo_file": stereo_file,
                "frame_table_file": frame_file,
                "spatial": result.diagnostics,
            }
        )
    meta = {
        "dialogues": dialogue_meta,
        "spatial_method": {
            "distance_model": "reference_distance / max(distance, reference_distance)",
            "pan_model": "equal_power_from_camera_right_projection",
            "orientation_model": "0.65 + 0.35 * max(0, dot(speaker_forward, to_camera))",
            "segmentation": "one gain knot per frame boundary with linear sample interpolation",
            "doppler": "radial velocity integrated per sample then duration-normalized to the exact line sample span",
            "smoothing": "half-cosine fades at line boundaries",
        },
    }
    return meta, data_files


def _overlap_pairs(project: ProjectSpec) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for first_index, first in enumerate(project.dialogues):
        for second in project.dialogues[first_index + 1 :]:
            overlap_start = max(first.start_sample, second.start_sample)
            overlap_end = min(first.end_sample, second.end_sample)
            if overlap_start < overlap_end:
                pairs.append(
                    {
                        "first_id": first.id,
                        "second_id": second.id,
                        "overlap_start_sample": overlap_start,
                        "overlap_end_sample": overlap_end,
                        "overlap_samples": overlap_end - overlap_start,
                    }
                )
    return sorted(pairs, key=lambda item: (item["overlap_start_sample"], item["first_id"], item["second_id"]))


def _execute_mix(
    project: ProjectSpec,
    spatial_artifact: Artifact,
) -> tuple[Dict[str, Any], Dict[str, bytes]]:
    spatial_by_id = {item["id"]: item for item in spatial_artifact.meta["dialogues"]}
    total_samples = project.total_samples
    mix = np.zeros((total_samples, 2), dtype=np.float64)
    stems: Dict[str, np.ndarray] = {}
    for index, dialogue in enumerate(project.dialogues):
        item = spatial_by_id[dialogue.id]
        stereo = _load_npy(spatial_artifact.files, item["stereo_file"])
        expected = dialogue.end_sample - dialogue.start_sample
        if stereo.shape != (expected, 2):
            raise BakerError(
                "E_SNAPSHOT_STATE",
                "Spatialized dialogue shape does not match its sample interval.",
                {
                    "id": dialogue.id,
                    "shape": list(stereo.shape),
                    "expected": [expected, 2],
                },
            )
        mix[dialogue.start_sample : dialogue.end_sample] += stereo
        stems[dialogue.id] = stereo
    mix_clipped = clipped_sample_count(mix, 1.0)
    mix_peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    data_files: Dict[str, bytes] = {
        "mix_prelimiter.npy": _npy_bytes(mix),
    }
    dialogue_meta: List[Dict[str, Any]] = []
    for index, dialogue in enumerate(project.dialogues):
        stereo = stems[dialogue.id]
        line_peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
        line_clipped = clipped_sample_count(stereo, 1.0)
        segment_mix = mix[dialogue.start_sample : dialogue.end_sample]
        contribution_clipped = int(
            np.count_nonzero(
                (np.max(np.abs(segment_mix), axis=1) > 1.0)
                & (np.max(np.abs(stereo), axis=1) > 0.0)
            )
        )
        stem_file = f"line_{index:03d}_stem.npy"
        data_files[stem_file] = _npy_bytes(stereo)
        dialogue_meta.append(
            {
                "id": dialogue.id,
                "stem_file": stem_file,
                "pre_peak_linear": line_peak,
                "pre_peak_dbfs": linear_to_db(line_peak),
                "pre_clipped_sample_count": line_clipped,
                "mix_clipping_contribution_count": contribution_clipped,
            }
        )
    overlap_pairs = _overlap_pairs(project)
    meta = {
        "dialogues": dialogue_meta,
        "mix": {
            "line_count": len(project.dialogues),
            "sample_count": total_samples,
            "pre_peak_linear": mix_peak,
            "pre_peak_dbfs": linear_to_db(mix_peak),
            "pre_clipped_sample_count": mix_clipped,
            "overlap_pair_count": len(overlap_pairs),
            "overlap_pairs": overlap_pairs,
        },
    }
    return meta, data_files


def _execute_loudness(
    project: ProjectSpec,
    mix_artifact: Artifact,
) -> tuple[Dict[str, Any], Dict[str, bytes]]:
    mix_pre = _load_npy(mix_artifact.files, "mix_prelimiter.npy")
    expected_shape = (project.total_samples, 2)
    if mix_pre.shape != expected_shape:
        raise BakerError(
            "E_SNAPSHOT_STATE",
            "Mix snapshot shape does not match the project timeline.",
            {"shape": list(mix_pre.shape), "expected": list(expected_shape)},
        )
    mix_by_id = {item["id"]: item for item in mix_artifact.meta["dialogues"]}
    stems_pre: Dict[str, np.ndarray] = {}
    for dialogue in project.dialogues:
        stems_pre[dialogue.id] = _load_npy(
            mix_artifact.files, mix_by_id[dialogue.id]["stem_file"]
        )

    pre_mix_measure = measure_integrated_loudness(mix_pre, project.sample_rate)
    pre_mix_true_peak = true_peak_dbtp(mix_pre, int(project.limiter["true_peak_oversample"]))
    pre_mix_lufs = pre_mix_measure["integrated_lufs"]
    if pre_mix_lufs is None:
        requested_gain_db = 0.0
    else:
        requested_gain_db = project.target_lufs - float(pre_mix_lufs)
    linear_gain = db_to_linear(requested_gain_db)
    if not math.isfinite(linear_gain) or linear_gain <= 0:
        raise BakerError(
            "E_LOUDNESS_GAIN",
            "Computed loudness gain is not finite.",
            {"requested_gain_db": requested_gain_db},
        )
    mix_gained = mix_pre * linear_gain
    stems_gained = {key: value * linear_gain for key, value in stems_pre.items()}
    gained_mix_clipped = clipped_sample_count(mix_gained, 1.0)
    gained_true_peak = true_peak_dbtp(
        mix_gained, int(project.limiter["true_peak_oversample"])
    )
    limiter_envelope, limiter_stats = lookahead_limiter_gain(
        mix_gained,
        project.sample_rate,
        project.true_peak_dbtp,
        project.limiter["lookahead_ms"],
        project.limiter["release_ms"],
    )
    mix_limited = mix_gained * limiter_envelope[:, None]
    stems_limited = {
        dialogue.id: stems_gained[dialogue.id]
        * limiter_envelope[dialogue.start_sample : dialogue.end_sample, None]
        for dialogue in project.dialogues
    }
    post_limiter_true_peak = true_peak_dbtp(
        mix_limited, int(project.limiter["true_peak_oversample"])
    )
    safety_ceiling = project.true_peak_dbtp - TRUE_PEAK_SAFETY_DB
    final_trim_db = 0.0
    if (
        post_limiter_true_peak is not None
        and post_limiter_true_peak > safety_ceiling
    ):
        final_trim_db = safety_ceiling - post_limiter_true_peak
    final_trim_linear = db_to_linear(final_trim_db)
    mix_final = mix_limited * final_trim_linear
    stems_final = {
        key: value * final_trim_linear for key, value in stems_limited.items()
    }
    final_mix_measure = measure_integrated_loudness(mix_final, project.sample_rate)
    final_true_peak = true_peak_dbtp(
        mix_final, int(project.limiter["true_peak_oversample"])
    )
    final_mix_peak = float(np.max(np.abs(mix_final))) if mix_final.size else 0.0
    final_mix_clipped = clipped_sample_count(mix_final, 1.0)
    final_lufs = final_mix_measure["integrated_lufs"]
    data_files: Dict[str, bytes] = {
        "mix_final.npy": _npy_bytes(mix_final),
    }
    dialogue_meta: List[Dict[str, Any]] = []
    for index, dialogue in enumerate(project.dialogues):
        before_stem = stems_pre[dialogue.id]
        after_stem = stems_final[dialogue.id]
        gained_stem = stems_gained[dialogue.id]
        before_measure = measure_integrated_loudness(before_stem, project.sample_rate)
        after_measure = measure_integrated_loudness(after_stem, project.sample_rate)
        after_peak = float(np.max(np.abs(after_stem))) if after_stem.size else 0.0
        gained_peak = float(np.max(np.abs(gained_stem))) if gained_stem.size else 0.0
        gained_segment = mix_gained[
            dialogue.start_sample : dialogue.end_sample
        ]
        gained_contribution_clipped = int(
            np.count_nonzero(
                (np.max(np.abs(gained_segment), axis=1) > 1.0)
                & (np.max(np.abs(gained_stem), axis=1) > 0.0)
            )
        )
        final_file = f"line_{index:03d}_final.npy"
        data_files[final_file] = _npy_bytes(after_stem)
        dialogue_meta.append(
            {
                "id": dialogue.id,
                "final_file": final_file,
                "before_integrated_lufs": before_measure["integrated_lufs"],
                "after_integrated_lufs": after_measure["integrated_lufs"],
                "post_gain_pre_limiter_peak_linear": gained_peak,
                "post_gain_pre_limiter_peak_dbfs": linear_to_db(gained_peak),
                "post_gain_pre_limiter_clipped_sample_count": clipped_sample_count(
                    gained_stem, 1.0
                ),
                "post_gain_mix_clipping_contribution_count": (
                    gained_contribution_clipped
                ),
                "post_peak_linear": after_peak,
                "post_peak_dbfs": linear_to_db(after_peak),
                "post_clipped_sample_count": clipped_sample_count(after_stem, 1.0),
            }
        )
    meta = {
        "measurement": loudness_parameters(project.sample_rate),
        "dialogues": dialogue_meta,
        "mix": {
            "before_integrated_lufs": pre_mix_lufs,
            "before_true_peak_dbtp": pre_mix_true_peak,
            "requested_linear_gain_db": requested_gain_db,
            "applied_linear_gain_linear": linear_gain,
            "post_gain_pre_limiter_clipped_sample_count": gained_mix_clipped,
            "gain_applied_true_peak_dbtp": gained_true_peak,
            "after_limiter_true_peak_before_trim_dbtp": post_limiter_true_peak,
            "final_trim_db": final_trim_db,
            "final_trim_linear": final_trim_linear,
            "final_true_peak_dbtp": final_true_peak,
            "final_peak_linear": final_mix_peak,
            "final_peak_dbfs": linear_to_db(final_mix_peak),
            "final_clipped_sample_count": final_mix_clipped,
            "after_integrated_lufs": final_lufs,
            "target_lufs": project.target_lufs,
            "target_error_lu": (
                float(final_lufs) - project.target_lufs
                if final_lufs is not None
                else None
            ),
            "limiter": limiter_stats,
            "true_peak_safety_margin_db": TRUE_PEAK_SAFETY_DB,
            "true_peak_limited": bool(
                limiter_stats["limited_sample_count"] > 0 or final_trim_db < 0
            ),
        },
    }
    return meta, data_files


def _build_subtitles(
    project: ProjectSpec,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    timebase = project.timebase
    frames: List[Dict[str, Any]] = []
    active_by_id: Dict[str, List[int]] = {dialogue.id: [] for dialogue in project.dialogues}
    for frame in range(project.duration_frames):
        frame_start, frame_end = timebase.frame_sample_span(frame)
        active_ids = []
        for dialogue in project.dialogues:
            if dialogue.start_sample < frame_end and dialogue.end_sample > frame_start:
                active_ids.append(dialogue.id)
                active_by_id[dialogue.id].append(frame)
        frames.append(
            {
                "frame": frame,
                "start_sample": frame_start,
                "end_sample": frame_end,
                "active_ids": active_ids,
            }
        )
    alignment_lines: List[Dict[str, Any]] = []
    alignment_ok = True
    for dialogue in project.dialogues:
        active_frames = active_by_id[dialogue.id]
        first_frame = active_frames[0] if active_frames else None
        last_frame = active_frames[-1] if active_frames else None
        expected_start = dialogue.start_frame if active_frames else None
        expected_last = dialogue.end_frame - 1 if active_frames else None
        contiguous = active_frames == list(
            range(dialogue.start_frame, dialogue.end_frame)
        )
        interval_ok = (
            dialogue.start_sample == timebase.frame_to_sample(dialogue.start_frame)
            and dialogue.end_sample == timebase.frame_to_sample(dialogue.end_frame)
        )
        line_ok = bool(contiguous and interval_ok)
        alignment_ok = alignment_ok and line_ok
        alignment_lines.append(
            {
                "id": dialogue.id,
                "declared_start_frame": dialogue.start_frame,
                "declared_end_frame_exclusive": dialogue.end_frame,
                "first_active_frame": first_frame,
                "last_active_frame": last_frame,
                "expected_first_active_frame": expected_start,
                "expected_last_active_frame": expected_last,
                "declared_start_sample": dialogue.start_sample,
                "declared_end_sample_exclusive": dialogue.end_sample,
                "contiguous": contiguous,
                "sample_boundaries_exact": interval_ok,
                "ok": line_ok,
            }
        )
    subtitles = {
        "schema_version": 1,
        "fps": project.fps_text,
        "fps_decimal": float(project.fps),
        "sample_rate": project.sample_rate,
        "frame_count": project.duration_frames,
        "timeline_rule": "active when dialogue_sample_span intersects frame_sample_span using half-open integer intervals",
        "frames": frames,
    }
    alignment = {
        "ok": alignment_ok,
        "timebase_boundary_roundtrip_ok": timebase.validate_roundtrip(
            project.duration_frames
        ),
        "lines": alignment_lines,
    }
    return subtitles, alignment


def _execute_package(
    project: ProjectSpec,
    out_dir: Path,
    loudness_artifact: Artifact,
) -> tuple[Dict[str, Any], Dict[str, bytes]]:
    mix_final = _load_npy(loudness_artifact.files, "mix_final.npy")
    expected_shape = (project.total_samples, 2)
    if mix_final.shape != expected_shape:
        raise BakerError(
            "E_SNAPSHOT_STATE",
            "Final mix snapshot shape does not match the project timeline.",
            {"shape": list(mix_final.shape), "expected": list(expected_shape)},
        )
    subtitles, alignment = _build_subtitles(project)
    subtitle_bytes = pretty_json_bytes(subtitles)
    subtitle_path = out_dir / SUBTITLE_FILE_NAME
    atomic_write(subtitle_path, subtitle_bytes)
    wav_metadata = {
        "bit_depth": 16,
        "channels": 2,
        "duration_frames": project.duration_frames,
        "duration_samples": project.total_samples,
        "fps": project.fps_text,
        "fps_decimal": float(project.fps),
        "input_digest": project.input_digest,
        "loudness_algorithm": "ITU-R_BS.1770-4",
        "sample_rate": project.sample_rate,
        "stage_versions": dict(STAGE_VERSIONS),
        "target_lufs": project.target_lufs,
        "tool": "dialogue_baker",
        "tool_version": __version__,
        "true_peak_dbtp": project.true_peak_dbtp,
    }
    wav_path = out_dir / WAV_FILE_NAME
    wav_write_stats = write_pcm16_wav(wav_path, mix_final, wav_metadata)
    if wav_write_stats["quantization_clipped_sample_count"] > 0:
        raise BakerError(
            "E_OUTPUT_CLIPPING",
            "Final float mix exceeded full scale before PCM quantization.",
            {
                "quantization_clipped_sample_count": wav_write_stats[
                    "quantization_clipped_sample_count"
                ],
                "pre_quantization_peak": wav_write_stats[
                    "pre_quantization_peak"
                ],
            },
        )
    format_validation = ffprobe_format(wav_path)
    encoded_samples, wav_info = read_pcm16_wav(wav_path)
    if wav_info["frame_count"] != project.total_samples:
        raise BakerError(
            "E_OUTPUT_FORMAT",
            "Packaged WAV frame count does not match the integer sample timeline.",
            {
                "actual": wav_info["frame_count"],
                "expected": project.total_samples,
            },
        )
    encoded_true_peak = true_peak_dbtp(
        encoded_samples, int(project.limiter["true_peak_oversample"])
    )
    if (
        encoded_true_peak is not None
        and encoded_true_peak > project.true_peak_dbtp + 1e-6
    ):
        raise BakerError(
            "E_TRUE_PEAK_AFTER_ENCODE",
            "Encoded WAV exceeds the configured true-peak ceiling.",
            {
                "encoded_true_peak_dbtp": encoded_true_peak,
                "ceiling_dbtp": project.true_peak_dbtp,
            },
        )
    encoded_loudness = measure_integrated_loudness(
        encoded_samples, project.sample_rate
    )
    meta = {
        "outputs": {
            "wav_file": WAV_FILE_NAME,
            "wav_sha256": sha256_file(wav_path),
            "wav_write": wav_write_stats,
            "encoded_audio": {
                "channels": wav_info["channels"],
                "sample_rate": wav_info["sample_rate"],
                "sample_width_bytes": wav_info["sample_width_bytes"],
                "frame_count": wav_info["frame_count"],
                "true_peak_dbtp": encoded_true_peak,
                "integrated_lufs": encoded_loudness["integrated_lufs"],
            },
            "subtitles_file": SUBTITLE_FILE_NAME,
            "subtitles_sha256": sha256_bytes(subtitle_bytes),
        },
        "format_validation": format_validation,
        "alignment": alignment,
    }
    return meta, {}


def _build_report(
    project: ProjectSpec,
    artifacts: Mapping[str, Artifact],
    subtitles_sha256: str,
) -> Dict[str, Any]:
    parse_meta = artifacts["parse"].meta
    spatial_meta = artifacts["spatialize"].meta
    mix_meta = artifacts["mix"].meta
    loudness_meta = artifacts["loudness"].meta
    package_meta = artifacts["package"].meta
    spatial_by_id = {item["id"]: item for item in spatial_meta["dialogues"]}
    mix_by_id = {item["id"]: item for item in mix_meta["dialogues"]}
    loudness_by_id = {item["id"]: item for item in loudness_meta["dialogues"]}
    dialogues_report: List[Dict[str, Any]] = []
    for item in parse_meta["dialogues"]:
        identifier = item["id"]
        spatial_item = spatial_by_id[identifier]
        mix_item = mix_by_id[identifier]
        loudness_item = loudness_by_id[identifier]
        dialogues_report.append(
            {
                "id": identifier,
                "speaker": item["speaker"],
                "start_frame": item["start_frame"],
                "end_frame": item["end_frame"],
                "start_sample": item["start_sample"],
                "end_sample": item["end_sample"],
                "text": item["text"],
                "source": item["audio"],
                "spatial": spatial_item["spatial"],
                "mix": {
                    "pre_peak_linear": mix_item["pre_peak_linear"],
                    "pre_peak_dbfs": mix_item["pre_peak_dbfs"],
                    "pre_clipped_sample_count": mix_item[
                        "pre_clipped_sample_count"
                    ],
                    "mix_clipping_contribution_count": mix_item[
                        "mix_clipping_contribution_count"
                    ],
                    "post_peak_linear": loudness_item["post_peak_linear"],
                    "post_peak_dbfs": loudness_item["post_peak_dbfs"],
                    "post_gain_pre_limiter_peak_linear": loudness_item[
                        "post_gain_pre_limiter_peak_linear"
                    ],
                    "post_gain_pre_limiter_peak_dbfs": loudness_item[
                        "post_gain_pre_limiter_peak_dbfs"
                    ],
                    "post_clipped_sample_count": loudness_item[
                        "post_clipped_sample_count"
                    ],
                    "post_gain_pre_limiter_clipped_sample_count": loudness_item[
                        "post_gain_pre_limiter_clipped_sample_count"
                    ],
                    "post_gain_mix_clipping_contribution_count": loudness_item[
                        "post_gain_mix_clipping_contribution_count"
                    ],
                },
                "loudness": {
                    "before_integrated_lufs": loudness_item[
                        "before_integrated_lufs"
                    ],
                    "after_integrated_lufs": loudness_item[
                        "after_integrated_lufs"
                    ],
                },
            }
        )
    stages_report = [
        {
            "stage": stage,
            "version": STAGE_VERSIONS[stage],
            "input_digest": artifacts[stage].input_digest,
            "snapshot_digest": artifacts[stage].snapshot_digest,
        }
        for stage in STAGE_ORDER
    ]
    return {
        "schema_version": 1,
        "tool": {
            "name": "dialogue_baker",
            "version": __version__,
        },
        "input_digest": project.input_digest,
        "project": {
            "audio_root_relative": True,
            "duration_frames": project.duration_frames,
            "duration_samples": project.total_samples,
            "fps": project.fps_text,
            "fps_decimal": float(project.fps),
            "sample_rate": project.sample_rate,
            "target_lufs": project.target_lufs,
            "true_peak_dbtp": project.true_peak_dbtp,
        },
        "timebase": parse_meta["timebase"],
        "config": parse_meta["config"],
        "stages": stages_report,
        "dialogues": dialogues_report,
        "mix": dict(mix_meta["mix"]),
        "loudness": {
            "measurement_parameters": loudness_meta["measurement"],
            "before": {
                "integrated_lufs": loudness_meta["mix"]["before_integrated_lufs"],
                "true_peak_dbtp": loudness_meta["mix"][
                    "before_true_peak_dbtp"
                ],
            },
            "after": {
                "integrated_lufs": loudness_meta["mix"]["after_integrated_lufs"],
                "true_peak_dbtp": loudness_meta["mix"]["final_true_peak_dbtp"],
                "peak_linear": loudness_meta["mix"]["final_peak_linear"],
                "peak_dbfs": loudness_meta["mix"]["final_peak_dbfs"],
                "clipped_sample_count": loudness_meta["mix"][
                    "final_clipped_sample_count"
                ],
            },
            "processing": {
                key: value
                for key, value in loudness_meta["mix"].items()
                if key
                not in {
                    "before_integrated_lufs",
                    "before_true_peak_dbtp",
                    "after_integrated_lufs",
                    "final_true_peak_dbtp",
                    "final_peak_linear",
                    "final_peak_dbfs",
                    "final_clipped_sample_count",
                }
            },
            "post_gain_pre_limiter_clipped_sample_count": loudness_meta["mix"][
                "post_gain_pre_limiter_clipped_sample_count"
            ],
        },
        "subtitles": {
            "file": package_meta["outputs"]["subtitles_file"],
            "sha256": subtitles_sha256,
            "frame_count": project.duration_frames,
            "alignment": package_meta["alignment"],
        },
        "output": {
            "wav_file": package_meta["outputs"]["wav_file"],
            "sha256": package_meta["outputs"]["wav_sha256"],
            "format_validation": package_meta["format_validation"],
            "encoded_audio": package_meta["outputs"]["encoded_audio"],
            "wav_metadata": {
                "fps": project.fps_text,
                "sample_rate": project.sample_rate,
                "target_lufs": project.target_lufs,
                "true_peak_dbtp": project.true_peak_dbtp,
            },
        },
    }


def _package_outputs_valid(out_dir: Path, meta: Mapping[str, Any]) -> bool:
    try:
        outputs = meta["outputs"]
        wav_path = out_dir / outputs["wav_file"]
        subtitle_path = out_dir / outputs["subtitles_file"]
        return (
            wav_path.is_file()
            and subtitle_path.is_file()
            and sha256_file(wav_path) == outputs["wav_sha256"]
            and sha256_bytes(subtitle_path.read_bytes())
            == outputs["subtitles_sha256"]
        )
    except (KeyError, OSError, TypeError):
        return False


def _ensure_stage(
    store: SnapshotStore,
    stage: str,
    upstream_digest: str,
    global_input_digest: str,
    force: bool,
    executor: Callable[[], tuple[Dict[str, Any], Dict[str, bytes]]],
    validator: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Artifact:
    expected_digest = store.expected_input_digest(
        stage, global_input_digest, upstream_digest
    )
    if not force:
        loaded, _reason = store.load(stage, expected_digest)
        if loaded is not None:
            if validator is None or validator(loaded.meta):
                return Artifact(
                    stage=stage,
                    input_digest=expected_digest,
                    snapshot_digest=loaded.snapshot_digest,
                    meta=loaded.meta,
                    files=loaded.files,
                    reused=True,
                )
    meta, data_files = executor()
    snapshot_digest = store.save(stage, expected_digest, meta, data_files)
    return Artifact(
        stage=stage,
        input_digest=expected_digest,
        snapshot_digest=snapshot_digest,
        meta=meta,
        files=dict(data_files),
        reused=False,
    )


def run_bake(
    project_path: Path,
    out_dir: Path,
    stop_after: Optional[str] = None,
    force: bool = False,
) -> BakeResult:
    if stop_after is not None and stop_after not in STAGE_ORDER:
        raise BakerError(
            "E_INVALID_STAGE",
            "Unknown stop-after stage.",
            {"stage": stop_after, "valid_stages": list(STAGE_ORDER)},
        )
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    store = SnapshotStore(out_dir)
    if force:
        shutil.rmtree(store.root, ignore_errors=True)
    project = load_project(Path(project_path))
    artifacts: Dict[str, Artifact] = {}
    stage_actions: Dict[str, str] = {}
    upstream = ""
    executors: Dict[
        str,
        Callable[[], tuple[Dict[str, Any], Dict[str, bytes]]],
    ] = {
        "parse": lambda: _execute_parse(project),
        "spatialize": lambda: _execute_spatialize(project, artifacts["parse"]),
        "mix": lambda: _execute_mix(project, artifacts["spatialize"]),
        "loudness": lambda: _execute_loudness(project, artifacts["mix"]),
        "package": lambda: _execute_package(project, out_dir, artifacts["loudness"]),
    }
    for stage in STAGE_ORDER:
        validator = (
            (lambda meta: _package_outputs_valid(out_dir, meta))
            if stage == "package"
            else None
        )
        artifact = _ensure_stage(
            store=store,
            stage=stage,
            upstream_digest=upstream,
            global_input_digest=project.input_digest,
            force=force,
            executor=executors[stage],
            validator=validator,
        )
        artifacts[stage] = artifact
        stage_actions[stage] = "reused" if artifact.reused else "executed"
        upstream = artifact.snapshot_digest
        if stop_after == stage:
            return BakeResult(
                status="stopped",
                completed_stage=stage,
                report=None,
                report_path=None,
                stage_actions=stage_actions,
            )

    report = _build_report(
        project,
        artifacts,
        artifacts["package"].meta["outputs"]["subtitles_sha256"],
    )
    report_path = out_dir / REPORT_FILE_NAME
    atomic_write(report_path, pretty_json_bytes(report))
    error_path = out_dir / "bake_error.json"
    if error_path.exists():
        error_path.unlink()
    return BakeResult(
        status="complete",
        completed_stage="package",
        report=report,
        report_path=report_path,
        stage_actions=stage_actions,
    )


def write_error_report(out_dir: Path, error: BakerError) -> Optional[Path]:
    try:
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "bake_error.json"
        atomic_write(path, pretty_json_bytes(json_safe(error.as_dict())))
        return path
    except OSError:
        return None

