"""End-to-end processing pipeline orchestrating the loop build and mix."""

import json
import logging
import math
import os

import numpy as np
import soundfile as sf

from .audioio import load_and_validate_track, to_sample_index
from .crossfade import build_loop
from .errors import OverwriteError, ParameterRangeError

logger = logging.getLogger("seamless_loop")


def _to_stereo(samples):
    """Return a stereo ``(frames, 2)`` array, duplicating mono channels."""
    if samples.ndim == 1:
        return np.repeat(samples[:, np.newaxis], 2, axis=1)
    return samples


def process_project(project):
    """Run the full pipeline for a validated ``project`` dictionary.

    Returns the path to the written WAV file.
    """
    sample_rate = _resolve_sample_rate(project)

    start_sample = to_sample_index(project["loop_start"], sample_rate)
    end_sample = to_sample_index(project["loop_end"], sample_rate)
    fade_len = to_sample_index(project["crossfade_duration"], sample_rate)
    loop_len = end_sample - start_sample

    if loop_len <= 0:
        raise ParameterRangeError(
            "loop length rounds to %d sample(s); the loop region must contain "
            "at least one sample" % loop_len
        )
    if fade_len < 1:
        raise ParameterRangeError(
            "crossfade duration rounds to %d sample(s); it must be at least "
            "one sample long" % fade_len
        )
    if loop_len < 2:
        raise ParameterRangeError(
            "loop length %d sample(s) cannot hold even a 1-sample crossfade "
            "per track" % loop_len
        )
    if loop_len < 2 * fade_len:
        # The project-level region is shorter than two configured fades.
        # Per-track build_loop dynamically shortens each fade to fit and logs
        # the adjustment, as required; surface an info line at project level.
        logger.info(
            "loop length %d samples is below twice the configured crossfade "
            "(%d); per-track crossfades will be shortened to fit",
            loop_len, fade_len,
        )

    logger.info(
        "loop region: samples [%d, %d) = %d samples; crossfade = %d samples",
        start_sample, end_sample, loop_len, fade_len,
    )

    tracks = _load_tracks(project, start_sample, end_sample, sample_rate)
    built = _build_track_loops(tracks, fade_len, loop_len)

    common_loop_len = built[0]["loop"].shape[0]

    total_samples, repeats = _plan_repetition(
        project, sample_rate, common_loop_len
    )

    mixed = _mix_tracks(built, common_loop_len, total_samples)
    mix_peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    output_gain = 1.0
    if mix_peak > 1.0:
        output_gain = 1.0 / mix_peak
        mixed = mixed * output_gain
        logger.warning(
            "mix peak %.4f exceeded full scale; applied gain %.6f to avoid "
            "clipping", mix_peak, output_gain,
        )

    output_path, report_path = _resolve_output_paths(project)
    _guard_overwrite(project, output_path, report_path)

    sf.write(output_path, mixed.astype(np.float32), sample_rate, subtype="FLOAT")
    logger.info("wrote stereo mix: %s (%d samples)", output_path, total_samples)

    report = _build_report(
        project=project,
        sample_rate=sample_rate,
        start_sample=start_sample,
        end_sample=end_sample,
        configured_fade=fade_len,
        common_loop_len=common_loop_len,
        total_samples=total_samples,
        repeats=repeats,
        output_gain=output_gain,
        output_path=output_path,
        report_path=report_path,
        built=built,
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logger.info("wrote processing report: %s", report_path)
    return output_path


def _resolve_sample_rate(project):
    """Derive the shared sample rate from the first input track."""
    import soundfile as _sf

    first = project["input_files"][0]
    if not os.path.isfile(first):
        from .errors import AudioFileError

        raise AudioFileError("input audio file not found: %s" % first)
    try:
        return int(_sf.info(first).samplerate)
    except Exception as exc:
        from .errors import AudioFileError

        raise AudioFileError(
            "cannot read audio file %s: %s" % (first, exc)
        ) from exc


def _load_tracks(project, start_sample, end_sample, sample_rate):
    tracks = []
    expected_sr = sample_rate
    for index, path in enumerate(project["input_files"]):
        track = load_and_validate_track(
            path, start_sample, end_sample, expected_sr=expected_sr
        )
        track["index"] = index
        tracks.append(track)
        logger.info(
            "loaded track %d: %s (%d ch, %s, %d Hz, %d frames)",
            index, path, track["channels"], track["subtype"],
            track["sample_rate"], track["total_frames"],
        )
    return tracks


def _build_track_loops(tracks, fade_len, loop_len):
    built = []
    reference_length = None
    for track in tracks:
        loop, effective_fade = build_loop(
            track["region"], fade_len, track_name=track["path"]
        )
        if reference_length is None:
            reference_length = loop.shape[0]
        elif loop.shape[0] != reference_length:
            raise ParameterRangeError(
                "track %r produced a %d-sample loop but the reference loop is "
                "%d samples; tracks must stay sample aligned"
                % (track["path"], loop.shape[0], reference_length)
            )
        built.append(
            {
                "index": track["index"],
                "path": track["path"],
                "source_channels": track["channels"],
                "subtype": track["subtype"],
                "loop": loop,
                "configured_fade": fade_len,
                "effective_fade": effective_fade,
                "fade_adjusted": effective_fade != fade_len,
                "loop_length_samples": loop.shape[0],
            }
        )
    return built


def _plan_repetition(project, sample_rate, loop_len):
    if project["target_duration"] is None:
        return loop_len, 1
    total_samples = to_sample_index(project["target_duration"], sample_rate)
    if total_samples < loop_len:
        raise ParameterRangeError(
            "target duration rounds to %d samples, shorter than one loop of "
            "%d samples" % (total_samples, loop_len)
        )
    repeats = int(math.ceil(total_samples / loop_len))
    if total_samples % loop_len != 0:
        logger.warning(
            "target duration (%d samples) is not an integer multiple of the "
            "loop (%d samples); output is truncated to the exact target",
            total_samples, loop_len,
        )
    return total_samples, repeats


def _mix_tracks(built, loop_len, total_samples):
    accumulator = None
    for item in built:
        stereo_loop = _to_stereo(item["loop"])
        repeats = int(math.ceil(total_samples / loop_len))
        tiled = np.tile(stereo_loop, (repeats, 1))[:total_samples]
        accumulator = tiled if accumulator is None else accumulator + tiled
    return accumulator


def _resolve_output_paths(project):
    output_dir = project["output_dir"]
    base_name = "%s_loop.wav" % project["project_name"]
    output_path = os.path.join(output_dir, base_name)
    report_path = os.path.join(
        output_dir, "%s_loop_report.json" % project["project_name"]
    )
    return os.path.abspath(output_path), os.path.abspath(report_path)


def _guard_overwrite(project, output_path, report_path):
    if project["overwrite"]:
        return
    existing = [path for path in (output_path, report_path) if os.path.exists(path)]
    if existing:
        raise OverwriteError(
            "overwrite=false but output file(s) already exist: %s"
            % ", ".join(existing)
        )


def _build_report(
    project,
    sample_rate,
    start_sample,
    end_sample,
    configured_fade,
    common_loop_len,
    total_samples,
    repeats,
    output_gain,
    output_path,
    report_path,
    built,
):
    track_reports = [
        {
            "index": item["index"],
            "input_file": item["path"],
            "source_channels": item["source_channels"],
            "source_subtype": item["subtype"],
            "crossfade_samples_configured": item["configured_fade"],
            "crossfade_samples_used": item["effective_fade"],
            "crossfade_adjusted": item["fade_adjusted"],
            "loop_length_samples": item["loop_length_samples"],
        }
        for item in built
    ]
    return {
        "project_name": project["project_name"],
        "sample_rate": sample_rate,
        "loop_region": {
            "start_seconds": project["loop_start"],
            "end_seconds": project["loop_end"],
            "start_sample": start_sample,
            "end_sample": end_sample,
            "region_length_samples": end_sample - start_sample,
        },
        "crossfade_samples_configured": configured_fade,
        "loop_length_samples": common_loop_len,
        "loop_duration_seconds": common_loop_len / sample_rate,
        "target_duration_seconds": project["target_duration"],
        "loop_repeats": repeats,
        "output_gain": output_gain,
        "output_file": output_path,
        "report_file": report_path,
        "tracks": track_reports,
        "output_summary": {
            "sample_rate": sample_rate,
            "channels": 2,
            "total_samples": total_samples,
            "duration_seconds": total_samples / sample_rate,
            "subtype": "FLOAT",
        },
    }
