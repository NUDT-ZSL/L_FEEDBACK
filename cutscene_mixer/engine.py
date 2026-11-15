"""End-to-end pipeline: descriptor in, mixed WAV and sync report out."""

import json
import os

import numpy as np
import soundfile as sf

from . import mixer
from .audio import load_track_audio
from .errors import AUDIO_LOAD_FAIL
from .schema import validate_descriptor
from .timeline import frame_to_sample, ms_to_samples


def run(descriptor_path, output_path, report_path, target_lufs):
    """Execute the mixing pipeline.

    Returns the process exit code: 0 on full success, 1 when some tracks
    failed to load but the mix was still produced. Schema validation
    failures are raised as SchemaError by ``validate_descriptor``.
    """
    with open(descriptor_path, "r", encoding="utf-8") as handle:
        descriptor = json.load(handle)
    desc = validate_descriptor(descriptor)

    output = desc["output"]
    sample_rate = output["sample_rate"]
    channels = output["channels"]
    frame_rate = desc["frame_rate"]
    limits = desc["gain_limits_db"]
    crossfade_samples = ms_to_samples(desc["crossfade_ms"], sample_rate)
    fade_samples = ms_to_samples(desc["fade_ms"], sample_rate)

    base_dir = os.path.dirname(os.path.abspath(descriptor_path))

    track_buffers = []
    track_reports = []
    load_failures = 0

    for index, track in enumerate(desc["tracks"]):
        track_id = track.get("id") or "track%d" % index
        path = track["path"]
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)

        events_report = [
            {
                "id": event.get("id") or "event%d" % ev_index,
                "frame": event["frame"],
                "track_sample": event["track_sample"],
                "absolute_sample": frame_to_sample(
                    event["frame"], frame_rate, sample_rate
                ),
            }
            for ev_index, event in enumerate(track["events"])
        ]

        report_entry = {
            "id": track_id,
            "path": track["path"],
            "events": events_report,
        }

        try:
            audio = load_track_audio(path, sample_rate, channels)
        except Exception as exc:  # decode/IO failure: skip, keep mixing
            report_entry["status"] = AUDIO_LOAD_FAIL
            report_entry["error"] = str(exc)
            track_reports.append(report_entry)
            load_failures += 1
            continue

        segments = mixer.build_segments(
            track["events"], audio.shape[0], frame_rate, sample_rate
        )
        buffer = mixer.assemble_track(
            audio, segments, crossfade_samples, fade_samples, channels
        )
        track_buffers.append(buffer)
        report_entry["status"] = "ok"
        report_entry["_buffer_index"] = len(track_buffers) - 1
        track_reports.append(report_entry)

    mix, gain_reports = mixer.mix_tracks(track_buffers, limits)
    for entry in track_reports:
        buffer_index = entry.pop("_buffer_index", None)
        if buffer_index is not None:
            entry.update(gain_reports[buffer_index])

    mix, mix_report = mixer.normalize_mix(mix, target_lufs, limits)

    _write_wav(output_path, mix, sample_rate, output["bit_depth"])

    report = {
        "input": os.path.abspath(descriptor_path),
        "output": os.path.abspath(output_path),
        "frame_rate": frame_rate,
        "sample_rate": sample_rate,
        "bit_depth": output["bit_depth"],
        "channels": channels,
        "target_lufs": target_lufs,
        "crossfade_ms": desc["crossfade_ms"],
        "fade_ms": desc["fade_ms"],
        "gain_limits_db": limits,
        "tracks": track_reports,
        "mix": dict(
            duration_samples=int(mix.shape[0]),
            duration_seconds=round(mix.shape[0] / sample_rate, 6),
            **mix_report
        ),
    }
    _write_json(report_path, report)
    return 1 if load_failures else 0


def _write_wav(path, mix, sample_rate, bit_depth):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    subtype = "PCM_16" if bit_depth == 16 else "PCM_24"
    if mix.shape[0] == 0:
        mix = np.zeros((0, 2), dtype=np.float64)
    sf.write(path, mix, sample_rate, subtype=subtype)


def _write_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
