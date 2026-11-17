"""Output verification: ffprobe readability, DTS monotonicity, frame
continuity and audio/video PTS alignment statistics."""
import numpy as np

from . import probe, tsvalidate

SYNC_TOLERANCE_MS = 20.0


def _monotonic(values, eps=1e-6):
    return all(b > a - eps for a, b in zip(values, values[1:]))


def sync_stats(video_pts, audio_pts):
    video = np.asarray(video_pts)
    audio = np.asarray(audio_pts)
    diffs = np.array([np.min(np.abs(audio - v)) for v in video]) if len(audio) else np.array([])
    drift = 0.0
    if len(video) > 1 and len(audio) > 1:
        a_last = float(audio[np.argmin(np.abs(audio - video[-1]))])
        a_first = float(audio[np.argmin(np.abs(audio - video[0]))])
        drift = abs((video[-1] - video[0]) - (a_last - a_first))
    return {
        "max_abs_ms": round(float(np.max(diffs)) * 1000.0, 3) if len(diffs) else None,
        "mean_abs_ms": round(float(np.mean(diffs)) * 1000.0, 3) if len(diffs) else None,
        "p95_abs_ms": round(float(np.percentile(diffs, 95)) * 1000.0, 3) if len(diffs) else None,
        "timeline_drift_ms": round(float(drift) * 1000.0, 3),
        "within_tolerance": bool(len(diffs)) and float(np.max(diffs)) * 1000.0 <= SYNC_TOLERANCE_MS,
    }


def continuity_stats(video_pts, nominal_frame_duration):
    deltas = np.diff(np.asarray(video_pts))
    gaps = []
    duplicates = []
    for index, delta in enumerate(deltas):
        if delta > nominal_frame_duration * 1.5:
            gaps.append({"after_frame": index, "gap_seconds": round(float(delta), 6)})
        elif delta < nominal_frame_duration * 0.5:
            duplicates.append({"at_frame": index + 1, "delta_seconds": round(float(delta), 6)})
    return {
        "frames": len(video_pts),
        "gaps": gaps,
        "duplicates": duplicates,
        "continuous": not gaps and not duplicates,
    }


def verify_output(path, switch_times, nominal_frame_duration=1.0 / 24.0):
    streams = probe.probe_streams(path)
    v_idx = probe.stream_index_by_type(streams, "video")
    a_idx = probe.stream_index_by_type(streams, "audio")
    v_packets = probe.probe_packets(path, v_idx)
    a_packets = probe.probe_packets(path, a_idx)
    video_pts = [p["pts"] for p in v_packets]
    audio_pts = [p["pts"] for p in a_packets]
    video_dts = [p["dts"] for p in v_packets]

    sync = sync_stats(video_pts, audio_pts)
    per_switch = []
    audio_arr = np.asarray(audio_pts)
    for t in switch_times:
        if not video_pts or not len(audio_arr):
            continue
        v_nearest = min(video_pts, key=lambda v: abs(v - t))
        a_nearest = float(np.min(np.abs(audio_arr - v_nearest)))
        per_switch.append({
            "switch_time": round(t, 6),
            "nearest_video_pts": round(v_nearest, 6),
            "abs_av_diff_ms": round(a_nearest * 1000.0, 3),
        })

    ts_info = tsvalidate.validate(path)
    return {
        "streams": [
            {"index": s["index"], "codec_type": s["codec_type"], "codec_name": s["codec_name"]}
            for s in streams
        ],
        "video_frames": len(v_packets),
        "audio_packets": len(a_packets),
        "video_dts_monotonic": _monotonic(video_dts),
        "audio_pts_monotonic": _monotonic(audio_pts),
        "sync": sync,
        "per_switch_sync": per_switch,
        "continuity": continuity_stats(video_pts, nominal_frame_duration),
        "ts_structure": ts_info,
    }
