"""Segment building (keyframe snapping + fallback) and MPEG-TS muxing."""
import subprocess
import tempfile
from pathlib import Path

from . import gop, probe

FFMPEG = "ffmpeg"


class MuxError(RuntimeError):
    pass


def build_segments(events, duration, keyframes_by_id, adjustments):
    """Turn planner events into source segments snapped to keyframes.

    Every segment starts exactly on a keyframe of its stream. Snapping a
    planned switch time backwards to a keyframe is recorded as a timeline
    adjustment so the output timeline stays free of gaps and duplicates.
    """
    segments = []
    for index, event in enumerate(events):
        planned_start = event["time"]
        kfs = keyframes_by_id[event["to"]]
        start = gop.floor_keyframe(kfs, planned_start)
        if start is None:
            start = 0.0
        if abs(start - planned_start) > 1e-3:
            adjustments.append({
                "type": "keyframe_snap",
                "stream": event["to"],
                "planned_time": round(planned_start, 6),
                "actual_time": round(start, 6),
                "note": "switch point moved to nearest preceding keyframe",
            })
        if segments and start <= segments[-1]["src_in"] + 1e-6:
            continue
        if segments:
            segments[-1]["src_out"] = start
        segments.append({
            "stream": event["to"],
            "src_in": start,
            "src_out": duration if index == len(events) - 1 else None,
            "event": event,
        })
    segments = [s for s in segments if s["src_out"] is None or s["src_out"] - s["src_in"] > 1e-6]
    for i in range(len(segments) - 1):
        if segments[i]["src_out"] is None:
            segments[i]["src_out"] = segments[i + 1]["src_in"]
    if segments and segments[-1]["src_out"] is None:
        segments[-1]["src_out"] = duration
    return segments


def apply_fallbacks(segments, streams_by_id, keyframes_by_id, duration, fallbacks, adjustments):
    """Decode-check every segment; on failure fall back to the last good stream."""
    usable = []
    last_good = None
    for segment in segments:
        stream = streams_by_id[segment["stream"]]
        ok, stderr = probe.decode_check(
            stream["path"], start=segment["src_in"],
            duration=segment["src_out"] - segment["src_in"])
        if ok:
            last_good = segment["stream"]
            usable.append(segment)
            continue
        candidates = [last_good] if last_good else []
        candidates += [s for s in streams_by_id if s not in candidates and s != segment["stream"]]
        recovered = None
        for candidate_id in candidates:
            if candidate_id is None:
                continue
            kfs = keyframes_by_id[candidate_id]
            fb_in = gop.floor_keyframe(kfs, segment["src_in"])
            if fb_in is None:
                fb_in = 0.0
            fb_out = min(segment["src_out"], duration)
            ok2, _ = probe.decode_check(
                streams_by_id[candidate_id]["path"], start=fb_in, duration=fb_out - fb_in)
            if ok2:
                recovered = dict(segment)
                recovered["stream"] = candidate_id
                recovered["src_in"] = fb_in
                recovered["src_out"] = fb_out
                break
        if recovered is None:
            raise MuxError(f"no usable stream for segment starting at {segment['src_in']}")
        fallbacks.append({
            "time": round(segment["src_in"], 6),
            "target_stream": segment["stream"],
            "fallback_stream": recovered["stream"],
            "reason": stderr.splitlines()[-1] if stderr else "stream unreadable",
        })
        if abs(recovered["src_in"] - segment["src_in"]) > 1e-3:
            adjustments.append({
                "type": "fallback_keyframe_snap",
                "stream": recovered["stream"],
                "planned_time": round(segment["src_in"], 6),
                "actual_time": round(recovered["src_in"], 6),
                "note": "fallback segment start moved to keyframe of fallback stream",
            })
        if usable and recovered["src_in"] < usable[-1]["src_out"] - 1e-6:
            usable[-1]["src_out"] = recovered["src_in"]
        last_good = recovered["stream"]
        usable.append(recovered)
    return usable


def extract_segment(source, start, end, dest):
    duration = end - start
    cmd = [
        FFMPEG, "-y", "-v", "error",
        "-ss", f"{start:.6f}", "-i", str(source),
        "-t", f"{duration:.6f}",
        "-c:v", "copy", "-an", "-sn", "-dn",
        "-f", "mpegts", str(dest),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise MuxError(f"segment extraction failed: {' '.join(cmd)}\n{proc.stderr}")


def mux(segments, streams_by_id, audio_path, output_path, workdir=None):
    """Extract segments, concatenate, mux with audio into MPEG-TS.

    Returns (segment_details, ffmpeg_stderr).
    """
    output_path = Path(output_path)
    details = []
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        tmp = Path(tmp)
        seg_files = []
        for index, segment in enumerate(segments):
            dest = tmp / f"seg_{index:03d}.ts"
            source = streams_by_id[segment["stream"]]["path"]
            extract_segment(source, segment["src_in"], segment["src_out"], dest)
            packets = probe.probe_packets(dest, 0)
            if not packets:
                raise MuxError(f"extracted segment {index} has no packets")
            details.append({
                "index": index,
                "stream": segment["stream"],
                "src_in": round(segment["src_in"], 6),
                "src_out": round(segment["src_out"], 6),
                "frames": len(packets),
                "first_pts": round(packets[0]["pts"], 6),
                "last_pts": round(packets[-1]["pts"], 6),
            })
            seg_files.append(dest)
        concat_file = tmp / "concat.txt"
        concat_file.write_text(
            "".join(f"file '{f}'\n" for f in seg_files), encoding="utf-8")
        cmd = [
            FFMPEG, "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-i", str(audio_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c", "copy",
            "-muxpreload", "0", "-muxdelay", "0",
            "-f", "mpegts", str(output_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise MuxError(f"final mux failed: {' '.join(cmd)}\n{proc.stderr}")
    return details, proc.stderr
