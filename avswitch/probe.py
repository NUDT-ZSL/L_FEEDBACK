"""ffprobe/ffmpeg wrappers used across the pipeline."""
import json
import subprocess

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


class ProbeError(RuntimeError):
    pass


def _run_json(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise ProbeError(f"command failed: {' '.join(str(c) for c in cmd)}\n{proc.stderr.strip()}")
    return json.loads(proc.stdout)


def probe_streams(path):
    data = _run_json([FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(path)])
    return data.get("streams", [])


def probe_format(path):
    data = _run_json([FFPROBE, "-v", "error", "-show_format", "-of", "json", str(path)])
    return data.get("format", {})


def probe_packets(path, stream_index):
    data = _run_json([
        FFPROBE, "-v", "error", "-select_streams", str(stream_index),
        "-show_packets", "-of", "json", str(path),
    ])
    packets = []
    for pkt in data.get("packets", []):
        packets.append({
            "pts": float(pkt["pts_time"]),
            "dts": float(pkt["dts_time"]),
            "duration": float(pkt.get("duration_time", 0.0)),
            "keyframe": "K" in pkt.get("flags", ""),
            "size": int(pkt.get("size", 0)),
        })
    return packets


def stream_index_by_type(streams, codec_type):
    for stream in streams:
        if stream.get("codec_type") == codec_type:
            return stream["index"]
    raise ProbeError(f"no {codec_type} stream found")


def decode_check(path, start=None, duration=None):
    """Decode a (sub-)range of a file. Returns (ok, stderr)."""
    cmd = [FFMPEG, "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    cmd += ["-f", "null", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ok = proc.returncode == 0 and not proc.stderr.strip()
    return ok, proc.stderr.strip()
