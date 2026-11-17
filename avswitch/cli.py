"""Command line entry: manifest -> adaptive switch plan -> MPEG-TS output + report."""
import argparse
import json
import sys
from pathlib import Path

from . import gop, muxer, planner, probe, verify


def _load_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    base = manifest_path.resolve().parent
    for stream in manifest["video_streams"]:
        stream["path"] = str((base / stream["path"]).resolve())
    manifest["audio_stream"]["path"] = str((base / manifest["audio_stream"]["path"]).resolve())
    return manifest


def _probe_video_streams(manifest, report):
    usable = []
    for stream in manifest["video_streams"]:
        try:
            streams = probe.probe_streams(stream["path"])
            v_idx = probe.stream_index_by_type(streams, "video")
            packets = probe.probe_packets(stream["path"], v_idx)
            keyframes = gop.keyframe_times(packets)
            if not packets or not keyframes:
                raise probe.ProbeError("no video packets / keyframes")
            duration = packets[-1]["pts"] + packets[-1]["duration"]
            stream["keyframes"] = keyframes
            stream["duration"] = duration
            usable.append(stream)
        except Exception as exc:
            report["unusable_streams"].append({
                "stream": stream["id"], "path": stream["path"], "reason": str(exc),
            })
    if not usable:
        raise muxer.MuxError("no usable video streams")
    return usable


def run(manifest_path, output_path, report_path=None, workdir=None):
    manifest = _load_manifest(manifest_path)
    output_path = Path(output_path)
    report = {
        "manifest": str(manifest_path),
        "output": str(output_path),
        "switches": [],
        "fallbacks": [],
        "timestamp_adjustments": [],
        "unusable_streams": [],
        "buffer_underruns": [],
    }

    usable = _probe_video_streams(manifest, report)
    audio = manifest["audio_stream"]
    audio_streams = probe.probe_streams(audio["path"])
    a_idx = probe.stream_index_by_type(audio_streams, "audio")
    audio_packets = probe.probe_packets(audio["path"], a_idx)
    audio_duration = audio_packets[-1]["pts"] + audio_packets[-1]["duration"]

    duration = min(max(s["duration"] for s in usable), audio_duration)
    keyframes_by_id = {s["id"]: s["keyframes"] for s in usable}
    streams_by_id = {s["id"]: s for s in usable}

    events, underruns, buffer_trace = planner.plan(
        usable, audio.get("bitrate_kbps", 0),
        manifest["bandwidth_curve"], manifest.get("switch_policy", {}), duration)
    report["buffer_underruns"] = underruns
    report["switches"] = [e for e in events if e["from"] is not None]

    segments = muxer.build_segments(events, duration, keyframes_by_id,
                                    report["timestamp_adjustments"])
    segments = muxer.apply_fallbacks(segments, streams_by_id, keyframes_by_id, duration,
                                     report["fallbacks"], report["timestamp_adjustments"])

    details, ffmpeg_stderr = muxer.mux(segments, streams_by_id, audio["path"],
                                       output_path, workdir=workdir)
    non_monotonic = ffmpeg_stderr.count("Non-monoton")
    if non_monotonic:
        report["timestamp_adjustments"].append({
            "type": "dts_fix",
            "count": non_monotonic,
            "note": "ffmpeg corrected non-monotonic DTS during final mux",
        })
    report["segments"] = details
    report["buffer_trace"] = buffer_trace

    switch_times = [s["src_in"] for s in segments[1:]]
    report["verification"] = verify.verify_output(output_path, switch_times)

    if report_path is None:
        report_path = str(output_path) + ".report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    report["report_path"] = str(report_path)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="avswitch",
        description="Adaptive multi-bitrate video switcher + AAC audio MPEG-TS muxer")
    parser.add_argument("manifest", help="path to the JSON manifest")
    parser.add_argument("output", help="output MPEG-TS file path")
    parser.add_argument("--report", help="report JSON path (default: <output>.report.json)")
    parser.add_argument("--workdir", help="directory for intermediate segment files")
    args = parser.parse_args(argv)

    report = run(args.manifest, args.output, report_path=args.report, workdir=args.workdir)
    ver = report["verification"]
    print(f"output:            {report['output']}")
    print(f"report:            {report['report_path']}")
    print(f"switches:          {len(report['switches'])}")
    print(f"fallbacks:         {len(report['fallbacks'])}")
    print(f"video frames:      {ver['video_frames']}")
    print(f"DTS monotonic:     {ver['video_dts_monotonic']}")
    print(f"max A/V sync diff: {ver['sync']['max_abs_ms']} ms")
    print(f"sync within 20ms:  {ver['sync']['within_tolerance']}")
    print(f"PES violations:    {len(ver['ts_structure']['pes_length_violations'])}")
    ok = (ver["video_dts_monotonic"] and ver["sync"]["within_tolerance"]
          and ver["continuity"]["continuous"]
          and not ver["ts_structure"]["pes_length_violations"])
    print(f"validation:        {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
