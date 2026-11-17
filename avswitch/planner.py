"""ABR switch planner.

Simulates playback in fixed-size chunks against the bandwidth curve from the
manifest. A chunk may switch up only when the target bitrate fits the
effective bandwidth (bandwidth * (1 - margin) - audio) AND the current
buffer can absorb the download time plus the switch overhead. Down-switches
happen immediately; up-switches require a minimum dwell time since the last
switch to avoid flapping.
"""


def bandwidth_at(curve, t):
    bw = curve[0]["bandwidth_kbps"]
    for point in curve:
        if point["time"] <= t + 1e-9:
            bw = point["bandwidth_kbps"]
        else:
            break
    return bw


def plan(video_streams, audio_kbps, curve, policy, duration):
    """Return (events, underruns, buffer_trace).

    events: list of {"time", "from", "to", "reason"} at chunk boundaries.
    """
    chunk = float(policy.get("segment_seconds", 2.0))
    margin = float(policy.get("buffer_margin", 0.15))
    overhead = float(policy.get("switch_overhead_seconds", 0.5))
    min_dwell = float(policy.get("min_dwell_seconds", 4.0))
    capacity = float(policy.get("buffer_capacity_seconds", 10.0))
    buffer_level = float(policy.get("initial_buffer_seconds", 2.0))

    ordered = sorted(video_streams, key=lambda s: s["bitrate_kbps"])
    current = ordered[0]
    events = [{"time": 0.0, "from": None, "to": current["id"], "reason": "startup_lowest"}]
    underruns = []
    buffer_trace = []
    last_switch = 0.0
    t = 0.0
    while t < duration - 1e-9:
        bw = max(bandwidth_at(curve, t), 1e-9)
        effective = bw * (1.0 - margin) - audio_kbps
        sustainable = max(0.0, (buffer_level + chunk - overhead) * bw / chunk)
        allowed = min(effective, sustainable)
        candidate = ordered[0]
        for stream in ordered:
            if stream["bitrate_kbps"] <= allowed + 1e-9:
                candidate = stream
        switched = False
        reason = None
        if candidate["bitrate_kbps"] < current["bitrate_kbps"] and candidate["id"] != current["id"]:
            switched, reason = True, "bandwidth_drop"
        elif candidate["id"] != current["id"] and (t - last_switch) >= min_dwell - 1e-9:
            switched, reason = True, "bandwidth_headroom"
        if switched:
            events.append({"time": t, "from": current["id"], "to": candidate["id"], "reason": reason})
            current = candidate
            last_switch = t
        cost = chunk * current["bitrate_kbps"] / bw + (overhead if switched else 0.0)
        buffer_level = min(capacity, buffer_level + chunk - cost)
        if buffer_level <= 0.0:
            underruns.append(round(t + chunk, 6))
            buffer_level = 0.0
            lowest = ordered[0]
            if current["id"] != lowest["id"]:
                events.append({
                    "time": t + chunk, "from": current["id"], "to": lowest["id"],
                    "reason": "buffer_underrun",
                })
                current = lowest
                last_switch = t + chunk
        buffer_trace.append({"time": round(t, 6), "buffer_seconds": round(buffer_level, 6),
                             "stream": current["id"], "bandwidth_kbps": bw})
        t += chunk
    return events, underruns, buffer_trace
