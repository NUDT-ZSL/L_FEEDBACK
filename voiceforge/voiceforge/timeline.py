"""Structured event timeline construction.

Intervals are rasterized onto the output sample axis.  A sweep-line pass
splits them at every activation boundary so the emitted events never overlap;
simultaneous players are merged into one event carrying a stable, sorted
participant list.  Missing-source players remain visible as marked events
instead of crashing the run.
"""

from .resampling import samples_to_ms


def build_timeline(intervals, missing_players, loudness_map, total_samples: int,
                   output_rate: int):
    missing_set = set(missing_players)
    segments = []
    for interval in intervals:
        start = interval["_start_sample"]
        end = min(interval["_end_sample"], total_samples)
        if end <= start:
            continue
        interval_id = interval["interval_id"]
        loudness = loudness_map.get(interval_id)
        participant = {
            "interval_id": interval_id,
            "player_id": interval["player_id"],
            "threshold": interval["threshold"],
            "gain": interval["gain"],
            "clipped": bool(loudness and loudness["new_clipped_samples"] > 0),
            "missing_source": interval["player_id"] in missing_set,
            "peak": loudness["peak"] if loudness else 0.0,
            "rms": loudness["rms"] if loudness else 0.0,
            "new_clipped_samples":
                loudness["new_clipped_samples"] if loudness else 0,
        }
        segments.append((start, 1, participant, start, end))
        segments.append((end, 0, participant, start, end))

    segments.sort(key=lambda item: (item[0], item[1]))
    events = []
    active = {}
    previous_boundary = None
    for boundary, kind, participant, start, end in segments:
        if previous_boundary is not None and boundary > previous_boundary and active:
            events.append(self_or_merged(previous_boundary, boundary, active,
                                         output_rate))
        if kind == 1:
            active[participant["interval_id"]] = participant
        else:
            active.pop(participant["interval_id"], None)
        previous_boundary = boundary
    return events


def self_or_merged(start_sample: int, end_sample: int, active: dict,
                   output_rate: int) -> dict:
    participants = sorted(active.values(),
                          key=lambda item: (item["player_id"], item["interval_id"]))
    activated_samples = end_sample - start_sample
    event = {
        "start_sample": start_sample,
        "end_sample": end_sample,
        "start_ms": round(samples_to_ms(start_sample), 6),
        "end_ms": round(samples_to_ms(end_sample), 6),
        "activation_samples": activated_samples,
        "activation_ms": round(samples_to_ms(activated_samples, output_rate), 6),
        "simultaneous": len(participants) > 1,
        "participants": participants,
        "players": [item["player_id"] for item in participants],
        "any_clipped": any(item["clipped"] for item in participants),
        "any_missing_source": any(item["missing_source"] for item in participants),
        "gains": {item["player_id"]: item["gain"] for item in participants},
    }
    return event

