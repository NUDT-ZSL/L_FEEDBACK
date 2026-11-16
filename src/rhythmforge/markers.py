"""Sample-accurate synchronization markers for section and BPM boundaries."""

from __future__ import annotations

from .timing import Section, TimeMap, TimeSignature


def build_markers(time_map: TimeMap, sections: list[Section],
                  signatures: list[TimeSignature]) -> list[dict]:
    """Return markers ordered by frame position.

    Marker types:

    * ``section`` - section start boundaries (and the timeline end),
    * ``bpm`` - every BPM anchor point,
    * ``timesig`` - time signature change points.

    Multiple annotations at the same frame are emitted as separate records so
    game-side scheduling can subscribe per type without losing information.
    """
    markers: list[dict] = []

    for index, section in enumerate(sections):
        frame = int(time_map.beat_to_frame(section.start_beat))
        markers.append({
            "type": "section",
            "name": section.section_id,
            "beat": section.start_beat,
            "frame": frame,
            "index": index,
        })
    end_frame = int(time_map.beat_to_frame(sections[-1].end_beat))
    markers.append({
        "type": "section",
        "name": f"{sections[-1].section_id}_end",
        "beat": sections[-1].end_beat,
        "frame": end_frame,
        "index": len(sections),
    })

    for index, point in enumerate(time_map.beats):
        markers.append({
            "type": "bpm",
            "name": f"bpm_{index}",
            "beat": float(point),
            "frame": int(time_map.beat_to_frame(float(point))),
            "bpm": float(time_map.bpms[index]),
        })

    for index, signature in enumerate(signatures):
        markers.append({
            "type": "timesig",
            "name": f"timesig_{index}",
            "beat": signature.beat,
            "frame": int(time_map.beat_to_frame(signature.beat)),
            "numerator": signature.numerator,
            "denominator": signature.denominator,
        })

    markers.sort(key=lambda item: (item["frame"], item["type"], item["name"]))
    return markers
