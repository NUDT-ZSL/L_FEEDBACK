"""Subtitle retiming onto the stretched dub timeline with duration constraints.

Constraint enforcement is deterministic: entries are processed in
(start, id) order; over-long entries are split into equal parts with the
text partitioned by character count; under-long entries are extended, and
on conflict merged with the following entry.
"""

from __future__ import annotations

import json
import math

from .errors import DubError, E_INPUT_UNREADABLE, E_SUBTITLE_OVERFLOW

EPS = 1e-9


def load_subtitles(path: str) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise DubError(E_INPUT_UNREADABLE, f"cannot read subtitles {path}: {exc}") from exc
    if not isinstance(data, list):
        raise DubError(E_INPUT_UNREADABLE, f"subtitles {path}: top level must be a list")
    entries = []
    for i, item in enumerate(data):
        try:
            entries.append(
                {
                    "id": str(item.get("id", f"sub-{i}")),
                    "start": float(item["start"]),
                    "end": float(item["end"]),
                    "text": str(item["text"]),
                    "segment": int(item["segment"]),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DubError(
                E_INPUT_UNREADABLE, f"subtitles {path}: invalid entry {i}: {exc}"
            ) from exc
    return entries


def validate_within_video(entries: list[dict], video_duration: float) -> None:
    for e in entries:
        if e["end"] > video_duration + 1e-3 or e["start"] < 0.0:
            raise DubError(
                E_SUBTITLE_OVERFLOW,
                f"subtitle {e['id']} [{e['start']:.3f}, {e['end']:.3f}] "
                f"exceeds video duration {video_duration:.3f}",
            )


def retime_entries(
    entries: list[dict],
    ref_segments: list[tuple[float, float]],
    ratios: list[float],
) -> list[dict]:
    """Map each entry from the reference timeline onto the stretched timeline."""
    out = []
    for e in entries:
        seg_idx = e["segment"]
        ref_start, _ = ref_segments[seg_idx]
        ratio = ratios[seg_idx]
        out.append(
            {
                "id": e["id"],
                "start": ref_start + (e["start"] - ref_start) * ratio,
                "end": ref_start + (e["end"] - ref_start) * ratio,
                "text": e["text"],
                "segment": seg_idx,
            }
        )
    out.sort(key=lambda x: (x["start"], x["id"]))
    return out


def _split_text(text: str, parts: int) -> list[str]:
    n = len(text)
    return [text[round(n * k / parts): round(n * (k + 1) / parts)] for k in range(parts)]


def enforce_constraints(
    entries: list[dict],
    max_duration: float,
    min_duration: float,
    video_duration: float,
) -> tuple[list[dict], list[dict]]:
    """Apply max/min display duration rules deterministically.

    Returns (entries, adjustments). Each adjustment records the entry id,
    the rule applied, and before/after times.
    """
    work = [dict(e) for e in entries]
    result: list[dict] = []
    adjustments: list[dict] = []
    idx = 0
    while idx < len(work):
        entry = work[idx]
        dur = entry["end"] - entry["start"]
        if dur > max_duration + EPS:
            parts = math.ceil(dur / max_duration)
            chunks = _split_text(entry["text"], parts)
            for k in range(parts):
                new_entry = {
                    "id": f"{entry['id']}#{k + 1}",
                    "start": entry["start"] + dur * k / parts,
                    "end": entry["start"] + dur * (k + 1) / parts,
                    "text": chunks[k],
                    "segment": entry["segment"],
                }
                adjustments.append(
                    {
                        "id": new_entry["id"],
                        "rule": "split_max_duration",
                        "before": {"start": round(entry["start"], 6), "end": round(entry["end"], 6)},
                        "after": {
                            "start": round(new_entry["start"], 6),
                            "end": round(new_entry["end"], 6),
                        },
                    }
                )
                result.append(new_entry)
            idx += 1
            continue
        if dur < min_duration - EPS:
            desired_end = entry["start"] + min_duration
            nxt = work[idx + 1] if idx + 1 < len(work) else None
            if nxt is not None and desired_end > nxt["start"] + EPS:
                merged = {
                    "id": f"{entry['id']}+{nxt['id']}",
                    "start": entry["start"],
                    "end": min(max(nxt["end"], desired_end), video_duration),
                    "text": f"{entry['text']} / {nxt['text']}",
                    "segment": entry["segment"],
                }
                adjustments.append(
                    {
                        "id": merged["id"],
                        "rule": "merge_min_duration",
                        "before": {
                            "start": round(entry["start"], 6),
                            "end": round(entry["end"], 6),
                        },
                        "after": {
                            "start": round(merged["start"], 6),
                            "end": round(merged["end"], 6),
                        },
                    }
                )
                work[idx] = merged
                del work[idx + 1]
                continue  # re-check merged entry against max rule
            new_end = min(desired_end, video_duration)
            adjustments.append(
                {
                    "id": entry["id"],
                    "rule": "extend_min_duration",
                    "before": {"start": round(entry["start"], 6), "end": round(entry["end"], 6)},
                    "after": {"start": round(entry["start"], 6), "end": round(new_end, 6)},
                }
            )
            entry = dict(entry, end=new_end)
        result.append(entry)
        idx += 1
    for e in result:
        e["start"] = round(e["start"], 6)
        e["end"] = round(e["end"], 6)
    return result, adjustments


def _fmt_srt_time(seconds: float) -> str:
    ms = round(seconds * 1000.0)
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(entries: list[dict], path: str) -> None:
    blocks = []
    for i, e in enumerate(entries, 1):
        blocks.append(
            f"{i}\n{_fmt_srt_time(e['start'])} --> {_fmt_srt_time(e['end'])}\n{e['text']}\n"
        )
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(blocks))
