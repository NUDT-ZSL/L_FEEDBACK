"""Deterministic in-memory subtitle samples used by the test suite."""

from __future__ import annotations

import os

import pysubs2


def make_ssa(events: list[tuple[int, int, str]], fmt: str = "srt") -> pysubs2.SSAFile:
    ssa = pysubs2.SSAFile()
    ssa.events.clear()
    for start, end, text in events:
        ssa.events.append(
            pysubs2.SSAEvent(
                start=start,
                end=end,
                text=text,
                style="Main" if fmt == "ass" else "Default",
            )
        )
    return ssa


def write_subtitle(
    path: str,
    events: list[tuple[int, int, str]],
    fmt: str = "srt",
    encoding: str = "utf-8",
) -> None:
    ssa = make_ssa(events, fmt=fmt)
    content = ssa.to_string(fmt)
    codec = {
        "utf-8": "utf-8",
        "utf-8-bom": "utf-8-sig",
        "utf-16-le": "utf-16",
        "utf-16-be": "utf-16",
    }[encoding]
    if encoding == "utf-16-be":
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        raw = b"\xfe\xff" + content.encode("utf-16-be")
        with open(path, "wb") as handle:
            handle.write(raw)
        return
    with open(path, "w", encoding=codec, newline="") as handle:
        handle.write(content)


def reference_events(count: int = 24, spacing: int = 4000, duration: int = 1500) -> list[tuple[int, int, str]]:
    lines = [
        "Welcome to the briefing.",
        "The schedule changes at noon.",
        "Please review the attached notes.",
        "Lunch will be served downstairs.",
        "Questions are welcome any time.",
        "The next session starts shortly.",
        "Keep your badge visible at all times.",
        "Wi-Fi details are on the handout.",
    ]
    return [
        (
            500 + index * spacing,
            500 + index * spacing + duration,
            lines[index % len(lines)],
        )
        for index in range(count)
    ]


def linear_drift(
    events: list[tuple[int, int, str]],
    scale: float,
    offset_ms: int,
) -> list[tuple[int, int, str]]:
    drifted: list[tuple[int, int, str]] = []
    for start, end, text in events:
        new_start = round(scale * start + offset_ms)
        new_end = new_start + (end - start)
        drifted.append((new_start, new_end, text))
    return drifted


def segmented_drift(
    events: list[tuple[int, int, str]],
    boundary: int,
    first_scale: float,
    first_offset: int,
    second_scale: float,
    second_offset: int,
) -> list[tuple[int, int, str]]:
    drifted: list[tuple[int, int, str]] = []
    for start, end, text in events:
        if start < boundary:
            new_start = round(first_scale * start + first_offset)
        else:
            new_start = round(second_scale * start + second_offset)
        drifted.append((new_start, new_start + (end - start), text))
    return drifted


def drop_event(
    events: list[tuple[int, int, str]], index: int
) -> list[tuple[int, int, str]]:
    return [event for event_index, event in enumerate(events) if event_index != index]


def swap_events(
    events: list[tuple[int, int, str]], index: int
) -> list[tuple[int, int, str]]:
    swapped = list(events)
    swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
    return swapped


TRANSLATIONS = [
    "欢迎参加本次简报。",
    "日程在中午发生变化。",
    "请查阅随附的笔记。",
    "楼下将供应午餐。",
    "随时欢迎提出问题。",
    "下一场会议即将开始。",
    "请始终佩戴您的证件。",
    "无线网络信息在讲义上。",
]

ARABIC_LINES = [
 "مرحبا بكم في الإحاطة.",
 "يتغير الجدول عند الظهر.",
 "يرجى مراجعة الملاحظات المرفقة.",
 "سيتم تقديم الغداء في الطابق السفلي.",
 "الأسئلة مرحب بها في أي وقت.",
 "تبدأ الجلسة التالية بعد قليل.",
 "احمل شارتك ظاهرة طوال الوقت.",
 "تفاصيل الشبكة في النشرة.",
]


def translated_events(
    events: list[tuple[int, int, str]], texts: list[str] | None = None
) -> list[tuple[int, int, str]]:
    texts = texts or TRANSLATIONS
    return [
        (start, end, texts[index % len(texts)])
        for index, (start, end, _text) in enumerate(events)
    ]


def overlap_events() -> list[tuple[int, int, str]]:
    return [
        (500, 1500, "Welcome to the briefing."),
        (1000, 2500, "Overlapping cue"),
        (4500, 6000, "The schedule changes at noon."),
        (-500, 400, "Negative cue"),
        (12500, 12500, "Zero duration cue"),
        (16500, 18000, "Please review the attached notes."),
    ]


def write_bytes(path: str, raw: bytes) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(raw)
