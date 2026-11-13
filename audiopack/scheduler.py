"""Same-channel overlap resolution.

Events for one language are placed on an integer sample timeline.  Higher
priority events claim their natural sample positions first; a lower priority
clip that overlaps any already-placed clip is shifted forward as a whole until
it no longer overlaps.  Shifting a low-priority event never moves a
higher-priority one, and a shift beyond the event's max delay marks the event
as dropped instead of placing it late.
"""

from __future__ import annotations

from dataclasses import dataclass

from .assets import LoadedClip
from .config import Config
from .manifest import Event, Manifest


@dataclass
class Placement:
    event_id: str
    lang: str
    voice_key: str
    source_name: str
    source_path: str
    natural_start: int
    start: int
    frames: int
    end: int
    priority: int
    shift_samples: int
    content_hash: str

    @property
    def shift_ms(self) -> float:
        return self.shift_samples / 48.0  # 48 samples == 1 ms

    def to_index_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "lang": self.lang,
            "source_file": self.source_name,
            "source_path": self.source_path,
            "voice_key": self.voice_key,
            "natural_start_sample": self.natural_start,
            "sample_offset": self.start,
            "offset_ms": self.start / 48.0,
            "frames": self.frames,
            "end_sample": self.end,
            "priority": self.priority,
            "shift_samples": self.shift_samples,
            "shift_ms": round(self.shift_ms, 3),
            "content_hash": self.content_hash,
        }


@dataclass
class DroppedEvent:
    event_id: str
    lang: str
    reason: str
    natural_start: int
    required_start: int
    max_delay_samples: int
    required_delay_samples: int
    priority: int

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "lang": self.lang,
            "reason": self.reason,
            "natural_start_sample": self.natural_start,
            "required_start_sample": self.required_start,
            "max_delay_samples": self.max_delay_samples,
            "required_delay_samples": self.required_delay_samples,
            "required_delay_ms": round(self.required_delay_samples / 48.0, 3),
            "max_delay_ms": round(self.max_delay_samples / 48.0, 3),
            "priority": self.priority,
        }


@dataclass
class LanguageSchedule:
    lang: str
    placements: list[Placement]
    dropped: list[DroppedEvent]


def schedule_language(
    lang: str,
    events: list[Event],
    manifest: Manifest,
    clips: dict[str, LoadedClip],
    config: Config,
) -> LanguageSchedule:
    """Resolve overlaps for one language on the integer sample timeline."""
    # Higher priority first; ties resolve deterministically by natural time
    # then id so output is reproducible across runs.
    ordered = sorted(
        events,
        key=lambda event: (
            -event.priority,
            manifest.start_sample(event, config),
            event.event_id,
        ),
    )

    placements: list[Placement] = []
    dropped: list[DroppedEvent] = []

    for event in ordered:
        if event.voice_key is None:
            # Control events (countdown markers without audio) carry no clip.
            continue
        clip = clips[event.voice_key]
        natural_start = manifest.start_sample(event, config)
        frames = clip.frames
        max_delay_samples = event.max_delay_ms * config.ms_frame

        candidate = natural_start
        blocking = True
        while blocking:
            blocking = False
            for placed in placements:
                overlap = candidate < placed.end and candidate + frames > placed.start
                if overlap:
                    # Move the whole lower-priority clip past the blocker.
                    candidate = placed.end
                    blocking = True
                    break

        shift = candidate - natural_start
        if shift > max_delay_samples:
            dropped.append(
                DroppedEvent(
                    event_id=event.event_id,
                    lang=lang,
                    reason="exceeds_max_delay",
                    natural_start=natural_start,
                    required_start=candidate,
                    max_delay_samples=max_delay_samples,
                    required_delay_samples=shift,
                    priority=event.priority,
                )
            )
            continue

        placements.append(
            Placement(
                event_id=event.event_id,
                lang=lang,
                voice_key=event.voice_key,
                source_name=clip.source_name,
                source_path=clip.path,
                natural_start=natural_start,
                start=candidate,
                frames=frames,
                end=candidate + frames,
                priority=event.priority,
                shift_samples=shift,
                content_hash=clip.content_hash,
            )
        )

    placements.sort(key=lambda item: (item.start, item.event_id))
    dropped.sort(key=lambda item: item.event_id)
    return LanguageSchedule(lang=lang, placements=placements, dropped=dropped)


def schedule_all(
    manifest: Manifest,
    clips_by_lang: dict[str, dict[str, LoadedClip]],
    config: Config,
) -> dict[str, LanguageSchedule]:
    schedules: dict[str, LanguageSchedule] = {}
    by_lang: dict[str, list[Event]] = {}
    for event in manifest.events:
        if event.voice_key is None:
            continue
        for lang in event.langs:
            by_lang.setdefault(lang, []).append(event)

    for lang in sorted(by_lang):
        schedules[lang] = schedule_language(
            lang, by_lang[lang], manifest, clips_by_lang[lang], config
        )
    return schedules
