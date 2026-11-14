"""In-memory project model parsed from the JSON project file."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Channel:
    channel_id: str
    gain_db: float
    muted: bool


@dataclass(frozen=True)
class Speaker:
    speaker_id: str
    seed: int
    base_frequency: float
    harmonic_ratio: float
    pitch_semitones: float
    source_duration_ms: float
    bound_channel: str
    pan: float


@dataclass(frozen=True)
class RawEvent:
    index: int
    tick: int
    type: str
    speaker: str
    channel: str | None = None


@dataclass(frozen=True)
class RenderFragment:
    """A contiguous [start_sample, end_sample) speech slice for one speaker."""

    speaker_id: str
    channel_id: str
    start_sample: int
    end_sample: int
    blocked: bool
    source_offset: int


@dataclass(frozen=True)
class PttBurst:
    speaker_id: str
    channel_id: str
    start_sample: int
    length_samples: int
    blocked: bool


@dataclass
class Timeline:
    fragments: list[RenderFragment] = field(default_factory=list)
    ptt_bursts: list[PttBurst] = field(default_factory=list)
    last_voice_end: int = 0


@dataclass(frozen=True)
class ExportOptions:
    output_wav: str
    manifest_path: str
    work_dir: str
    chunk_samples: int
    intro_ms: float
    outro_ms: float
    enable_prompts: bool


@dataclass
class Project:
    sample_rate: int
    channels: dict[str, Channel]
    speakers: dict[str, Speaker]
    events: list[RawEvent]
    export: ExportOptions
    tick_rate: int
    raw: dict
    canonical_hash: str
    timeline: Timeline | None = None
