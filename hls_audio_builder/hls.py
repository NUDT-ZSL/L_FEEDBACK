from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .errors import ConsistencyError
from .timeline import TimelineSegment


@dataclass(frozen=True)
class MediaPlaylistInfo:
    durations: Tuple[Decimal, ...]
    uris: Tuple[str, ...]


@dataclass(frozen=True)
class MasterEntry:
    bandwidth: int
    average_bandwidth: int
    uri: str


def duration_text(samples: int, sample_rate: int) -> str:
    return f"{(Decimal(samples) / Decimal(sample_rate)):.9f}"


def build_media_playlist(
    segments: Sequence[TimelineSegment],
    sample_rate: int,
    segment_uri_prefix: str = "",
) -> str:
    durations = [
        Decimal(segment.sample_count) / Decimal(sample_rate)
        for segment in segments
    ]
    target_duration = max(1, int(math.ceil(float(max(durations)))))
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        f"#EXT-X-TARGETDURATION:{target_duration}",
        "#EXT-X-MEDIA-SEQUENCE:0",
        "#EXT-X-PLAYLIST-TYPE:VOD",
        "#EXT-X-INDEPENDENT-SEGMENTS",
    ]
    for segment in segments:
        uri = f"{segment_uri_prefix}segment_{segment.index:05d}.aac"
        lines.append(f"#EXTINF:{duration_text(segment.sample_count, sample_rate)},")
        lines.append(uri)
    lines.append("#EXT-X-ENDLIST")
    return "\n".join(lines) + "\n"


def build_master_playlist(entries: Sequence[MasterEntry]) -> str:
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        "#EXT-X-INDEPENDENT-SEGMENTS",
    ]
    for entry in entries:
        lines.append(
            "#EXT-X-STREAM-INF:"
            f"BANDWIDTH={entry.bandwidth},"
            f"AVERAGE-BANDWIDTH={entry.average_bandwidth},"
            'CODECS="mp4a.40.2"'
        )
        lines.append(entry.uri)
    return "\n".join(lines) + "\n"


def parse_media_playlist(path: Path) -> MediaPlaylistInfo:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConsistencyError(f"cannot read media playlist {path}: {exc}") from exc

    durations: List[Decimal] = []
    uris: List[str] = []
    pending_duration = None
    for line in lines:
        if line.startswith("#EXTINF:"):
            value = line[len("#EXTINF:") :].split(",", 1)[0]
            try:
                pending_duration = Decimal(value)
            except InvalidOperation as exc:
                raise ConsistencyError(
                    f"invalid EXTINF in media playlist {path}: {line}"
                ) from exc
        elif line and not line.startswith("#"):
            if pending_duration is None:
                raise ConsistencyError(
                    f"segment URI without EXTINF in media playlist {path}: {line}"
                )
            durations.append(pending_duration)
            uris.append(line)
            pending_duration = None
    return MediaPlaylistInfo(durations=tuple(durations), uris=tuple(uris))


def parse_master_playlist(path: Path) -> List[MasterEntry]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConsistencyError(f"cannot read master playlist {path}: {exc}") from exc

    entries: List[MasterEntry] = []
    pending: Optional[Dict[str, int]] = None
    for line in lines:
        if line.startswith("#EXT-X-STREAM-INF:"):
            values: dict[str, int] = {}
            for key, value in re.findall(r"([A-Z-]+)=([0-9]+)", line):
                values[key] = int(value)
            pending = values
        elif line and not line.startswith("#") and pending is not None:
            try:
                entries.append(
                    MasterEntry(
                        bandwidth=pending["BANDWIDTH"],
                        average_bandwidth=pending["AVERAGE-BANDWIDTH"],
                        uri=line,
                    )
                )
            except KeyError as exc:
                raise ConsistencyError(
                    f"master playlist entry is missing {exc.args[0]}: {path}"
                ) from exc
            pending = None
    return entries

