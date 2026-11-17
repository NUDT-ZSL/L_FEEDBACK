"""Timeline manifest loading and validation.

Manifest schema (JSON):

    {
      "version": 1,
      "output": {"sample_rate": 48000, "channels": 2, "window_ms": 100},
      "segments": [
        {
          "id": "ambience",
          "path": "samples/a.wav",
          "fade_in":  {"length_ms": 0, "curve": "linear"},
          "fade_out": {"length_ms": 250, "curve": "equal_power"}
        }
      ]
    }

Paths are resolved relative to the manifest file. The crossfade overlap
between segment i and i+1 equals segment i's fade_out length in output
frames; the next segment's fade_in is applied independently over its own
head, so a symmetric crossfade sets both to the same length.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field

from .dsp import CURVES
from .errors import BusinessError, ErrorCode, ExitCode

MANIFEST_VERSION = 1


@dataclass
class FadeSpec:
    length_ms: float
    curve: str


@dataclass
class SegmentSpec:
    id: str
    path: str
    fade_in: FadeSpec
    fade_out: FadeSpec


@dataclass
class Manifest:
    sample_rate: int
    channels: int
    window_ms: float
    segments: list = field(default_factory=list)
    sha256: str = ""
    path: str = ""


def _invalid(message, details=None):
    return BusinessError(
        ErrorCode.E_MANIFEST_INVALID,
        message,
        ExitCode.MANIFEST_INVALID,
        details,
    )


def _parse_fade(obj, seg_id, which):
    if not isinstance(obj, dict):
        raise _invalid(f"segment '{seg_id}': '{which}' must be an object")
    length_ms = obj.get("length_ms")
    curve = obj.get("curve", "linear")
    if not isinstance(length_ms, (int, float)) or isinstance(length_ms, bool):
        raise _invalid(f"segment '{seg_id}': '{which}.length_ms' must be a number")
    if length_ms < 0:
        raise BusinessError(
            ErrorCode.E_FADE_NEGATIVE,
            f"segment '{seg_id}': '{which}.length_ms' is negative ({length_ms})",
            ExitCode.RENDER_ERROR,
            {"segment": seg_id, "fade": which, "length_ms": length_ms},
        )
    if curve not in CURVES:
        raise _invalid(
            f"segment '{seg_id}': '{which}.curve' must be one of {list(CURVES)}"
        )
    return FadeSpec(length_ms=float(length_ms), curve=curve)


def load_manifest(path):
    if not os.path.isfile(path):
        raise BusinessError(
            ErrorCode.E_INPUT_MISSING,
            f"manifest not found: {path}",
            ExitCode.INPUT_MISSING,
        )
    try:
        with open(path, "rb") as fh:
            raw_bytes = fh.read()
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _invalid(f"manifest is not valid JSON: {exc}")
    if not isinstance(raw, dict):
        raise _invalid("manifest root must be an object")

    output = raw.get("output")
    if not isinstance(output, dict):
        raise _invalid("manifest must contain an 'output' object")
    sample_rate = output.get("sample_rate")
    channels = output.get("channels")
    window_ms = output.get("window_ms")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise _invalid("'output.sample_rate' must be a positive integer")
    if channels not in (1, 2):
        raise _invalid("'output.channels' must be 1 (mono) or 2 (stereo)")
    if not isinstance(window_ms, (int, float)) or isinstance(window_ms, bool) or window_ms <= 0:
        raise _invalid("'output.window_ms' must be a positive number")

    raw_segments = raw.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise _invalid("manifest must contain a non-empty 'segments' list")

    base_dir = os.path.dirname(os.path.abspath(path))
    segments = []
    seen_ids = set()
    for index, item in enumerate(raw_segments):
        if not isinstance(item, dict):
            raise _invalid(f"segment #{index} must be an object")
        seg_id = item.get("id")
        seg_path = item.get("path")
        if not isinstance(seg_id, str) or not seg_id:
            raise _invalid(f"segment #{index}: 'id' must be a non-empty string")
        if seg_id in seen_ids:
            raise _invalid(f"duplicate segment id: '{seg_id}'")
        seen_ids.add(seg_id)
        if not isinstance(seg_path, str) or not seg_path:
            raise _invalid(f"segment '{seg_id}': 'path' must be a non-empty string")
        abs_path = seg_path if os.path.isabs(seg_path) else os.path.join(base_dir, seg_path)
        fade_in = _parse_fade(item.get("fade_in", {"length_ms": 0}), seg_id, "fade_in")
        fade_out = _parse_fade(item.get("fade_out", {"length_ms": 0}), seg_id, "fade_out")
        segments.append(SegmentSpec(id=seg_id, path=abs_path, fade_in=fade_in, fade_out=fade_out))

    return Manifest(
        sample_rate=sample_rate,
        channels=channels,
        window_ms=float(window_ms),
        segments=segments,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        path=os.path.abspath(path),
    )
