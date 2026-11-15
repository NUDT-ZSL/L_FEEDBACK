"""Manifest parsing and per-segment validation.

Manifest schema (JSON):
{
  "crossfade_ms": 20,                # optional, default 20
  "output": {"sample_rate": 48000, "bits": 24, "channels": 2},  # optional
  "segments": [
    {"source": "a.wav", "start_sec": 0.0, "end_sec": 5.0,
     "target_loudness_lkfs": -23.0, "fade_in_ms": 10, "fade_out_ms": 10}
  ]
}
Relative source paths resolve against the manifest directory.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

from .errors import Code, PipelineError

DEFAULT_CROSSFADE_MS = 20.0
OUTPUT_SAMPLE_RATE = 48000
OUTPUT_BITS = 24
OUTPUT_CHANNELS = 2

_REQUIRED_FIELDS = ("source", "start_sec", "end_sec", "target_loudness_lkfs")


@dataclass
class SegmentSpec:
    index: int
    source: Optional[str] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    target_lkfs: Optional[float] = None
    fade_in_ms: float = 0.0
    fade_out_ms: float = 0.0
    # Pre-validation failure (recorded, segment skipped):
    error_code: str = Code.OK
    error_message: str = ""


@dataclass
class Manifest:
    path: str
    sha256: str
    crossfade_ms: float
    jobs: Optional[int] = None
    sample_rate: int = OUTPUT_SAMPLE_RATE
    bits: int = OUTPUT_BITS
    channels: int = OUTPUT_CHANNELS
    segments: List[SegmentSpec] = field(default_factory=list)


def _as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def load_manifest(path: str) -> Manifest:
    if not os.path.isfile(path):
        raise PipelineError(Code.E_FILE_NOT_FOUND,
                            f"manifest not found: {path}")
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise PipelineError(Code.E_DECODE_ERROR, f"cannot read manifest: {exc}")
    digest = hashlib.sha256(raw).hexdigest()
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise PipelineError(Code.E_DECODE_ERROR, f"manifest is not valid JSON: {exc}")
    if not isinstance(doc, dict) or not isinstance(doc.get("segments"), list):
        raise PipelineError(Code.E_DECODE_ERROR,
                            "manifest must be an object with a 'segments' list")

    crossfade = _as_number(doc.get("crossfade_ms", DEFAULT_CROSSFADE_MS))
    if crossfade is None or crossfade < 0:
        raise PipelineError(Code.E_DECODE_ERROR, "invalid crossfade_ms")

    jobs_raw = doc.get("jobs")
    jobs = None
    if jobs_raw is not None:
        if isinstance(jobs_raw, int) and not isinstance(jobs_raw, bool) and jobs_raw >= 1:
            jobs = jobs_raw
        else:
            raise PipelineError(Code.E_DECODE_ERROR, "invalid jobs value")

    out = doc.get("output", {}) or {}
    rate = int(out.get("sample_rate", OUTPUT_SAMPLE_RATE))
    bits = int(out.get("bits", OUTPUT_BITS))
    channels = int(out.get("channels", OUTPUT_CHANNELS))
    if rate != OUTPUT_SAMPLE_RATE or bits != OUTPUT_BITS or channels != OUTPUT_CHANNELS:
        raise PipelineError(
            Code.E_UNSUPPORTED_FORMAT,
            f"output must be {OUTPUT_SAMPLE_RATE} Hz / {OUTPUT_BITS} bit / "
            f"{OUTPUT_CHANNELS} ch for this delivery profile")

    base = os.path.dirname(os.path.abspath(path))
    segments: List[SegmentSpec] = []
    for idx, entry in enumerate(doc["segments"]):
        spec = SegmentSpec(index=idx)
        if not isinstance(entry, dict):
            spec.error_code = Code.E_MISSING_FIELD
            spec.error_message = "segment entry is not an object"
            segments.append(spec)
            continue
        missing = [k for k in _REQUIRED_FIELDS if k not in entry]
        if missing:
            spec.error_code = Code.E_MISSING_FIELD
            spec.error_message = "missing field(s): " + ", ".join(missing)
            segments.append(spec)
            continue
        start = _as_number(entry.get("start_sec"))
        end = _as_number(entry.get("end_sec"))
        target = _as_number(entry.get("target_loudness_lkfs"))
        fade_in = _as_number(entry.get("fade_in_ms", 0.0))
        fade_out = _as_number(entry.get("fade_out_ms", 0.0))
        source = entry.get("source")
        if not isinstance(source, str) or not source:
            spec.error_code = Code.E_MISSING_FIELD
            spec.error_message = "field 'source' must be a non-empty string"
        elif start is None or end is None or target is None:
            spec.error_code = Code.E_MISSING_FIELD
            spec.error_message = "start_sec/end_sec/target_loudness_lkfs must be numbers"
        elif fade_in is None or fade_out is None or fade_in < 0 or fade_out < 0:
            spec.error_code = Code.E_MISSING_FIELD
            spec.error_message = "fade durations must be non-negative numbers"
        elif start < 0 or end <= start:
            spec.error_code = Code.E_INVALID_RANGE
            spec.error_message = f"invalid range [{start}, {end})"
        if not os.path.isabs(source or ""):
            source = os.path.normpath(os.path.join(base, source or ""))
        spec.source = source
        spec.start_sec = start
        spec.end_sec = end
        spec.target_lkfs = target
        spec.fade_in_ms = fade_in or 0.0
        spec.fade_out_ms = fade_out or 0.0
        if spec.error_code == Code.OK and not os.path.isfile(source):
            spec.error_code = Code.E_FILE_NOT_FOUND
            spec.error_message = f"source not found: {source}"
        segments.append(spec)

    return Manifest(path=os.path.abspath(path), sha256=digest,
                    crossfade_ms=crossfade, jobs=jobs, segments=segments)
