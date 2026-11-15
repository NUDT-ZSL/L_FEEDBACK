"""Descriptor schema validation.

Expected descriptor shape (JSON)::

    {
      "frame_rate": 30.0,                     # required, > 0
      "output": {                             # required
        "sample_rate": 48000,                 # int, 8000..192000
        "bit_depth": 16,                      # 16 or 24
        "channels": 2                         # 1 or 2
      },
      "crossfade_ms": 10.0,                   # optional, >= 0
      "fade_ms": 5.0,                         # optional, >= 0
      "gain_limits_db": {"min": -20.0, "max": 10.0},  # optional, min < max
      "tracks": [                             # required, >= 1 entry
        {
          "id": "dialog",                     # optional label
          "path": "samples/dialog.wav",       # required, non-empty
          "events": [                         # required, >= 1 entry
            {"id": "line1",                   # optional label
             "track_sample": 0,               # int >= 0
             "frame": 30}                     # number >= 0
          ]
        }
      ]
    }
"""

from .errors import SchemaError

_MIN_SAMPLE_RATE = 8000
_MAX_SAMPLE_RATE = 192000
_BIT_DEPTHS = (16, 24)
_CHANNEL_COUNTS = (1, 2)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_descriptor(desc):
    """Validate a parsed descriptor dict.

    Returns a normalized copy with defaults applied.
    Raises :class:`SchemaError` listing every failing field path.
    """
    errors = []

    if not isinstance(desc, dict):
        raise SchemaError(["$"])

    frame_rate = desc.get("frame_rate")
    if not _is_number(frame_rate) or frame_rate <= 0:
        errors.append("frame_rate")

    output = desc.get("output")
    if not isinstance(output, dict):
        errors.append("output")
        output = {}
    sample_rate = output.get("sample_rate")
    if not _is_int(sample_rate) or not (_MIN_SAMPLE_RATE <= sample_rate <= _MAX_SAMPLE_RATE):
        errors.append("output.sample_rate")
    bit_depth = output.get("bit_depth")
    if bit_depth not in _BIT_DEPTHS:
        errors.append("output.bit_depth")
    channels = output.get("channels")
    if channels not in _CHANNEL_COUNTS:
        errors.append("output.channels")

    crossfade_ms = desc.get("crossfade_ms", 10.0)
    if not _is_number(crossfade_ms) or crossfade_ms < 0:
        errors.append("crossfade_ms")

    fade_ms = desc.get("fade_ms", 5.0)
    if not _is_number(fade_ms) or fade_ms < 0:
        errors.append("fade_ms")

    gain_limits = desc.get("gain_limits_db", {"min": -20.0, "max": 10.0})
    if not isinstance(gain_limits, dict):
        errors.append("gain_limits_db")
        gain_limits = {}
    gain_min = gain_limits.get("min", -20.0)
    gain_max = gain_limits.get("max", 10.0)
    if not _is_number(gain_min):
        errors.append("gain_limits_db.min")
    if not _is_number(gain_max):
        errors.append("gain_limits_db.max")
    if _is_number(gain_min) and _is_number(gain_max) and gain_min >= gain_max:
        errors.append("gain_limits_db")

    tracks = desc.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        errors.append("tracks")
        tracks = []
    for index, track in enumerate(tracks):
        prefix = "tracks[%d]" % index
        if not isinstance(track, dict):
            errors.append(prefix)
            continue
        path = track.get("path")
        if not isinstance(path, str) or not path.strip():
            errors.append(prefix + ".path")
        events = track.get("events")
        if not isinstance(events, list) or not events:
            errors.append(prefix + ".events")
            continue
        for ev_index, event in enumerate(events):
            ev_prefix = "%s.events[%d]" % (prefix, ev_index)
            if not isinstance(event, dict):
                errors.append(ev_prefix)
                continue
            track_sample = event.get("track_sample")
            if not _is_int(track_sample) or track_sample < 0:
                errors.append(ev_prefix + ".track_sample")
            frame = event.get("frame")
            if not _is_number(frame) or frame < 0:
                errors.append(ev_prefix + ".frame")

    if errors:
        raise SchemaError(errors)

    normalized = dict(desc)
    normalized["output"] = {
        "sample_rate": sample_rate,
        "bit_depth": bit_depth,
        "channels": channels,
    }
    normalized["crossfade_ms"] = float(crossfade_ms)
    normalized["fade_ms"] = float(fade_ms)
    normalized["gain_limits_db"] = {"min": float(gain_min), "max": float(gain_max)}
    return normalized
