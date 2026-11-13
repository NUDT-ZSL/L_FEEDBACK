"""Timeline alignment, overlap resolution, gain/limit and degradation policy."""

import math
import os

from . import wavio
from .errors import WavFormatError
from .resample import resample, expected_output_len

CROSSFADE_MS = 5.0


def ms_to_samples(ms, sample_rate):
    """Global, consistent rounding: round-half-up toward +inf at .5."""
    exact = ms * sample_rate / 1000.0
    return int(math.floor(exact + 0.5))


def db_to_linear(db):
    return 10.0 ** (db / 20.0)


def pcm16_from_float(values, headroom_db):
    """Hard limit with a configurable peak margin; deterministic rounding."""
    limit = 32767.0 * (10.0 ** (headroom_db / 20.0))  # headroom_db <= 0
    out = [0] * len(values)
    clipped = 0
    for i, v in enumerate(values):
        if v > limit:
            out[i] = int(math.ceil(limit))
            clipped += 1
        elif v < -limit:
            out[i] = -int(math.ceil(limit))
            clipped += 1
        else:
            # Round half away from zero for stable integer output.
            out[i] = int(math.floor(v + 0.5)) if v >= 0 else -int(math.floor(-v + 0.5))
    return out, clipped, limit


class DegradationReport:
    def __init__(self):
        self.recoverable = []
        self.fatal = []

    def add_recoverable(self, kind, event, detail):
        self.recoverable.append(
            {"kind": kind, "event": event, "detail": detail}
        )

    def add_fatal(self, kind, event, detail):
        self.fatal.append({"kind": kind, "event": event, "detail": detail})

    def as_dict(self):
        return {"recoverable": self.recoverable, "fatal": self.fatal}


def _event_key(row):
    return (int(row["track"]), int(row["event_index"]), int(row["start_ms"]))


def mix_clip(rows, wav_dir, target_rate, clip_ms, headroom_db, report):
    """Render one clip. Returns (pcm16 list, event metadata list, total samples)."""
    total_samples = ms_to_samples(clip_ms, target_rate)
    tracks = {}
    event_meta = []

    for row in sorted(rows, key=_event_key):
        track = int(row["track"])
        start_ms = int(row["start_ms"])
        dur_ms = int(row["duration_ms"])
        gain_db = float(row["gain_db"])
        missing = str(row.get("missing", "0")).strip() in ("1", "true", "True")
        force_mute = str(row.get("force_mute", "0")).strip() in ("1", "true", "True")
        ev_id = "%s/t%s/e%s" % (row["clip_id"], track, row["event_index"])

        start = ms_to_samples(start_ms, target_rate)
        end = ms_to_samples(start_ms + dur_ms, target_rate)
        meta = {
            "event": ev_id,
            "track": track,
            "start_ms": start_ms,
            "duration_ms": dur_ms,
            "target_start_sample": start,
            "target_end_sample": end,
            "applied_gain_db": gain_db,
            "gain_linear": db_to_linear(gain_db),
            "placeholder": False,
            "muted": bool(force_mute),
            "source_rate": None,
            "source_samples": None,
            "resampled_samples": None,
            "notes": [],
        }

        buf = tracks.setdefault(track, [0.0] * total_samples)

        # Recoverable: declared missing -> silence placeholder.
        if missing:
            report.add_recoverable("missing_declared", ev_id, "missing flag set")
            meta["placeholder"] = True
            meta["notes"].append("silence_placeholder:declared_missing")
            # A placeholder on the same track still covers prior audio in its
            # window: write explicit silence through the overlap path.
            _place_on_track(
                tracks.setdefault(track, [0.0] * total_samples),
                [0.0] * max(0, end - start), start, total_samples,
                target_rate, 1.0, report, ev_id,
            )
            event_meta.append(meta)
            continue

        wav_name = row.get("audio_file", "")
        wav_path = os.path.join(wav_dir, wav_name) if wav_name else ""
        try:
            if not wav_name:
                raise WavFormatError("empty audio_file", code="E_WAV_OPEN")
            wav = wavio.read_wav(wav_path)
        except WavFormatError as exc:
            # Unreadable / malformed source: recoverable silence placeholder,
            # while pure structural failures remain recorded distinctly.
            if exc.code == "E_WAV_FORMAT":
                report.add_fatal("wav_format", ev_id, exc.to_record())
                raise
            report.add_recoverable("wav_unreadable", ev_id, exc.to_record())
            meta["placeholder"] = True
            meta["notes"].append("silence_placeholder:unreadable")
            event_meta.append(meta)
            continue

        meta["source_rate"] = wav.sample_rate
        meta["source_samples"] = len(wav.samples)

        if len(wav.samples) == 0:
            report.add_recoverable("zero_duration", ev_id, "wav has zero samples")
            meta["placeholder"] = True
            meta["notes"].append("silence_placeholder:zero_length")
            event_meta.append(meta)
            continue

        resampled, rmeta = resample(wav.samples, wav.sample_rate, target_rate)
        meta["resampled_samples"] = len(resampled)

        wanted = end - start
        if len(resampled) != wanted:
            if len(resampled) > wanted:
                report.add_recoverable(
                    "duration_exceeds_window",
                    ev_id,
                    {"have": len(resampled), "window": wanted, "action": "truncate"},
                )
                meta["notes"].append("truncated_to_window")
            else:
                report.add_recoverable(
                    "duration_short",
                    ev_id,
                    {"have": len(resampled), "window": wanted, "action": "pad_silence"},
                )
                meta["notes"].append("padded_silence")
            if len(resampled) > wanted:
                resampled = resampled[:wanted]
            else:
                resampled = resampled + [0.0] * (wanted - len(resampled))

        if force_mute:
            resampled = [0.0] * len(resampled)
            meta["notes"].append("force_muted")
            report.add_recoverable("forced_mute", ev_id, "force_mute flag set")

        gain = db_to_linear(gain_db)
        _place_on_track(buf, resampled, start, total_samples, target_rate, gain, report, ev_id)
        event_meta.append(meta)

    # Mix tracks in ascending track number: deterministic sum order.
    mix = [0.0] * total_samples
    for track in sorted(tracks):
        buf = tracks[track]
        for i in range(total_samples):
            mix[i] += buf[i]

    pcm, clipped, limit = pcm16_from_float(mix, headroom_db)
    return pcm, event_meta, total_samples, clipped, limit


def _place_on_track(buf, audio, start, total, rate, gain, report, ev_id):
    """Place audio; same-track overlap is covered with a later-event fade-in."""
    fade_len = ms_to_samples(CROSSFADE_MS, rate)
    n = len(audio)
    # Clip placement to the legal clip window (handles negative offsets).
    src0 = 0
    dst0 = start
    if dst0 < 0:
        src0 = -dst0
        dst0 = 0
    dst1 = min(total, start + n)
    length = dst1 - dst0
    if length <= 0:
        report.add_recoverable("fully_outside_window", ev_id, {"start": start})
        return

    for k in range(length):
        idx = dst0 + k
        sample = audio[src0 + k] * gain
        # Crossfade only at a partial-overlap boundary (event begins after the
        # prior one started). At a shared start, the later event (including a
        # silence placeholder) covers the earlier event immediately and fully.
        if start > 0 and k < fade_len and buf[idx] != 0.0:
            # Equal-power-ish linear crossfade: old fades out, new fades in.
            a_in = (k + 1) / float(fade_len)
            buf[idx] = buf[idx] * (1.0 - a_in) + sample * a_in
        else:
            buf[idx] = sample
