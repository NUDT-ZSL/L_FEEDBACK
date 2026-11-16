"""Timeline JSON loading, conversion, and validation."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .errors import LipAlignError


PathLike = Union[str, os.PathLike]


def round_half_up(value: float) -> int:
    """Round a non-negative value to the nearest integer, ties upward."""

    if value < 0:
        return -int(math.floor(-value + 0.5))
    return int(math.floor(value + 0.5))


@dataclass(frozen=True)
class Phoneme:
    name: str
    mouth: str
    start_sample: int
    end_sample: int

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample


@dataclass(frozen=True)
class Sentence:
    sentence_id: str
    text: str
    start_sample: int
    end_sample: int
    phonemes: List[Phoneme]

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample


@dataclass(frozen=True)
class Timeline:
    source_path: Path
    sample_rate: int
    duration_samples: int
    seed: int
    sentences: List[Sentence]

    @property
    def duration_sec(self) -> float:
        return self.duration_samples / float(self.sample_rate)


def _json_invalid(message: str, path: PathLike) -> LipAlignError:
    return LipAlignError(
        "E_JSON_INVALID", "{}: {}".format(message, path), exit_code=2
    )


def _timeline_invalid(message: str) -> LipAlignError:
    return LipAlignError("E_TIMELINE_INVALID", message, exit_code=2)


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _timeline_invalid("{} must be an integer".format(field))
    return int(value)


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _timeline_invalid("{} must be a number".format(field))
    number = float(value)
    if not math.isfinite(number):
        raise _timeline_invalid("{} must be finite".format(field))
    return number


def _sample_boundary(
    item: Dict[str, Any],
    sample_key: str,
    seconds_key: str,
    sample_rate: int,
    field: str,
) -> int:
    if sample_key in item:
        return _require_int(item[sample_key], field)
    if seconds_key in item:
        return round_half_up(_require_number(item[seconds_key], field) * sample_rate)
    raise _timeline_invalid("Missing {} or {}".format(sample_key, seconds_key))


def _parse_phoneme(
    raw: Any, sentence_index: int, phoneme_index: int, sample_rate: int
) -> Phoneme:
    if not isinstance(raw, dict):
        raise _timeline_invalid(
            "Sentence {} phoneme {} must be an object".format(
                sentence_index, phoneme_index
            )
        )
    name = raw.get("phoneme", raw.get("name"))
    mouth = raw.get("mouth", raw.get("mouth_shape", raw.get("category")))
    if not isinstance(name, str) or not name.strip():
        raise _timeline_invalid(
            "Sentence {} phoneme {} has no phoneme label".format(
                sentence_index, phoneme_index
            )
        )
    if not isinstance(mouth, str) or not mouth.strip():
        raise _timeline_invalid(
            "Sentence {} phoneme {} has no mouth category".format(
                sentence_index, phoneme_index
            )
        )
    start = _sample_boundary(
        raw,
        "start_sample",
        "start_sec",
        sample_rate,
        "sentence[{}].phonemes[{}].start".format(sentence_index, phoneme_index),
    )
    end = _sample_boundary(
        raw,
        "end_sample",
        "end_sec",
        sample_rate,
        "sentence[{}].phonemes[{}].end".format(sentence_index, phoneme_index),
    )
    return Phoneme(name=name.strip(), mouth=mouth.strip(), start_sample=start, end_sample=end)


def _parse_sentence(
    raw: Any, index: int, sample_rate: int
) -> Sentence:
    if not isinstance(raw, dict):
        raise _timeline_invalid("sentence[{}] must be an object".format(index))
    sentence_id_raw = raw.get("id", raw.get("sentence_id", "s{:03d}".format(index + 1)))
    if not isinstance(sentence_id_raw, str) or not sentence_id_raw.strip():
        raise _timeline_invalid("sentence[{}].id must be a non-empty string".format(index))
    text = raw.get("text", "")
    if not isinstance(text, str):
        raise _timeline_invalid("sentence[{}].text must be a string".format(index))
    start = _sample_boundary(
        raw, "start_sample", "start_sec", sample_rate, "sentence[{}].start".format(index)
    )
    end = _sample_boundary(
        raw, "end_sample", "end_sec", sample_rate, "sentence[{}].end".format(index)
    )
    if start < 0 or end <= start:
        raise _timeline_invalid(
            "sentence[{}] must have 0 <= start_sample < end_sample".format(index)
        )
    phone_raw = raw.get("phonemes")
    if not isinstance(phone_raw, list) or not phone_raw:
        raise _timeline_invalid(
            "sentence[{}].phonemes must be a non-empty list".format(index)
        )
    phonemes = [
        _parse_phoneme(item, index, phone_index, sample_rate)
        for phone_index, item in enumerate(phone_raw)
    ]
    previous_start = start
    for phone_index, phone in enumerate(phonemes):
        if phone.start_sample < start or phone.end_sample > end:
            raise _timeline_invalid(
                "sentence[{}].phonemes[{}] lies outside the sentence".format(
                    index, phone_index
                )
            )
        if phone.start_sample < previous_start or phone.end_sample <= phone.start_sample:
            raise _timeline_invalid(
                "sentence[{}].phonemes[{}] is out of order or empty".format(
                    index, phone_index
                )
            )
        previous_start = phone.start_sample
    return Sentence(
        sentence_id=sentence_id_raw.strip(),
        text=text,
        start_sample=start,
        end_sample=end,
        phonemes=phonemes,
    )


def load_timeline(path: PathLike) -> Timeline:
    source_path = Path(path)
    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LipAlignError(
            "E_INPUT", "Cannot read timeline: {}".format(source_path), exit_code=2
        ) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _json_invalid(
            "Invalid JSON at line {} column {}: {}".format(
                exc.lineno, exc.colno, exc.msg
            ),
            source_path,
        ) from exc
    if not isinstance(payload, dict):
        raise _timeline_invalid("Timeline root must be a JSON object")
    if "sample_rate" not in payload:
        raise _timeline_invalid("Timeline is missing sample_rate")
    sample_rate = _require_int(payload["sample_rate"], "sample_rate")
    if sample_rate < 8000:
        raise _timeline_invalid("sample_rate must be at least 8000")
    seed_raw = payload.get("seed", 0)
    seed = _require_int(seed_raw, "seed")
    raw_sentences = payload.get("sentences")
    if not isinstance(raw_sentences, list) or not raw_sentences:
        raise _timeline_invalid("Timeline sentences must be a non-empty list")
    sentences = [
        _parse_sentence(item, index, sample_rate)
        for index, item in enumerate(raw_sentences)
    ]
    seen_ids = set()
    previous_end = 0
    for index, sentence in enumerate(sentences):
        if sentence.sentence_id in seen_ids:
            raise _timeline_invalid(
                "Duplicate sentence id: {}".format(sentence.sentence_id)
            )
        seen_ids.add(sentence.sentence_id)
        if sentence.start_sample < previous_end:
            raise _timeline_invalid(
                "sentence[{}] overlaps or is out of order".format(index)
            )
        previous_end = sentence.end_sample

    if "duration_samples" in payload:
        duration_samples = _require_int(payload["duration_samples"], "duration_samples")
    elif "duration_sec" in payload:
        duration_samples = round_half_up(
            _require_number(payload["duration_sec"], "duration_sec") * sample_rate
        )
    else:
        duration_samples = sentences[-1].end_sample
    if duration_samples <= 0:
        raise _timeline_invalid("duration_samples must be positive")
    if sentences[-1].end_sample > duration_samples:
        raise _timeline_invalid("A sentence extends beyond duration_samples")
    return Timeline(
        source_path=source_path,
        sample_rate=sample_rate,
        duration_samples=duration_samples,
        seed=seed,
        sentences=sentences,
    )

