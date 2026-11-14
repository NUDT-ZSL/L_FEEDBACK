"""Unicode normalization and text similarity helpers."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

ASS_OVERRIDE_TAG = re.compile(r"\\[Nnh]")
ASS_DRAWING_TAG = re.compile(r"\{[^}]*\}")
WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Apply NFC normalization while leaving the original cue text untouched.

    Override blocks (``{...}``) and line breaks are removed only for the
    comparison copy; callers keep the raw string for output.
    """

    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", text)
    normalized = ASS_DRAWING_TAG.sub(" ", normalized)
    normalized = ASS_OVERRIDE_TAG.sub(" ", normalized)
    normalized = normalized.replace("\r", " ").replace("\n", " ")
    normalized = WHITESPACE.sub(" ", normalized).strip()
    return normalized


def text_similarity(left: str, right: str) -> float:
    """Return a deterministic similarity in ``[0, 1]`` based on NFC text."""

    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm and not right_norm:
        return 0.0
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm, autojunk=False).ratio()
