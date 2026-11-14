"""Deterministic byte-order-mark based encoding detection.

The supported encodings are UTF-8 (with or without BOM), UTF-16 LE and
UTF-16 BE. BOM takes precedence; files without a BOM are decoded as UTF-8,
which also covers plain ASCII.
"""

from __future__ import annotations

from dataclasses import dataclass

UTF8_BOM = b"\xef\xbb\xbf"
UTF16_LE_BOM = b"\xff\xfe"
UTF16_BE_BOM = b"\xfe\xff"


@dataclass(frozen=True)
class EncodingResult:
    encoding: str
    codec: str
    has_bom: bool


def detect_encoding(raw: bytes) -> EncodingResult:
    """Detect the text encoding of ``raw`` from its leading bytes.

    UTF-16 without a BOM is ambiguous and is not part of the contract, so a
    missing BOM deterministically selects UTF-8.
    """

    if raw.startswith(UTF8_BOM):
        return EncodingResult("utf-8-sig", "utf-8", True)
    if raw.startswith(UTF16_LE_BOM):
        return EncodingResult("utf-16-le", "utf-16-le", True)
    if raw.startswith(UTF16_BE_BOM):
        return EncodingResult("utf-16-be", "utf-16-be", True)
    return EncodingResult("utf-8", "utf-8", False)


def decode_bytes(raw: bytes) -> tuple[str, EncodingResult]:
    """Decode ``raw`` and return the text together with detection metadata."""

    result = detect_encoding(raw)
    if result.has_bom:
        if result.encoding == "utf-8-sig":
            text = raw[3:].decode("utf-8")
        else:
            text = raw[2:].decode(result.codec)
        return text, result
    return raw.decode(result.codec), result
