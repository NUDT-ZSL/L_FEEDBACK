from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .errors import UsageError


@dataclass(frozen=True)
class QualityPreset:
    name: str
    shadow_resolution: int
    pcf_samples: int
    supersample: int


PRESETS: Dict[str, QualityPreset] = {
    "low": QualityPreset("low", 128, 1, 1),
    "medium": QualityPreset("medium", 256, 4, 2),
    "high": QualityPreset("high", 512, 9, 3),
}

def get_preset(name: str) -> QualityPreset:
    try:
        return PRESETS[name]
    except KeyError:
        raise UsageError(
            "unsupported quality preset %r (choose low, medium, or high)" % name
        )


def validate_block_size(block_size: int) -> int:
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise UsageError("unsupported block size %r (must be a positive integer)" % block_size)
    if block_size > 32768:
        raise UsageError("unsupported block size %r (maximum is 32768)" % block_size)
    return block_size
