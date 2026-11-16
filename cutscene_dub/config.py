from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .errors import BusinessError


@dataclass(frozen=True)
class RenderConfig:
    silence_threshold_dbfs: float = -45.0
    silence_margin_ms: float = 5.0
    max_speedup: float = 1.15
    duck_depth_db: float = 16.0
    duck_fade_in_ms: float = 120.0
    duck_fade_out_ms: float = 120.0
    max_gain_jump_db: float = 0.5
    peak_ceiling_dbfs: float = -1.0
    dialogue_gain_db: float = 0.0
    music_gain_db: float = 0.0
    ffmpeg_bin: str = "ffmpeg"

    def validate(self) -> None:
        checks = [
            (
                "silence_threshold_dbfs",
                self.silence_threshold_dbfs,
                -120.0,
                -1.0,
                True,
            ),
            ("silence_margin_ms", self.silence_margin_ms, 0.0, 1000.0, True),
            ("max_speedup", self.max_speedup, 1.0, 4.0, True),
            ("duck_depth_db", self.duck_depth_db, 0.0, 60.0, True),
            ("duck_fade_in_ms", self.duck_fade_in_ms, 0.0, 10_000.0, True),
            ("duck_fade_out_ms", self.duck_fade_out_ms, 0.0, 10_000.0, True),
            ("max_gain_jump_db", self.max_gain_jump_db, 0.000001, 24.0, True),
            ("peak_ceiling_dbfs", self.peak_ceiling_dbfs, -60.0, -1.0, True),
            ("dialogue_gain_db", self.dialogue_gain_db, -60.0, 24.0, True),
            ("music_gain_db", self.music_gain_db, -60.0, 24.0, True),
        ]
        for name, value, minimum, maximum, inclusive in checks:
            if not math.isfinite(value):
                raise BusinessError(
                    code="E_CONFIG",
                    message=f"{name} must be finite.",
                    path=f"--{name.replace('_', '-')}",
                )
            if inclusive and not (minimum <= value <= maximum):
                raise BusinessError(
                    code="E_CONFIG",
                    message=f"{name} must be between {minimum} and {maximum}.",
                    path=f"--{name.replace('_', '-')}",
                    details={"value": value},
                )
        if not self.ffmpeg_bin.strip():
            raise BusinessError(
                code="E_CONFIG",
                message="ffmpeg_bin must not be empty.",
                path="--ffmpeg-bin",
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def peak_ceiling_linear(self) -> float:
        return 10.0 ** (self.peak_ceiling_dbfs / 20.0)

