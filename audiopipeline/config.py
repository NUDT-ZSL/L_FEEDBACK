"""Pipeline configuration with range validation.

A :class:`Config` object holds every tunable parameter.  Defaults match the
business requirements (48 kHz stereo, -16 LUFS platform target, -1 dBTP true
peak ceiling, 10 ms cosine crossfades, 400 ms / 75 % loudness windows).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields

from .errors import PipelineError

SAMPLE_RATE = 48_000
CHANNELS = 2

# (min, max) inclusive ranges used by :meth:`Config.validate`.
RANGES: dict[str, tuple[float, float]] = {
    "crossfade_ms": (0.1, 500.0),
    "tail_fade_ms": (0.0, 500.0),
    "head_fade_ms": (0.0, 500.0),
    "loudness_block_ms": (100.0, 3000.0),
    "loudness_overlap": (0.0, 0.9),
    "absolute_gate_lufs": (-100.0, 0.0),
    "relative_gate_offset_lu": (-20.0, 20.0),
    "target_integrated_lufs": (-40.0, 0.0),
    "true_peak_limit_dbtp": (-10.0, 0.0),
    "max_gain_db": (-12.0, 40.0),
    "tp_upsample_factor": (1.0, 8.0),
}


@dataclass
class Config:
    # mixing / fades ----------------------------------------------------
    overlap_strategy: str = "replace"          # "replace" | "mix"
    crossfade_ms: float = 10.0                 # loop-point crossfade
    tail_fade_ms: float = 10.0                 # tail fade after trim
    head_fade_ms: float = 0.0                  # optional leading fade-in
    # loudness ----------------------------------------------------------
    target_integrated_lufs: float = -16.0      # platform master target
    true_peak_limit_dbtp: float = -1.0
    category_targets_lufs: dict[str, float] = field(
        default_factory=lambda: {
            "dialog": -18.0,
            "sfx": -20.0,
            "music": -22.0,
        }
    )
    loudness_block_ms: float = 400.0
    loudness_overlap: float = 0.75             # 75 % overlap
    absolute_gate_lufs: float = -70.0
    relative_gate_offset_lu: float = -10.0
    max_gain_db: float = 20.0
    # true peak ---------------------------------------------------------
    tp_upsample_factor: float = 4.0

    def validate(self) -> None:
        """Raise :class:`PipelineError` (``E_CONFIG_INVALID``) on bad values."""
        for name, (lo, hi) in RANGES.items():
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise PipelineError(
                    "E_CONFIG_INVALID", f"参数 {name} 必须是数值，实际为 {type(value).__name__}"
                )
            if not (lo <= float(value) <= hi):
                raise PipelineError(
                    "E_CONFIG_INVALID",
                    f"参数 {name}={value} 超出合法范围 [{lo}, {hi}]",
                )
        if self.overlap_strategy not in ("replace", "mix"):
            raise PipelineError(
                "E_CONFIG_INVALID",
                f"overlap_strategy 仅支持 replace/mix，实际为 {self.overlap_strategy!r}",
            )
        targets = self.category_targets_lufs
        if not isinstance(targets, dict) or not targets:
            raise PipelineError("E_CONFIG_INVALID", "category_targets_lufs 必须是非空映射")
        for cat, lufs in targets.items():
            if not isinstance(lufs, (int, float)) or not (-40.0 <= float(lufs) <= 0.0):
                raise PipelineError(
                    "E_CONFIG_INVALID",
                    f"分类目标响度 {cat}={lufs} 超出合法范围 [-40, 0] LUFS",
                )

    @classmethod
    def from_file(cls, path: str) -> "Config":
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except FileNotFoundError:
            raise PipelineError("E_CONFIG_INVALID", f"配置文件不存在: {path}")
        except json.JSONDecodeError as exc:
            raise PipelineError("E_CONFIG_INVALID", f"配置文件不是合法 JSON: {exc}")
        if not isinstance(raw, dict):
            raise PipelineError("E_CONFIG_INVALID", "配置文件顶层必须是对象")
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise PipelineError("E_CONFIG_INVALID", f"配置包含未知参数: {sorted(unknown)}")
        cfg = cls(**{k: v for k, v in raw.items() if k in known})
        cfg.validate()
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
