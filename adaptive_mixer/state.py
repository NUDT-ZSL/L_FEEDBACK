"""Resumable-render state persistence (checkpoint files)."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import MixerError

STATE_VERSION = 1


@dataclass
class Checkpoint:
    state_version: int
    config_version: str
    output_path: str
    sample_rate: int
    total_samples: int
    completed_scenes: List[str]
    completed_samples: int
    scene_reports: List[Dict[str, Any]]
    crossfades: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_version": self.state_version,
            "config_version": self.config_version,
            "output_path": self.output_path,
            "sample_rate": self.sample_rate,
            "total_samples": self.total_samples,
            "completed_scenes": list(self.completed_scenes),
            "completed_samples": self.completed_samples,
            "scene_reports": list(self.scene_reports),
            "crossfades": list(self.crossfades),
        }


def checkpoint_path(state_file) -> Path:
    return Path(state_file)


def save_checkpoint(state_file, checkpoint: Checkpoint) -> None:
    path = Path(state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2)
    try:
        descriptor, tmp_name = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_name, path)
    except OSError as exc:
        raise MixerError("E_OUTPUT_IO", str(exc),
                         {"path": str(path)}) from exc


def load_checkpoint(state_file) -> Optional[Checkpoint]:
    path = Path(state_file)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        required = ("state_version", "config_version", "output_path",
                    "completed_scenes", "completed_samples",
                    "total_samples", "sample_rate")
        for key in required:
            if key not in raw:
                raise KeyError(key)
        return Checkpoint(
            state_version=int(raw["state_version"]),
            config_version=str(raw["config_version"]),
            output_path=str(raw["output_path"]),
            sample_rate=int(raw["sample_rate"]),
            total_samples=int(raw["total_samples"]),
            completed_scenes=list(raw["completed_scenes"]),
            completed_samples=int(raw["completed_samples"]),
            scene_reports=list(raw.get("scene_reports", [])),
            crossfades=list(raw.get("crossfades", [])),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise MixerError("E_STATE_CORRUPT", f"状态文件损坏: {exc}",
                         {"path": str(path)}) from exc


def assert_compatible(checkpoint: Checkpoint, config_version: str,
                      output_path: str, sample_rate: int,
                      total_samples: int) -> None:
    """Reject stale-state checkpoints (version / layout mismatch)."""
    if checkpoint.state_version != STATE_VERSION:
        raise MixerError(
            "E_STATE_VERSION",
            f"状态文件格式版本 {checkpoint.state_version} 已过期",
            {"expected": STATE_VERSION,
             "actual": checkpoint.state_version})
    if checkpoint.config_version != config_version:
        raise MixerError(
            "E_STATE_VERSION",
            "配置已变更，旧版本断点状态必须重新混音",
            {"expected": config_version,
             "actual": checkpoint.config_version})
    if Path(checkpoint.output_path) != Path(output_path):
        raise MixerError("E_STATE_VERSION",
                         "断点状态指向的输出路径与当前任务不一致")
    if checkpoint.sample_rate != sample_rate \
            or checkpoint.total_samples != total_samples:
        raise MixerError("E_STATE_VERSION",
                         "断点状态的采样率/总长度与当前配置不一致")
