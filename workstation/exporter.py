"""可恢复的多声道 WAV 分块导出管线。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from . import STATE_VERSION
from .fingerprint import params_fingerprint
from .mixer import Mixer


@dataclass
class ExportConfig:
    """导出参数。"""

    total_frames: int
    channels: int = 2
    block_size: int = 512
    subtype: str = "PCM_16"
    samplerate: int = 48000

    def fingerprint_payload(self, timeline_dict: list, tracks_dict: dict) -> dict:
        return {
            "version": STATE_VERSION,
            "total_frames": self.total_frames,
            "channels": self.channels,
            "block_size": self.block_size,
            "subtype": self.subtype,
            "samplerate": self.samplerate,
            "timeline": timeline_dict,
            "tracks": tracks_dict,
        }


class ExportManager:
    """按块写入 WAV，并把导出进度持久化到 JSON 状态文件。"""

    def __init__(
        self,
        mixer: Mixer,
        output_path: Path,
        state_path: Path,
        config: ExportConfig,
    ) -> None:
        self.mixer = mixer
        self.output_path = Path(output_path)
        self.state_path = Path(state_path)
        self.config = config
        tracks_dict = {
            name: cfg.to_dict() for name, cfg in sorted(mixer.tracks.items())
        }
        payload = config.fingerprint_payload(
            mixer.timeline.to_list(), tracks_dict
        )
        self.input_fingerprint = params_fingerprint(payload)

    def _write_state(self, frames_written: int, status: str) -> None:
        state = {
            "version": STATE_VERSION,
            "input_fingerprint": self.input_fingerprint,
            "output_path": str(self.output_path),
            "total_frames": self.config.total_frames,
            "frames_written": int(frames_written),
            "block_size": self.config.block_size,
            "channels": self.config.channels,
            "subtype": self.config.subtype,
            "samplerate": self.config.samplerate,
            "status": status,
        }
        tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.state_path)

    def _read_state(self) -> dict | None:
        if not self.state_path.exists():
            return None
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def run(self, crash_after_blocks: int | None = None) -> dict:
        """执行导出；``crash_after_blocks`` 用于在指定块数后模拟进程退出。"""
        state = self._read_state()
        resume = False
        start_frame = 0
        if state is not None:
            if state.get("input_fingerprint") != self.input_fingerprint:
                # 入参指纹变化：拒绝恢复，从头重新导出
                start_frame = 0
                if self.output_path.exists():
                    self.output_path.unlink()
            elif (
                state.get("status") == "in_progress"
                and self.output_path.exists()
            ):
                resume = True
                start_frame = int(state["frames_written"])
            else:
                start_frame = 0
                if self.output_path.exists():
                    self.output_path.unlink()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if resume and start_frame > 0:
            handle = sf.SoundFile(
                str(self.output_path), mode="r+b"
            )
            handle.seek(start_frame)
        else:
            handle = sf.SoundFile(
                str(self.output_path),
                mode="w",
                samplerate=self.config.samplerate,
                channels=self.config.channels,
                subtype=self.config.subtype,
                format="WAV",
            )

        blocks_written = 0
        frames_written = start_frame
        try:
            self._write_state(frames_written, "in_progress")
            for offset, block in self.mixer.stream(
                self.config.total_frames, self.config.block_size
            ):
                if offset < start_frame:
                    continue
                handle.write(block)
                frames_written += block.shape[0]
                blocks_written += 1
                self._write_state(frames_written, "in_progress")
                if crash_after_blocks is not None and blocks_written >= crash_after_blocks:
                    handle.flush()
                    raise _SimulatedCrash(frames_written)
        finally:
            handle.close()
        self._write_state(frames_written, "complete")
        return {
            "output_path": str(self.output_path),
            "state_path": str(self.state_path),
            "frames_written": frames_written,
            "total_frames": self.config.total_frames,
            "resumed": resume,
            "input_fingerprint": self.input_fingerprint,
        }


class _SimulatedCrash(RuntimeError):
    """测试用：写完若干块后模拟进程异常退出（状态文件已落盘）。"""

    def __init__(self, frames_written: int) -> None:
        self.frames_written = frames_written
        super().__init__(f"simulated crash after {frames_written} frames")

