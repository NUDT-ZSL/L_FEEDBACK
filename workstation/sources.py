"""素材库：WAV 解码、错误码登记与项目采样率重采样缓存。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .errors import AudioLoadError, ErrorCode, make_error_record
from .fingerprint import file_fingerprint
from .resample import resample_linear


@dataclass(frozen=True)
class Source:
    """已解码并（必要时）重采样到项目采样率的素材。"""

    source_id: str
    path: Path
    fingerprint: str
    original_rate: int
    project_rate: int
    frames: np.ndarray  # 形状 (m,) 或 (m, c)，float64，已按项目采样率重采样

    @property
    def length(self) -> int:
        return int(self.frames.shape[0])

    @property
    def channels(self) -> int:
        return 1 if self.frames.ndim == 1 else int(self.frames.shape[1])


class SourceLibrary:
    """按 source_id 管理素材，单素材失败只登记错误、不中断整体流程。"""

    def __init__(self, project_rate: int, base_dir: Path) -> None:
        self.project_rate = int(project_rate)
        self.base_dir = Path(base_dir)
        self._sources: dict[str, Source] = {}
        self.errors: list[dict] = []

    def resolve(self, raw_path: str | Path) -> Path:
        """相对路径相对工程根目录解析，绝对路径原样返回。"""
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.base_dir / path
        return path

    def load(self, source_id: str, raw_path: str | Path) -> Source | None:
        """加载单个素材；失败时登记 E_MISSING / E_FORMAT / E_SHORT 并返回 None。"""
        path = self.resolve(raw_path)
        try:
            if not path.exists():
                raise AudioLoadError(path, ErrorCode.E_MISSING, "文件不存在")
            try:
                info = sf.info(str(path))
            except Exception as exc:  # soundfile 对不支持/损坏格式抛异常
                raise AudioLoadError(path, ErrorCode.E_FORMAT, str(exc)) from exc
            if info.frames <= 0:
                raise AudioLoadError(path, ErrorCode.E_SHORT, "解码后帧数为零")
            frames, original_rate = sf.read(str(path), always_2d=False, dtype="float64")
            if frames.size == 0 or frames.shape[0] == 0:
                raise AudioLoadError(path, ErrorCode.E_SHORT, "解码后帧数为零")
            resampled = resample_linear(
                np.asarray(frames, dtype=np.float64),
                int(original_rate),
                self.project_rate,
            )
            source = Source(
                source_id=source_id,
                path=path,
                fingerprint=file_fingerprint(path),
                original_rate=int(original_rate),
                project_rate=self.project_rate,
                frames=resampled,
            )
        except AudioLoadError as exc:
            self.errors.append(make_error_record(exc.path, exc.code, exc.detail))
            return None
        self._sources[source_id] = source
        return source

    def get(self, source_id: str) -> Source | None:
        return self._sources.get(source_id)

    def items(self):
        return self._sources.items()

    def fingerprints(self) -> dict[str, str]:
        return {sid: src.fingerprint for sid, src in self._sources.items()}

    def error_codes(self) -> list[str]:
        return [record["code"] for record in self.errors]

