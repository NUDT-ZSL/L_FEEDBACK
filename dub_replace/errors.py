"""Business error taxonomy.

Every recoverable/expected failure carries a stable business code plus an
actionable hint so callers can surface deterministic error categories.
"""

from __future__ import annotations


class DubReplaceError(Exception):
    """Base class for all classified business errors."""

    code: str = "E_INTERNAL"
    default_hint: str = "请检查输入数据后重试；若问题持续存在请保留日志联系维护人员。"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint or self.default_hint

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "hint": self.hint}


class MetadataError(DubReplaceError):
    code = "E_METADATA"
    default_hint = "请对照元数据 schema 修正动画 meta.json 后重新运行。"


class MissingAudioError(DubReplaceError):
    code = "E_MISSING_AUDIO"
    default_hint = "请补齐缺失的 WAV 文件，或修正元数据中的相对路径后重新运行。"


class SampleRateError(DubReplaceError):
    code = "E_SAMPLE_RATE"
    default_hint = "请按元数据声明的采样率重新导出 WAV，或使用 FFmpeg (-ar) 重采样后重试。"


class ChannelLayoutError(DubReplaceError):
    code = "E_CHANNEL"
    default_hint = "请核对元数据 channel_layout/channels 声明与实际 WAV 声道数后重新导出。"


class OverlapError(DubReplaceError):
    code = "E_OVERLAP"
    default_hint = "请修正元数据 dialogue 中各句的 start_ms/end_ms，保证片段互不重叠。"


class StretchLimitError(DubReplaceError):
    """Raised as a classified event (non-fatal) when a ratio is clamped."""

    code = "E_STRETCH"
    default_hint = "可放宽 stretch_limits 范围，或重新剪辑方言片段使其时长接近原句。"


class CorruptAudioError(DubReplaceError):
    code = "E_CORRUPT"
    default_hint = "文件无法解析，建议用 `ffmpeg -v error -i file.wav -f null -` 校验并重新导出 PCM WAV。"


class FFmpegUnavailableError(DubReplaceError):
    code = "E_FFMPEG"
    default_hint = "未在 PATH 中找到 FFmpeg，请安装并将其 bin 目录加入 PATH 后重新运行。"


class InternalError(DubReplaceError):
    code = "E_INTERNAL"
    default_hint = "出现未预期的内部错误，请保留日志与样本复现后联系维护人员。"
