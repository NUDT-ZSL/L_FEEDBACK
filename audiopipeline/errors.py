"""Business error codes for the audio alignment pipeline.

Every business error code uses the ``E_`` prefix and is collected in
:data:`ERROR_CATALOG`.  Errors are split in two severity levels:

* ``fatal``  -> the whole run aborts with a non-zero process exit code.
* ``event``  -> the offending event is skipped, other events keep processing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorDef:
    code: str
    severity: str  # "fatal" | "event"
    message: str


ERROR_CATALOG: dict[str, ErrorDef] = {
    # ---- fatal / manifest level ----------------------------------------
    "E_MANIFEST_PARSE": ErrorDef(
        "E_MANIFEST_PARSE", "fatal", "事件清单无法解析，不是合法 JSON 或顶层结构错误"
    ),
    "E_MANIFEST_FIELD": ErrorDef(
        "E_MANIFEST_FIELD", "fatal", "清单缺少必填字段(如 events / total_duration_ms)或类型错误"
    ),
    "E_DURATION_INVALID": ErrorDef(
        "E_DURATION_INVALID", "fatal", "清单声明的总时长非法(缺失、非正数或非数值)"
    ),
    "E_CONFIG_INVALID": ErrorDef(
        "E_CONFIG_INVALID", "fatal", "配置文件参数缺失或超出合法范围"
    ),
    # ---- per-event, recoverable ----------------------------------------
    "E_FIELD_MISSING": ErrorDef(
        "E_FIELD_MISSING", "event", "事件缺少必填字段(id / path / start_ms)"
    ),
    "E_TIME_INVALID": ErrorDef(
        "E_TIME_INVALID", "event", "事件时间参数非法(负数、非数值或偏移超出素材)"
    ),
    "E_FILE_NOT_FOUND": ErrorDef(
        "E_FILE_NOT_FOUND", "event", "事件引用的音频文件不存在"
    ),
    "E_FORMAT_UNSUPPORTED": ErrorDef(
        "E_FORMAT_UNSUPPORTED", "event", "不支持的音频格式(仅支持 WAV / FLAC)"
    ),
    "E_DECODE_FAILED": ErrorDef(
        "E_DECODE_FAILED", "event", "FFmpeg 解码失败或文件损坏"
    ),
    "E_LOUDNESS_FAILED": ErrorDef(
        "E_LOUDNESS_FAILED", "event", "响度分析失败(素材为静音或测量异常)"
    ),
}


class PipelineError(Exception):
    """Fatal pipeline error carrying an :data:`ERROR_CATALOG` code."""

    def __init__(self, code: str, reason: str = ""):
        if code not in ERROR_CATALOG:
            raise KeyError(f"unknown error code {code!r}")
        super().__init__(f"{code}: {reason or ERROR_CATALOG[code].message}")
        self.code = code
        self.reason = reason


def catalog_as_list() -> list[dict[str, str]]:
    """Serialise the error catalog for inclusion in reports."""
    return [
        {"code": d.code, "severity": d.severity, "message": d.message}
        for d in sorted(ERROR_CATALOG.values(), key=lambda d: d.code)
    ]
