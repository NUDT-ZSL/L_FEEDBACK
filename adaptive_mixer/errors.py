"""Structured error codes (all error codes use the ``E_`` prefix)."""

from __future__ import annotations

from typing import Any, Dict, Optional


ERROR_CODES: Dict[str, str] = {
    "E_CONFIG_PARSE": "场景配置 JSON 解析失败",
    "E_CONFIG_MISSING_FIELD": "场景配置缺少必需字段",
    "E_CONFIG_VALUE": "场景配置字段取值非法",
    "E_SAMPLE_RATE_MISMATCH": "帧率/采样率与音频文件不一致",
    "E_TRACK_NOT_FOUND": "分轨 WAV 文件不存在或不可读",
    "E_TRACK_FORMAT": "分轨 WAV 格式不符合要求",
    "E_TRACK_LENGTH": "分轨长度与场景时间轴不一致",
    "E_SCENE_OVERLAP": "场景区间重叠或不连续",
    "E_CROSSFADE_RANGE": "交叉淡化时长超出合法区间",
    "E_LUFS_TARGET": "响度目标超出支持范围",
    "E_STATE_VERSION": "断点状态版本与当前配置不匹配",
    "E_STATE_CORRUPT": "断点状态文件损坏",
    "E_OUTPUT_IO": "输出文件读写失败",
    "E_LIMITER": "峰值限制后仍未达到 -1 dBTP",
    "E_INTERNAL": "引擎内部错误",
}


class MixerError(Exception):
    """Exception carrying a stable, prefixed machine-readable error code."""

    def __init__(
        self,
        code: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not code.startswith("E_"):
            code = "E_INTERNAL"
        self.code = code
        self.details = dict(details or {})
        full_message = message or ERROR_CODES.get(code, code)
        super().__init__(full_message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": True,
            "code": self.code,
            "message": str(self),
            "details": self.details,
        }
