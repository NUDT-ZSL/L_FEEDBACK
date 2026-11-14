"""业务错误码定义。

错误码名称、触发条件与退出码在单文件渲染与批量渲染两条入口路径上
完全一致，由本模块单点定义。
"""

from __future__ import annotations

import json
import sys
from typing import Any, Mapping

# 错误码 -> 进程退出码（均为非零整数）
ERROR_CODES: dict[str, int] = {
    "E_INPUT_MISSING": 2,
    "E_LAYER_INVALID": 3,
    "E_MIX_MODE": 4,
    "E_UV_OUT_OF_BOUNDS": 5,
    # 防御性兜底：不属于四类业务错误的未预期运行时故障
    "E_RUNTIME": 1,
}

# 各错误码的统一触发条件说明
ERROR_CONDITIONS: dict[str, str] = {
    "E_INPUT_MISSING": "材质描述文件或其引用的贴图文件不存在/不可读",
    "E_LAYER_INVALID": "层结构非法：缺字段、权重非法、分辨率不一致或总权重非正",
    "E_MIX_MODE": "层的混合模式不在 overlay/mix/weighted 之内",
    "E_UV_OUT_OF_BOUNDS": "UV 查询在 error 包裹策略下落出 [0,1) 贴图范围",
    "E_RUNTIME": "未预期的运行时异常（防御性兜底，正常校验路径不会产生）",
}


class MaterialError(Exception):
    """带结构化业务错误码的异常。"""

    def __init__(self, code: str, message: str, details: Mapping[str, Any] | None = None):
        if code not in ERROR_CODES:
            raise KeyError(f"未登记的错误码: {code}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error_code": self.code,
            "exit_code": ERROR_CODES[self.code],
            "condition": ERROR_CONDITIONS[self.code],
            "message": self.message,
            "details": self.details,
        }

    @property
    def exit_code(self) -> int:
        return ERROR_CODES[self.code]


def emit_error(err: MaterialError, stream=sys.stderr) -> None:
    """以单行 JSON 输出结构化错误信息。"""
    json.dump(err.to_dict(), stream, ensure_ascii=False, sort_keys=True)
    stream.write("\n")
    stream.flush()
