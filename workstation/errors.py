"""稳定可枚举的业务错误码与异常定义。"""

from __future__ import annotations

from enum import Enum
from pathlib import Path


class ErrorCode(str, Enum):
    """业务错误码，取值为稳定字符串，便于在 JSON 报告中枚举。"""

    E_MISSING = "E_MISSING"
    E_FORMAT = "E_FORMAT"
    E_SHORT = "E_SHORT"


class AudioLoadError(Exception):
    """素材加载失败，携带素材路径与稳定错误码。"""

    def __init__(self, path: Path, code: ErrorCode, detail: str = "") -> None:
        self.path = Path(path)
        self.code = ErrorCode(code)
        self.detail = detail
        super().__init__(f"{self.code.value}: {self.path} ({detail})")


def make_error_record(path: Path, code: ErrorCode | str, detail: str = "") -> dict:
    """构造写入 JSON 报告的错误记录。"""
    return {
        "path": str(Path(path)),
        "code": ErrorCode(code).value,
        "detail": detail,
    }

