"""稳定指纹与入参指纹计算。

内容指纹只基于文件字节（不依赖修改时间）；入参指纹基于规范化 JSON。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_fingerprint(path: Path, chunk_size: int = 1 << 20) -> str:
    """对文件原始字节计算 SHA-256，结果与文件修改时间无关。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def params_fingerprint(payload: Any) -> str:
    """对任意可 JSON 序列化入参计算稳定的 SHA-256 指纹。"""
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

