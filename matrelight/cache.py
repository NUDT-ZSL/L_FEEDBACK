"""可重光照材质缓存（.mzc）。

缓存是一个确定性 ZIP：JSON 元数据（排序键、固定分隔符）+ 原始 float32
数组。所有条目使用固定 ZIP 时间戳 (1980-01-01) 与固定压缩级别，因此
同输入同种子逐字节一致，不依赖 np.savez 的内部时间戳行为。
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np

CACHE_VERSION = "1.0"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def _write_fixed_zip(path: Path, entries: list[tuple[str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name, payload in sorted(entries):
            info = zipfile.ZipInfo(filename=name, date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o644 & 0xFFFF) << 16
            info.create_system = 3
            zf.writestr(info, payload)


def _array_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return buffer.getvalue()


def write_cache(
    path: str | Path,
    metadata: dict,
    arrays: dict[str, np.ndarray],
) -> None:
    payload = {
        "cache_version": CACHE_VERSION,
        **metadata,
        "array_order": sorted(arrays.keys()),
    }
    json_bytes = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    entries: list[tuple[str, bytes]] = [("metadata.json", json_bytes)]
    for name in sorted(arrays.keys()):
        entries.append((f"arrays/{name}.npy", _array_bytes(arrays[name])))
    _write_fixed_zip(Path(path), entries)


def read_cache(path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    """读回缓存，主要供测试与后续引擎复用。"""
    arrays: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(path, mode="r") as zf:
        metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
        for name in metadata["array_order"]:
            with zf.open(f"arrays/{name}.npy") as fh:
                arrays[name] = np.load(io.BytesIO(fh.read()), allow_pickle=False)
    return metadata, arrays
