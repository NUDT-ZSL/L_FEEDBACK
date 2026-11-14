"""贴图读写，全部为确定性输出（不写入时间戳等元数据）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .errors import MaterialError


def require_existing(path: str | Path, what: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise MaterialError(
            "E_INPUT_MISSING",
            f"{what}不存在或不可读: {p}",
            {"path": str(p)},
        )
    return p


def load_rgb(path: str | Path) -> np.ndarray:
    """读取为 float64 RGB，值域 [0,1]，形状 (H,W,3)。"""
    p = require_existing(path, "贴图文件")
    try:
        with Image.open(p) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    except MaterialError:
        raise
    except Exception as exc:  # 损坏文件统一归类为输入缺失/不可读
        raise MaterialError(
            "E_INPUT_MISSING",
            f"贴图无法解码: {p} ({exc})",
            {"path": str(p)},
        ) from exc
    return arr


def load_gray(path: str | Path) -> np.ndarray:
    """读取单通道（粗糙度/权重），float64，值域 [0,1]，形状 (H,W)。"""
    return load_rgb(path)[..., 0]


def load_normal_gl(path: str | Path) -> np.ndarray:
    """读取 OpenGL 风格(+Y)切线空间法线贴图并解码到单位向量。

    编码约定：RGB 存 [0,255] 的 (x,y,z)，解码为 x=2r-1, y=2g-1, z=2b-1。
    这里只做线性解码，重归一化由合成阶段统一完成。
    """
    encoded = load_rgb(path)
    return encoded * 2.0 - 1.0


def _to_uint8(values01: np.ndarray) -> np.ndarray:
    clipped = np.clip(values01, 0.0, 1.0)
    return np.rint(clipped * 255.0).astype(np.uint8)


def save_rgb_png(path: str | Path, rgb01: np.ndarray) -> None:
    """确定性保存 RGB PNG（固定压缩参数，无元数据）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(_to_uint8(rgb01), mode="RGB")
    img.save(path, format="PNG", compress_level=6, optimize=False)


def save_normal_png(path: str | Path, normals: np.ndarray) -> None:
    """把单位法线重新编码为 OpenGL(+Y) RGB PNG。"""
    encoded = normals * 0.5 + 0.5
    save_rgb_png(path, encoded)


def save_npy(path: str | Path, array: np.ndarray) -> None:
    """确定性保存浮点数组（.npy 不包含时间戳）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array, dtype=np.float32), allow_pickle=False)


def tone_map(hdr: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """Reinhard 色调映射，供反射图的 8 位预览 PNG 使用。"""
    exposed = np.clip(hdr, 0.0, None) * exposure
    return exposed / (1.0 + exposed)
