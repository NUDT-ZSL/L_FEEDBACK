"""最近邻 UV 采样，支持 error/clamp/repeat 三种包裹策略。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .errors import MaterialError
from .image_io import load_normal_gl, load_rgb
from .material import LayerSpec


class LayerTextures:
    """单层已加载贴图，全部 float64。"""

    def __init__(self, spec: LayerSpec, width: int, height: int):
        self.spec = spec
        self.normal = load_normal_gl(spec.normal_path)
        self.color = load_rgb(spec.color_path)
        self.roughness = load_rgb(spec.roughness_path)[..., 0]
        for label, arr in (
            ("normal", self.normal),
            ("base_color", self.color),
            ("roughness", self.roughness),
        ):
            if arr.shape[0] != height or arr.shape[1] != width:
                raise MaterialError(
                    "E_LAYER_INVALID",
                    f"层 {spec.name} 的 {label} 分辨率 {arr.shape[1]}x{arr.shape[0]} "
                    f"与目标 {width}x{height} 不一致",
                    {"layer": spec.name, "field": label},
                )


def _pixel_coords(u: float, v: float, width: int, height: int):
    x = int(np.floor(u * width))
    y = int(np.floor(v * height))
    return x, y


def resolve_pixel(
    u: float,
    v: float,
    width: int,
    height: int,
    wrap: str,
    context: dict,
) -> tuple[int, int]:
    """把 UV 解析成像素坐标；error 模式下越界抛 E_UV_OUT_OF_BOUNDS。"""
    if not (np.isfinite(u) and np.isfinite(v)):
        raise MaterialError("E_UV_OUT_OF_BOUNDS", "UV 含非有限值", context)
    in_range = (0.0 <= u < 1.0) and (0.0 <= v < 1.0)
    if in_range:
        return _pixel_coords(u, v, width, height)
    if wrap == "error":
        raise MaterialError(
            "E_UV_OUT_OF_BOUNDS",
            f"UV ({u},{v}) 落出 [0,1) 贴图范围且 uv_wrap=error",
            {**context, "u": u, "v": v},
        )
    if wrap == "clamp":
        uc = min(max(u, 0.0), 1.0 - 1e-9)
        vc = min(max(v, 0.0), 1.0 - 1e-9)
        return _pixel_coords(uc, vc, width, height)
    # repeat：包裹回 [0,1)
    ur = u - np.floor(u)
    vr = v - np.floor(v)
    return _pixel_coords(ur, vr, width, height)


def sample_uv(textures: list[LayerTextures], material) -> list[dict]:
    """执行材质描述中的显式 UV 查询，结果同时用于缓存与越界验证。"""
    results: list[dict] = []
    for u, v in material.uv_checks:
        x, y = resolve_pixel(
            u,
            v,
            material.width,
            material.height,
            material.uv_wrap,
            {"uv": [u, v]},
        )
        entry = {"uv": [u, v], "pixel": [x, y], "layers": []}
        for tex in textures:
            spec: LayerSpec = tex.spec
            tu = u * spec.uv_scale[0] + spec.uv_offset[0]
            tv = v * spec.uv_scale[1] + spec.uv_offset[1]
            lx, ly = resolve_pixel(
                tu,
                tv,
                material.width,
                material.height,
                material.uv_wrap,
                {"uv": [u, v], "layer": spec.name, "transformed_uv": [tu, tv]},
            )
            entry["layers"].append(
                {
                    "layer": spec.name,
                    "transformed_uv": [tu, tv],
                    "pixel": [lx, ly],
                    "normal": tex.normal[ly, lx].tolist(),
                    "base_color": tex.color[ly, lx].tolist(),
                    "roughness": float(tex.roughness[ly, lx]),
                }
            )
        results.append(entry)
    return results
