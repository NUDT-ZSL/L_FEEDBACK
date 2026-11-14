"""YAML 材质描述解析与层结构校验。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .errors import MaterialError

VALID_MODES = ("overlay", "mix", "weighted")
# 允许中文别名，统一映射到内部规范名
MODE_ALIASES = {
    "overlay": "overlay",
    "mix": "mix",
    "weighted": "weighted",
    "叠加": "overlay",
    "混合": "mix",
    "加权": "weighted",
}
VALID_WRAP = ("error", "clamp", "repeat")


@dataclass
class LayerSpec:
    index: int
    name: str
    normal_path: Path
    color_path: Path
    roughness_path: Path
    weight: float
    mode: str
    uv_scale: tuple[float, float]
    uv_offset: tuple[float, float]


@dataclass
class MaterialSpec:
    source_path: Path
    material_name: str
    width: int
    height: int
    tangent_threshold_deg: float
    uv_wrap: str
    view_dir: np.ndarray
    lights: list[np.ndarray]
    layers: list[LayerSpec]
    uv_checks: list[tuple[float, float]] = field(default_factory=list)


def _as_vec3(value: Any, field_name: str, ctx: dict[str, Any]) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise MaterialError(
            "E_LAYER_INVALID",
            f"字段 {field_name} 必须是长度为 3 的向量",
            ctx,
        )
    try:
        vec = np.asarray([float(v) for v in value], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MaterialError("E_LAYER_INVALID", f"字段 {field_name} 含非数值", ctx) from exc
    if not np.isfinite(vec).all():
        raise MaterialError("E_LAYER_INVALID", f"字段 {field_name} 含非有限值", ctx)
    return vec


def _as_uv_pair(value: Any, field_name: str, ctx: dict[str, Any]) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise MaterialError("E_LAYER_INVALID", f"字段 {field_name} 必须是长度为 2 的 UV 对", ctx)
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise MaterialError("E_LAYER_INVALID", f"字段 {field_name} 含非数值", ctx) from exc


def _resolve_mode(raw_mode: Any, ctx: dict[str, Any]) -> str:
    if not isinstance(raw_mode, str):
        raise MaterialError("E_MIX_MODE", "混合模式必须是字符串", ctx)
    mode = MODE_ALIASES.get(raw_mode.strip().lower())
    if mode is None:
        raise MaterialError(
            "E_MIX_MODE",
            f"未知混合模式: {raw_mode!r}，支持 overlay/mix/weighted",
            {**ctx, "raw_mode": raw_mode, "valid_modes": list(VALID_MODES)},
        )
    return mode


def parse_material(path: str | Path) -> MaterialSpec:
    """解析并校验材质描述文件，所有业务错误在此抛出 MaterialError。"""
    p = Path(path)
    if not p.is_file():
        raise MaterialError("E_INPUT_MISSING", f"材质描述文件不存在: {p}", {"path": str(p)})
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise MaterialError(
            "E_LAYER_INVALID",
            f"YAML 解析失败: {exc}",
            {"path": str(p)},
        ) from exc
    if not isinstance(raw, dict):
        raise MaterialError("E_LAYER_INVALID", "材质描述根节点必须是映射", {"path": str(p)})

    base_dir = p.parent
    name = str(raw.get("name", p.stem))
    resolution = raw.get("resolution")
    if (
        not isinstance(resolution, (list, tuple))
        or len(resolution) != 2
        or not all(isinstance(v, int) and v > 0 for v in resolution)
    ):
        raise MaterialError(
            "E_LAYER_INVALID",
            "resolution 必须是两个正整数 [width, height]",
            {"path": str(p)},
        )
    width, height = int(resolution[0]), int(resolution[1])

    threshold = float(raw.get("tangent_threshold_deg", 30.0))
    if not np.isfinite(threshold) or threshold <= 0.0 or threshold >= 180.0:
        raise MaterialError(
            "E_LAYER_INVALID",
            "tangent_threshold_deg 必须在 (0,180) 度之间",
            {"path": str(p)},
        )

    uv_wrap = str(raw.get("uv_wrap", "error")).strip().lower()
    if uv_wrap not in VALID_WRAP:
        raise MaterialError(
            "E_LAYER_INVALID",
            f"uv_wrap 必须是 {VALID_WRAP} 之一",
            {"path": str(p), "uv_wrap": uv_wrap},
        )

    view_dir = _as_vec3(raw.get("view_dir", [0.0, 0.0, 1.0]), "view_dir", {"path": str(p)})
    if np.linalg.norm(view_dir) < 1e-12:
        raise MaterialError("E_LAYER_INVALID", "view_dir 不能为零向量", {"path": str(p)})

    raw_lights = raw.get("lights", [])
    if not isinstance(raw_lights, list) or not raw_lights:
        raise MaterialError("E_LAYER_INVALID", "lights 必须是非空方向列表", {"path": str(p)})
    lights = [
        _as_vec3(item, f"lights[{i}]", {"path": str(p)}) for i, item in enumerate(raw_lights)
    ]
    for i, light in enumerate(lights):
        if np.linalg.norm(light) < 1e-12:
            raise MaterialError(
                "E_LAYER_INVALID", f"lights[{i}] 不能为零向量", {"path": str(p)}
            )

    raw_layers = raw.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise MaterialError("E_LAYER_INVALID", "layers 必须是非空层列表", {"path": str(p)})

    layers: list[LayerSpec] = []
    for i, raw_layer in enumerate(raw_layers):
        ctx = {"path": str(p), "layer_index": i}
        if not isinstance(raw_layer, dict):
            raise MaterialError("E_LAYER_INVALID", "每个层必须是映射", ctx)
        layer_name = str(raw_layer.get("name", f"layer_{i}"))
        ctx["layer_name"] = layer_name
        for key in ("normal", "base_color", "roughness"):
            if not isinstance(raw_layer.get(key), str):
                raise MaterialError(
                    "E_LAYER_INVALID", f"层缺少字符串贴图字段 {key}", ctx
                )
        try:
            weight = float(raw_layer.get("weight", 1.0))
        except (TypeError, ValueError) as exc:
            raise MaterialError("E_LAYER_INVALID", "weight 必须是数值", ctx) from exc
        if not np.isfinite(weight) or weight < 0.0:
            raise MaterialError("E_LAYER_INVALID", "weight 必须是非负有限值", ctx)

        mode = _resolve_mode(raw_layer.get("mode"), ctx)
        uv_scale = _as_uv_pair(raw_layer.get("uv_scale", [1.0, 1.0]), "uv_scale", ctx)
        uv_offset = _as_uv_pair(raw_layer.get("uv_offset", [0.0, 0.0]), "uv_offset", ctx)

        layers.append(
            LayerSpec(
                index=i,
                name=layer_name,
                normal_path=(base_dir / raw_layer["normal"]).resolve(),
                color_path=(base_dir / raw_layer["base_color"]).resolve(),
                roughness_path=(base_dir / raw_layer["roughness"]).resolve(),
                weight=weight,
                mode=mode,
                uv_scale=uv_scale,
                uv_offset=uv_offset,
            )
        )

    uv_checks: list[tuple[float, float]] = []
    for item in raw.get("uv_checks", []) or []:
        uv_checks.append(_as_uv_pair(item, "uv_checks", {"path": str(p)}))

    return MaterialSpec(
        source_path=p.resolve(),
        material_name=name,
        width=width,
        height=height,
        tangent_threshold_deg=threshold,
        uv_wrap=uv_wrap,
        view_dir=view_dir,
        lights=lights,
        layers=layers,
        uv_checks=uv_checks,
    )
