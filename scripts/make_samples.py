"""生成本地确定性样本：贴图、材质 YAML 与光照列表。

不使用任何外部素材。所有贴图由固定种子的 numpy RNG 程序化生成，
重复运行逐字节一致。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _save_png(path: Path, rgb01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.rint(np.clip(rgb01, 0, 1) * 255).astype(np.uint8), "RGB")
    img.save(path, format="PNG", compress_level=6, optimize=False)


def _grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float64)
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float64)
    return np.meshgrid(xs, ys)


def make_normal_map(path: Path, width: int, height: int, kind: str, seed: int) -> None:
    gx, gy = _grid(width, height)
    if kind == "base":
        nx = 0.15 * np.sin(3.0 * gx)
        ny = 0.15 * np.cos(3.0 * gy)
    elif kind == "detail":
        nx = 0.35 * np.sign(np.sin(8.0 * gx))
        ny = 0.30 * np.sign(np.sin(8.0 * gy))
    elif kind == "noise":
        rng = np.random.default_rng(seed)
        nx = rng.normal(0.0, 0.25, size=(height, width))
        ny = rng.normal(0.0, 0.25, size=(height, width))
    elif kind == "seam":
        # 右半方向符号翻转，制造一条垂直切线不连续接缝
        nx = np.tile(np.linspace(0.2, 1.0, width), (height, 1))
        nx[:, width // 2:] = -nx[:, width // 2:]
        nx *= 0.6
        ny = np.zeros((height, width))
    else:
        raise ValueError(kind)
    nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 1e-6, 1.0))
    normal = np.stack([nx, ny, nz], axis=-1)
    encoded = normal * 0.5 + 0.5
    _save_png(path, encoded)


def make_color_map(path: Path, width: int, height: int, color: tuple[float, float, float]) -> None:
    gx, gy = _grid(width, height)
    shade = 0.85 + 0.15 * ((gx + gy) * 0.5)
    rgb = np.zeros((height, width, 3), dtype=np.float64)
    for c in range(3):
        rgb[..., c] = np.clip(color[c] * shade, 0, 1)
    _save_png(path, rgb)


def make_roughness_map(path: Path, width: int, height: int, low: float, high: float) -> None:
    gx, gy = _grid(width, height)
    value = (low + high) * 0.5 + (high - low) * 0.5 * np.sin(5.0 * gx) * np.cos(5.0 * gy)
    rgb = np.repeat(value[..., None], 3, axis=-1)
    _save_png(path, rgb)


def _write_yaml(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build(root: Path, size: int = 64) -> None:
    tex = root / "textures"
    for layer, kind, seed in (
        ("base", "base", 101),
        ("detail", "detail", 202),
        ("noise", "noise", 303),
    ):
        make_normal_map(tex / f"normal_{layer}.png", size, size, kind, seed)
    make_normal_map(tex / "normal_seam.png", size, size, "seam", 404)
    make_color_map(tex / "color_base.png", size, size, (0.72, 0.25, 0.18))
    make_color_map(tex / "color_detail.png", size, size, (0.20, 0.55, 0.75))
    make_color_map(tex / "color_noise.png", size, size, (0.35, 0.65, 0.30))
    make_roughness_map(tex / "rough_base.png", size, size, 0.35, 0.55)
    make_roughness_map(tex / "rough_detail.png", size, size, 0.15, 0.30)
    make_roughness_map(tex / "rough_noise.png", size, size, 0.55, 0.85)

    common_header = f"""resolution: [{size}, {size}]
tangent_threshold_deg: 30.0
uv_wrap: error
view_dir: [0.0, 0.0, 1.0]
lights:
  - [0.5, 0.6, 0.8]
  - [-0.7, 0.2, 0.6]
  - [0.0, -0.8, 0.5]
uv_checks:
  - [0.25, 0.25]
  - [0.75, 0.50]
"""

    _write_yaml(
        root / "material_mixed.yaml",
        common_header
        + """name: mixed_material
layers:
  - name: base
    normal: textures/normal_base.png
    base_color: textures/color_base.png
    roughness: textures/rough_base.png
    weight: 1.0
    mode: overlay
  - name: detail
    normal: textures/normal_detail.png
    base_color: textures/color_detail.png
    roughness: textures/rough_detail.png
    weight: 0.6
    mode: mix
  - name: noise
    normal: textures/normal_noise.png
    base_color: textures/color_noise.png
    roughness: textures/rough_noise.png
    weight: 0.3
    mode: overlay
""",
    )

    _write_yaml(
        root / "material_weighted.yaml",
        common_header
        + """name: weighted_material
layers:
  - name: base
    normal: textures/normal_base.png
    base_color: textures/color_base.png
    roughness: textures/rough_base.png
    weight: 2.0
    mode: weighted
  - name: detail
    normal: textures/normal_detail.png
    base_color: textures/color_detail.png
    roughness: textures/rough_detail.png
    weight: 1.0
    mode: weighted
  - name: detail_duplicate
    normal: textures/normal_detail.png
    base_color: textures/color_detail.png
    roughness: textures/rough_detail.png
    weight: 1.0
    mode: weighted
""",
    )

    # 低阈值 + 接缝法线，端到端真实触发切线连续性修正
    seam_header = f"""resolution: [{size}, {size}]
tangent_threshold_deg: 5.0
uv_wrap: error
view_dir: [0.0, 0.0, 1.0]
lights:
  - [0.5, 0.6, 0.8]
uv_checks:
  - [0.25, 0.25]
"""
    _write_yaml(
        root / "material_seam.yaml",
        seam_header
        + """name: seam_material
layers:
  - name: base
    normal: textures/normal_seam.png
    base_color: textures/color_base.png
    roughness: textures/rough_base.png
    weight: 1.0
    mode: overlay
""",
    )

    # 错误样本：未知混合模式
    _write_yaml(
        root / "case_bad_mode.yaml",
        common_header
        + """name: bad_mode
layers:
  - name: base
    normal: textures/normal_base.png
    base_color: textures/color_base.png
    roughness: textures/rough_base.png
    weight: 1.0
    mode: screen_plus
""",
    )

    # 错误样本：UV 越界（uv_scale 使变换后 UV 超出 [0,1)）
    _write_yaml(
        root / "case_uv_oob.yaml",
        common_header
        + """name: uv_oob
layers:
  - name: base
    normal: textures/normal_base.png
    base_color: textures/color_base.png
    roughness: textures/rough_base.png
    weight: 1.0
    mode: overlay
    uv_scale: [1.4, 1.0]
    uv_offset: [0.0, 0.0]
""",
    )

    # 错误样本：引用不存在的贴图
    _write_yaml(
        root / "case_missing_texture.yaml",
        common_header
        + """name: missing_texture
layers:
  - name: base
    normal: textures/does_not_exist.png
    base_color: textures/color_base.png
    roughness: textures/rough_base.png
    weight: 1.0
    mode: overlay
""",
    )

    # 错误样本：零权重
    _write_yaml(
        root / "case_invalid_layer.yaml",
        common_header
        + """name: invalid_layer
layers:
  - name: base
    normal: textures/normal_base.png
    base_color: textures/color_base.png
    roughness: textures/rough_base.png
    weight: 0.0
    mode: weighted
  - name: detail
    normal: textures/normal_detail.png
    base_color: textures/color_detail.png
    roughness: textures/rough_detail.png
    weight: 0.0
    mode: weighted
""",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="生成确定性材质样本")
    parser.add_argument("-o", "--output", default="samples")
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    build(Path(args.output), args.size)
    print(f"样本已生成于 {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
