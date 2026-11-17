from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np

from .cache import RenderCache
from .config import (
    Camera,
    CascadeSettings,
    DirectionalLight,
    PointLight,
    RenderSettings,
    ToneSettings,
)
from .errors import RenderError, ValidationError
from .renderer import render_scene
from .samples import generate_heightfield, generate_weight_maps


def _load_array(path: str, name: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise ValidationError(f"{name} 文件不存在: {path}")
    try:
        if path.lower().endswith(".npy"):
            return np.load(path, allow_pickle=False)
        delimiter = "," if path.lower().endswith(".csv") else None
        return np.loadtxt(path, delimiter=delimiter)
    except Exception as exc:
        raise ValidationError(f"无法读取 {name}: {path}: {exc}") from exc


def _parse_dirty(values: list[str] | None) -> tuple[int, int, int, int] | None:
    if values is None:
        return None
    tokens: list[str] = []
    for value in values:
        tokens.extend(part for part in value.replace(";", ",").split(",") if part != "")
    if len(tokens) != 4:
        raise ValidationError("--dirty-rect 需要四个整数: x0 y0 x1 y1")
    try:
        return tuple(int(token) for token in tokens)  # type: ignore[return-value]
    except ValueError as exc:
        raise ValidationError("--dirty-rect 只能包含整数") from exc


def _parse_float_triple(value: str, name: str = "三维向量") -> tuple[float, float, float]:
    try:
        parts = [part for part in value.replace(";", ",").split(",") if part != ""]
        if len(parts) != 3:
            raise ValueError
        result = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} 需要逗号分隔的三个数值") from exc
    return result  # type: ignore[return-value]


def _parse_point_light(value: str) -> PointLight:
    parts = [part for part in value.replace(";", ",").split(",") if part != ""]
    if len(parts) not in (5, 8):
        raise ValidationError(
            "--point-light 格式为 x,y,z,range,intensity[,r,g,b]"
        )
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValidationError("--point-light 只能包含数值") from exc
    position = tuple(numbers[0:3])
    light_range = numbers[3]
    intensity = numbers[4]
    color = tuple(numbers[5:8]) if len(numbers) == 8 else (1.0, 0.82, 0.62)
    return PointLight(position, color, intensity, light_range)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="render_terrain",
        description="确定性 CPU 地形软光栅离线预览渲染器",
    )
    parser.add_argument("--height-size", type=int, default=65, help="生成高度场的规模 N（N x N）")
    parser.add_argument("--height-file", help="可选高度场 .npy/.txt/.csv，覆盖生成样本")
    parser.add_argument("--weights-file", help="可选权重图 .npy，形状 (图层数, N, N)")
    parser.add_argument("--layer-count", type=int, default=4, help="生成权重图的图层数")
    parser.add_argument("--seed", type=int, default=1337, help="高度场随机种子")
    parser.add_argument("--weight-seed", type=int, default=4242, help="权重图随机种子")
    parser.add_argument("--resolution", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--width", type=int, help="输出宽度（也可使用 --resolution）")
    parser.add_argument("--height", type=int, help="输出高度（也可使用 --resolution）")
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--cascades", "--cascade-count", dest="cascades", type=int, default=3)
    parser.add_argument("--shadow-size", type=int, default=512)
    parser.add_argument("--cascade-lambda", type=float, default=0.68)
    parser.add_argument("--cascade-overlap", type=float, default=0.12)
    parser.add_argument("--shadow-pcf-radius", type=int, default=1)
    parser.add_argument("--no-backface-culling", action="store_true")
    parser.add_argument("--unit-scale", type=float, default=0.25, help="高度场相邻顶点世界间距")
    parser.add_argument("--height-scale", type=float, default=2.0, help="高度值世界放大系数")
    parser.add_argument("--camera-eye", type=_parse_float_triple, default=(0.0, 9.0, 12.0))
    parser.add_argument("--camera-target", type=_parse_float_triple, default=(0.0, 0.4, 0.0))
    parser.add_argument("--camera-up", type=_parse_float_triple, default=(0.0, 1.0, 0.0))
    parser.add_argument("--fov", type=float, default=48.0)
    parser.add_argument("--near", type=float, default=0.15)
    parser.add_argument("--far", type=float, default=80.0)
    parser.add_argument("--light-direction", type=_parse_float_triple, default=(-0.62, -0.72, -0.42))
    parser.add_argument("--light-color", type=_parse_float_triple, default=(1.0, 0.95, 0.86))
    parser.add_argument("--light-intensity", type=float, default=1.15)
    parser.add_argument(
        "--point-light",
        action="append",
        default=None,
        help="可重复: x,y,z,range,intensity[,r,g,b]；提供时替换默认双点光",
    )
    parser.add_argument("--ambient", type=_parse_float_triple, default=(0.095, 0.115, 0.15))
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--dirty-rect", nargs="+", metavar="X0 Y0 X1 Y1", help="增量渲染脏区")
    parser.add_argument("--cache-file", help="增量渲染缓存 .npz；不存在时首次全量渲染并创建")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--debug", action="store_true", help="内部错误时显示堆栈")
    return parser


def _resolve_resolution(args: argparse.Namespace) -> tuple[int, int]:
    if args.resolution is not None:
        if args.width is not None or args.height is not None:
            raise ValidationError("--resolution 不能与 --width/--height 同时使用")
        return int(args.resolution[0]), int(args.resolution[1])
    width = 320 if args.width is None else args.width
    height = 240 if args.height is None else args.height
    return int(width), int(height)


def run(args: argparse.Namespace) -> int:
    width, height = _resolve_resolution(args)
    dirty_rect = _parse_dirty(args.dirty_rect)
    settings = RenderSettings(
        width=width,
        height=height,
        tile_size=args.tile_size,
        cascades=CascadeSettings(
            count=args.cascades,
            shadow_size=args.shadow_size,
            split_lambda=args.cascade_lambda,
            overlap_fraction=args.cascade_overlap,
        ),
        ambient=args.ambient,
        tone=ToneSettings(args.exposure, args.gamma),
        backface_culling=not args.no_backface_culling,
        shadow_pcf_radius=args.shadow_pcf_radius,
    )
    camera = Camera(
        eye=args.camera_eye,
        target=args.camera_target,
        up=args.camera_up,
        fov_y_deg=args.fov,
        near=args.near,
        far=args.far,
    )
    light = DirectionalLight(args.light_direction, args.light_color, args.light_intensity)
    point_lights = None
    if args.point_light:
        point_lights = tuple(_parse_point_light(value) for value in args.point_light)
    if args.height_file:
        heightfield = _load_array(args.height_file, "高度场")
    else:
        heightfield = generate_heightfield(args.height_size, args.seed)
    if args.weights_file:
        weights = _load_array(args.weights_file, "权重图")
    else:
        weights = generate_weight_maps(heightfield, args.layer_count, args.weight_seed)
    if weights.ndim != 3 or weights.shape[1:] != heightfield.shape:
        raise ValidationError(
            f"权重图尺寸不匹配: 高度场 {heightfield.shape}, 权重图 {weights.shape}"
        )
    cache = None
    if args.cache_file and os.path.isfile(args.cache_file):
        cache = RenderCache.load(args.cache_file)
    result = render_scene(
        heightfield,
        weights,
        camera=camera,
        directional_light=light,
        point_lights=point_lights,
        settings=settings,
        unit_scale=args.unit_scale,
        height_scale=args.height_scale,
        cache=cache,
        dirty_rect=dirty_rect,
        input_metadata={
            "heightfield_source": "file" if args.height_file else "deterministic_generator",
            "heightfield_seed": None if args.height_file else args.seed,
            "weights_source": "file" if args.weights_file else "deterministic_generator",
            "weights_seed": None if args.weights_file else args.weight_seed,
            "generated_layer_count": None if args.weights_file else args.layer_count,
        },
    )
    result.save(args.output_dir, cache_path=args.cache_file)
    incremental = result.manifest["incremental"]
    print(
        "rendered "
        f"{settings.width}x{settings.height}, mode={incremental['mode']}, "
        f"tiles={incremental['rendered_tiles']}/{incremental['total_tiles']}, "
        f"warnings={','.join(result.manifest['warning_codes']) or 'none'}, "
        f"output={os.path.abspath(args.output_dir)}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ValidationError as exc:
        print(f"ERROR: 参数校验失败: {exc}", file=sys.stderr)
        return 2
    except RenderError as exc:
        print(f"ERROR: 渲染失败: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"ERROR: 文件操作失败: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: 内部错误: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1
