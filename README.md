# softrender — 离线硬表面机甲预览软光栅化渲染器

纯 CPU（NumPy）软件光栅化渲染器，用于游戏离线资产生成流程中的高质量预览图输出。
跨平台（Windows / Linux / macOS），无需 GPU、无需外部模型或贴图：所有样本
（机甲 OBJ、材质 JSON、切线空间法线贴图 PNG、场景描述）均由代码按随机种子确定性生成。

## 环境

- Python 3.11+（Windows 官方安装包即可）
- `pip install numpy pillow`（NumPy ≥ 1.26，Pillow ≥ 10.0）

## 快速开始

```bash
# 一键：按种子生成样本资产并渲染预览图
python -m softrender run --seed 42 --output preview.png \
    --width 512 --height 512 --threads 4 --shadow-res 512

# 仅生成确定性样本资产（OBJ + 材质 JSON + 法线贴图 + scene.json）
python -m softrender generate --dir assets --seed 42

# 从已有 scene.json 渲染
python -m softrender render --scene assets/scene.json --output preview.png \
    --width 512 --height 512 --threads 4 --shadow-res 512

# 可复现性校验：同参数多线程跑两次 + 单线程参考，对比 SHA-256
python -m softrender verify --seed 42 --width 256 --height 256 \
    --threads 4 --shadow-res 256 --workdir verify_out

# 单元 / 集成测试
python -m unittest discover -s tests -v
```

参数校验：线程数允许范围 `1 .. os.cpu_count()`；阴影贴图分辨率须为 2 的幂
（16–8192）；图像分辨率 16–4096。非法参数、缺失/非法材质或模型文件均输出
明确错误到 stderr 并以非零退出码结束（参数错误 3，资产错误 4，渲染错误 5），
PNG 通过临时文件 + 原子重命名写出，失败时绝不产生部分图像。

## 架构

| 模块 | 职责 |
| --- | --- |
| `softrender/obj_loader.py` | OBJ 解析：`v/vt/vn/f`，1 起始索引与负索引，多边形扇形三角化 |
| `softrender/scene.py` | 材质 JSON（albedo/roughness/normal_map）与场景 JSON 校验 |
| `softrender/assets.py` | 确定性程序化机甲模型、地面、法线贴图与场景生成 |
| `softrender/normalmap.py` | 可平铺高度场 → RGB8 切线空间法线贴图（+Y 向上）；双线性采样、重复寻址、重归一化 |
| `softrender/geometry.py` | 4x4 矩阵变换、合并包围盒相机、视锥剔除、近裁剪面 Sutherland–Hodgman 裁剪 + 重新三角化、1/256 像素定点量化、UV 微分切线基（退化时回退面法线正交基） |
| `softrender/raster.py` | 整数边函数 + top-left 填充规则、透视校正插值、float32 Z-buffer（小于比较）、16x16 分块多线程（每块独立局部深度，合并到全局 Z-buffer，逐像素确定） |
| `softrender/shadow.py` | 2 盏平行光正交阴影贴图 + 点光源 6 面立方体阴影贴图（径向深度、跨面无缝寻址）、4x4 PCF 双线性比较采样、斜率缩放 + 法线偏移深度偏差 |
| `softrender/shading.py` | Cook-Torrance GGX（Schlick 菲涅尔 + Smith 几何项）、点光源平方衰减（最小距离截断）、切线空间法线贴图应用 |
| `softrender/tonemap.py` | ACES（Narkowicz 拟合）色调映射 + sRGB 伽马编码 → 8-bit PNG |
| `softrender/pipeline.py` | 加载 → 阴影贴图 → 主通道 → 色调映射编排与耗时统计 |
| `softrender/cli.py` | `generate / render / run / verify` 命令行入口 |

关键实现要点：

- 所有几何变换使用 4x4 矩阵（模型/视图/投影）；屏幕坐标保留 1/256 像素定点精度
  （优于要求的 1/16）。
- 近裁剪面裁剪保证透视除法后不存在 w ≤ 0 顶点（有单元测试断言）。
- HDR 帧缓冲为 RGB 各 32 位浮点；ACES + sRGB 后三通道统一量化为 8 位。
- 多线程确定性：16x16 瓦片互不重叠，每瓦片独立局部深度缓冲，结果与线程数、
  调度顺序无关，与单线程参考逐像素一致。

## 验证结果（本机实测）

- `python -m unittest discover -s tests -v`：15 个测试全部通过
  （OBJ 负索引/越界、材质校验、线程数校验、top-left 规则无裂缝无重复覆盖、
  近裁剪 w>0、多线程与单线程逐像素一致、CLI 错误路径无部分图像）。
- `python -m softrender verify --seed 42 --width 256 --height 256 --threads 4 --shadow-res 256`：
  多线程两次与单线程参考的 PNG SHA-256 完全一致
  （`7623e9af…c6b27`），输出 `verify: PASS`。
- 非法参数（`--shadow-res 100`、`--threads 0`）、缺失材质文件、非法材质 JSON：
  均输出 `error: ...` 到 stderr，退出码非零（3/4），且不产生输出 PNG。
- 渲染耗时统计打印到 stdout（`render stats: load=… shadow_maps=… main_pass=… total=…`），
  不参与图像哈希。
