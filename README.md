# 离线软光栅室内渲染器

一个只依赖 NumPy 与 Pillow 的确定性 CPU 软光栅渲染器，用于多灯光、阴影和材质分层的离线美术验收。渲染器不使用 GPU、OpenGL、Docker 或外部数据集。

样例预览：`samples/indoor_preview.png`（由确定性样例经 medium 预设渲染）。

## 安装

需要 64 位 Python 3.8 或更高版本，以及可用的 pip。

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

依赖版本：`numpy>=1.20`、`Pillow>=8.0`。

## 快速开始

```bash
# 1. 重新生成确定性场景
python generate_scene.py samples/indoor.json --width 320 --height 240

# 2. 整体渲染 + ACES/gamma PNG + 分层 PFM + JSON 报告
python -m soft_renderer render samples/indoor.json \
  out/indoor.pfm \
  --preset medium \
  --block-size 4 \
  --png out/indoor.png \
  --layers-dir out/layers \
  --report out/report.json

# 3. 从分层文件按深度重新合成
python -m soft_renderer composite \
  --layers-dir out/layers \
  -o out/composite.pfm \
  --png out/composite.png \
  --report out/composite_report.json
```

也可以省略 `render` 子命令：

```bash
python soft_render.py samples/indoor.json out/indoor.pfm --block-size 8 --preset low
```

预览偏亮时可加 `--exposure 0.55`；曝光只影响 PNG，不会修改 PFM HDR 数据。

## 命令行

### `render`

```text
render SCENE [OUTPUT] [选项]
  -o, --output PATH       输出 PFM，也可以使用位置参数
  --block-size N          任意正整数，默认 4（验收覆盖 1/2/4/8）
  --no-tiling             将整个超采样帧作为一个块渲染
  --preset NAME           low/medium/high，默认 medium
  --material ID           只渲染指定材质层
  --layers-dir DIR        输出每个材质的颜色 PFM、深度 PFM 和 layers.json
  --png PATH              ACES 色调映射并做 2.2 gamma 校正的预览
  --report PATH           输出参数、阶段耗时、阴影分类和像素哈希
  --exposure VALUE        PNG 曝光，默认 1.0
  --quiet                 成功时不打印摘要
```

### `composite`

```text
composite --layers-dir DIR [OUTPUT] [选项]
  -o, --output PATH       合成后的 PFM
  --png PATH              合成帧的预览
  --report PATH           合成耗时和像素哈希
  --exposure VALUE        PNG 曝光
```

### `generate`

```text
generate OUTPUT --width N --height N
```

`generate_scene.py` 等价，用于生成项目内确定性室内样例。

## 场景 JSON

完整样例见 `samples/indoor.json`。最小结构如下：

```json
{
  "version": 1,
  "color_space": "srgb",
  "image": {"width": 320, "height": 240},
  "camera": {
    "position": [0, 2.15, 7.2],
    "target": [0, 1.05, -0.4],
    "up": [0, 1, 0],
    "fov_y_deg": 42,
    "near": 0.05,
    "far": 100
  },
  "background": [0.015, 0.018, 0.022],
  "ambient": [0.025, 0.03, 0.04],
  "materials": [
    {
      "id": 0,
      "name": "floor",
      "base_color": [0.42, 0.25, 0.11],
      "roughness": 0.58,
      "metallic": 0.0
    }
  ],
  "meshes": [
    {
      "vertices": [
        {"position": [-5, 0, 5], "uv": [0, 0]},
        {"position": [5, 0, 5], "uv": [2, 0]},
        {"position": [5, 0, -3], "uv": [2, 2]}
      ],
      "triangles": [
        {"indices": [0, 1, 2], "material": 0}
      ]
    }
  ],
  "lights": [
    {
      "name": "warm_point",
      "type": "point",
      "position": [-2.75, 3.4, 2.2],
      "color": [1.0, 0.69, 0.39],
      "intensity": 92.0,
      "attenuation": {
        "constant": 1.0,
        "linear": 0.12,
        "quadratic": 0.42,
        "range": 13.0
      },
      "shadow_bias": 0.0012,
      "cast_shadow": true
    },
    {
      "name": "cool_spot",
      "type": "spot",
      "position": [3.1, 4.15, 3.65],
      "direction": [-0.51, -0.57, -0.64],
      "color": [0.45, 0.72, 1.0],
      "intensity": 145.0,
      "inner_angle_deg": 18,
      "outer_angle_deg": 33,
      "attenuation": {"constant": 1, "linear": 0.1, "quadratic": 0.35, "range": 15}
    }
  ]
}
```

约定：

- 每个顶点包含 `position`，`normal` 与 `uv` 可省略。
- 缺失法线按相邻三角形面积加权叉积累加后归一化；提供法线时会归一化。
- 三角形可写成 `{"indices": [i, j, k], "material": id}`，也可用 `mesh.indices`/`mesh.triangles` 加 `material_ids`。
- 默认所有颜色按 sRGB 输入并转换到线性空间；设 `"color_space": "linear"` 可跳过转换。
- 聚光灯也可用 `target` 代替 `direction`。`inner_angle_deg < outer_angle_deg`。
- 每盏灯可用 `shadow_resolution` 覆盖预设阴影图分辨率。
- 背景、环境光、材质基色和灯光颜色都保存在同一个 HDR 线性计算域。
- `--material ID` 输出该材质层，层外像素为线性黑（0,0,0）。

## 质量预设

| 预设 | 阴影图分辨率 | PCF 采样数 | 超采样倍率 |
| --- | ---: | ---: | ---: |
| low | 128 | 1 | 1 |
| medium | 256 | 4 | 2 |
| high | 512 | 9 | 3 |

点光源使用 6 面立方体阴影图；聚光灯使用覆盖外锥角的单张透视阴影图，超过 84° 的宽角聚光灯自动改用立方体阴影图。每盏灯拥有独立深度图和可选独立分辨率。报告中的 `shadow_statistics` 分别统计全受光、半影和全遮挡的片元样本数。

## 实现要点

**几何与颜色**：`soft_renderer/scene.py` 完成 JSON 校验、网格扁平化、材质 ID 检查、面积加权法线、sRGB 到线性的转换以及相机/灯光参数校验。

**分块光栅化**：屏幕被切成互不重叠的块。每块维护自己的深度缓冲和 G-buffer，完成三角形覆盖、透视校正属性插值和着色后再写回整帧。空间索引只跳过完全不与任何材质三角形相交的块，不改变像素结果。同一预设与输入下，块尺寸 1/2/4/8 生成的 PFM 字节一致。

**物理着色**：材质使用 GGX 法线分布、Smith-Schlick 几何项、Schlick Fresnel、Lambert 漫反射和金属度工作流。距离衰减为 `1/(c + l*d + q*d^2)`，带 `(1-(d/range)^4)^2` 平滑截断；聚光灯在内外锥角之间使用 smoothstep。多灯结果直接线性累加，不裁剪 HDR。

**阴影**：阴影图存储从光源到表面的径向距离。查询时使用斜率感知的深度偏移、法线偏移和方形 PCF 网格；点光源按主轴选择立方体面。PCF 可见率自然形成 0（全遮挡）、分数（半影）、1（受光）三类。

**材质分层**：每个材质独立渲染到颜色 HDR 层和深度层，层内仍有完整 Z-buffer。合成时逐像素选择最近深度；精确同深度时按较小材质 ID 稳定决胜。整体 `render` 也复用同一合成函数，因此磁盘分层合成与整体帧逐位一致。

**分层文件分辨率**：为保证超采样边缘也能精确复合成最终帧，`material_<id>.pfm` 和深度 PFM 使用内部渲染分辨率（输出尺寸 × supersample）。`layers.json` 同时记录最终输出尺寸、内部渲染尺寸和倍率；`composite` 在内部高分辨率完成深度选择后再盒式降采样。

**HDR 与预览**：PFM 使用 `PF/Pf` 与负 scale 标记小端 float32，按 PFM 规范从底行向上写，读回时翻转。PNG 始终从同一次渲染的最终 float32 HDR 缓冲计算：曝光、ACES fitted filmic 曲线、2.2 gamma、8 位量化。

**可复现哈希**：报告中的 `pixel_result_sha256` 对最终 HDR 的宽高和按行主序存储的小端 float32 字节做 SHA-256。耗时字段会变化，但相同输入与预设的像素哈希稳定。

**原子输出**：所有 PFM、深度 PFM、PNG、JSON 报告先写入同目录临时文件并 `fsync`，全部编码成功后统一 `os.replace`。提交中失败会删除已安装文件、恢复旧文件并清理临时文件；编码或目录错误不会留下半写入文件。

## 错误处理

- 成功返回 `0`。
- 场景结构、文件 IO、渲染或输出失败返回 `1`。
- 用法错误、未知预设或未支持块尺寸返回 `2`。
- 错误原因写到 stderr，例如 `error: lights[1].direction: ...`。

## 测试与验收

```bash
# 15 个单元/集成测试
python -m unittest discover -v

# 完整确定性验收，保留中间文件并写 JSON 报告
python verify_acceptance.py \
  --workdir artifacts/acceptance \
  --report docs/acceptance_report.json
```

验收覆盖：确定性样本生成、块尺寸逐位一致、PNG 与 HDR 固定变换一致、medium 超采样分层逐位合成、低/中/高预设参数和哈希稳定性、阴影三类可见率、损坏 JSON/非法块尺寸/不可写输出路径的非零退出与无残留。

实际运行结果见 `docs/VALIDATION.md`。

## 项目结构

```text
soft_renderer/
  scene.py        场景解析、法线、颜色与参数校验
  geometry.py     相机、近裁剪、投影、重心坐标
  raster.py       分块覆盖、深度缓冲、G-buffer、阴影图光栅
  shading.py      GGX/PBR、衰减、多灯累加、可见率统计
  shadows.py      点光立方体阴影、聚光阴影、PCF
  renderer.py     材质层渲染、深度合成、超采样
  imageio.py      PFM、ACES/gamma PNG、原子输出事务
  layers.py       分层清单、磁盘合成
  sample_scene.py 确定性室内场景
  cli.py          命令行与报告
samples/          由生成脚本产出的样例场景和预览
tests/            单元与集成测试
verify_acceptance.py
generate_scene.py
```
