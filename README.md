# Deterministic Tiled Deferred Renderer

面向本地离线预览的延迟着色渲染器。场景、几何、纹理、相机和光源都由同一份 JSON
确定性生成，不读取外部模型或图片。渲染器支持无头 FBO 与 GLFW 窗口两种模式，
输出 RGBA PNG 和带 G-Buffer、tile 光源列表、景深直方图及哈希的统计 JSON。

## 运行环境

- Python 3.10 至 3.12（当前验证环境使用 Python 3.14，代码未使用 3.13/3.14 专有特性）
- OpenGL 3.3 Core 或更高
- 无头模式：Linux 使用 EGL surfaceless/device；Windows 使用 ModernGL standalone WGL
- 窗口模式：可用显示器；Windows 优先 GLFW，未安装时自动回退到原生 Win32/WGL；
  Linux 窗口模式需要 `glfw`

安装：

```bash
python -m pip install -r requirements.txt
```

如果只跑无头模式，前四项依赖即可：

```bash
python -m pip install numpy moderngl PyOpenGL pillow
```

Linux 窗口模式额外需要：

```bash
python -m pip install glfw
```

Windows 下即使只安装任务给定的 `numpy moderngl PyOpenGL pillow`，窗口模式也会
回退到 ctypes 实现的 Win32 窗口与 WGL 上下文；安装 `glfw` 时优先使用 GLFW。
窗口统计 JSON 中的 `window_backend` 会记录实际后端。

## 快速开始

生成确定性默认场景并渲染：

```bash
python -m deferred_renderer --write-default-scene scenes/default.json
python -m deferred_renderer --scene scenes/default.json --out artifacts/render --mode headless
```

窗口模式（隐藏窗口便于自动比对）：

```bash
python -m deferred_renderer \
  --scene scenes/default.json \
  --out artifacts/window \
  --mode window \
  --window-hidden
```

显示窗口并保持 5 秒：

```bash
python -m deferred_renderer \
  --scene scenes/default.json \
  --out artifacts/window_visible \
  --mode window \
  --window-seconds 5
```

输出：

- `frame_0000.png`：RGBA8、已做 sRGB 编码的显示帧
- `stats.json`：渲染统计、格式约定、tile 光源列表、直方图、哈希、耗时与错误信息

常用覆盖参数：

```bash
--width 96 --height 54 --tile-size 16
--no-dof
--no-reference-check
--stats path/to/stats.json
```

## 场景格式

`scenes/default.json` 由 `default_scene_dict()` 确定性生成，主要字段：

```json
{
  "schema_version": 1,
  "seed": 194,
  "camera": {
    "position": [4.8, 3.6, 7.8],
    "target": [0.0, 1.05, 0.0],
    "up": [0.0, 1.0, 0.0],
    "fov_y_deg": 48.0,
    "near": 0.1,
    "far": 60.0
  },
  "render": {
    "width": 320,
    "height": 180,
    "tile_size": 16,
    "g_buffer": {"attachments": [
      "albedo_roughness", "normal_side", "linear_depth", "material_id"
    ]},
    "tone_mapping": {"curve": "aces", "exposure": 1.2, "srgb": true},
    "dof": {
      "enabled": true,
      "focus_distance": 6.8,
      "aperture": 1.1,
      "focal_length_mm": 100.0,
      "sensor_width_mm": 36.0,
      "max_blur_radius": 6.0
    },
    "reference_check": {"enabled": true, "abs_threshold": 0.01}
  },
  "materials": [],
  "lights": [],
  "objects": []
}
```

几何类型为 `floor`、`sphere`、`box`。物体使用材质名称引用材质；`box` 可设置
`"open": true` 去掉顶面，以便在正常相机位置看到背面材质。材质包含基础颜色、
粗糙度、程序化图案、纹理尺寸、UV 缩放、背面粗糙度修正和法线扰动强度。

## 管线

渲染顺序：

1. 按确定性的整数噪声/正弦图案生成反照率、粗糙度与切线空间法线纹理。
2. 生成球体、箱体、地板的顶点、索引、法线、切线与 UV，并计算资产 SHA-256。
3. 按屏幕 tile 行优先顺序用 scissor 绘制 G-Buffer。
4. 读取 G-Buffer 线性深度，重建每个 tile 的世界空间深度包围体。
5. 用精确球体/凸视锥相交测试生成每 tile 光源索引列表，上传到整数纹理。
6. 按同一 tile 顺序执行延迟光照；着色器只通过 G-Buffer 读取像素属性。
7. 对线性 HDR 做前景/背景分层景深合成。
8. 应用曝光、曲线和 sRGB 编码，写入 RGBA8 FBO，读取并保存 PNG。
9. 如需窗口模式，直接把同一 RGBA8 离屏纹理 blit 到默认帧缓冲。

FBO、G-Buffer 纹理、HDR 纹理、DOF 纹理和 tile 索引纹理都只创建一次并在整个
渲染过程中复用。

### G-Buffer 附件约定

| Location | 名称 | 格式 | 编码与范围 |
|---:|---|---|---|
| 0 | `albedo_roughness` | RGBA16F | RGB 为线性反照率 `[0,1]`；A 为未做背面修正的基础粗糙度 `[0,1]` |
| 1 | `normal_side` | RGBA16F | XYZ 为世界空间有符号单位法线 `[-1,1]`；W 为 `+1` 正面、`-1` 背面 |
| 2 | `linear_depth` | R32F | 正视图空间深度；有效几何在 `[near,far]`；背景哨兵为 `0` |
| 3 | `material_id` | R32UI | 无符号整数材质 ID；背景为 `0` |

线性深度在 G-Buffer 顶点着色器中由裁剪空间深度和近远平面计算：

```text
z_linear = 2*near*far / (far + near - z_ndc*(far - near))
```

后续阶段使用同一约定。屏幕像素的世界坐标由 `inv_projection` 恢复视图射线，再乘以
`linear_depth`，最后经 `inv_view` 变换到世界空间。tile 裁减、延迟着色和 CPU
参考实现使用相同公式。

### Tile 光源裁减

每个 tile 从 G-Buffer 取有效线性深度最小/最大值，用四个屏幕边界和两个深度值构造
8 个世界空间角点，形成深度有界的凸视锥。对每个光源包围球：

- 球心在凸视锥内：必定加入；
- 球心在外：计算到 6 个面的 12 个三角形的最近距离，距离不超过半径则加入；
- 否则不加入。

这是精确的球体/凸多面体相交测试，避免仅做 6 平面测试时在角点附近产生误判。
tile 索引纹理每行第一个 `uint` 是光源数量，后续 `uint` 是升序光源索引。统计 JSON
同时记录每 tile 的 `lights` 数组、平均光源数、最大值和光源数直方图；tile 坐标原点
为左下角，与 OpenGL `gl_FragCoord` 一致。

### 双面材质与参考校验

G-Buffer 记录 `gl_FrontFacing` 推导的侧面符号和面向观察者的世界法线。延迟着色
从 `material_id` 附件读取材质 ID，再应用该材质的背面粗糙度缩放/偏置。材质 ID
不会在延迟阶段重新光栅化。

当 `reference_check.enabled=true` 时，CPU 参考实现读取同一份 G-Buffer，遍历全部
光源逐像素计算与 GPU 相同的 GGX/菲涅耳 BRDF，并和 GPU 的 tile 裁减 HDR 结果比较。
`max_abs_error` 必须小于配置的 `abs_threshold`；当前默认阈值为 `0.01`。

### 景深与色调映射

弥散圆半径使用薄透镜近似：

```text
r_px = 0.5 * (f/N) * f * |z-focus|
       / max(z*|focus-f|, 1e-6) / (sensor_width_m/width_px)
```

背景 `z=0` 使用 `0.75 * max_blur_radius`。景深着色器把样本分为前景层和背景层，
前景层只收集焦点前样本，背景层只收集焦点后/背景样本，最终按“背景先、前景后”的
顺序合成。统计 JSON 包含 16 桶全像素直方图、几何像素直方图、有效半径的最小/平均/
最大值以及前景/背景像素数。

线性 HDR 先乘曝光，再应用 `linear`、`reinhard`、`aces` 或 `filmic` 曲线，最后用
分段 sRGB 转换编码到 RGBA8。

## 错误处理

所有输入错误写入统计 JSON，同时向 stderr 输出 `[业务码]` 和细节，并返回非零退出码。

| 业务码 | 触发条件 | 退出码 |
|---|---|---:|
| `E_MISSING_ATTACHMENT` | 缺少 G-Buffer 声明或必需附件 | 2 |
| `E_BAD_LIGHT` | 光源包围球半径非法、光源字段非有限等 | 2 |
| `E_ZERO_AREA_GEOMETRY` | 地板/球体/箱体尺寸为零，或生成网格含退化三角形 | 2 |
| `E_SCENE_SCHEMA` | 其他场景结构/范围错误 | 2 |
| `E_SCENE_IO` / `E_SCENE_JSON` | 场景读取或 JSON 解析失败 | 2 |
| `E_WINDOW_CREATE` | GLFW 缺失、窗口或上下文创建失败 | 3 |
| `E_GL_CONTEXT` / `E_SHADER_*` | OpenGL 上下文或着色器失败 | 3 |
| `E_REFERENCE_MISMATCH` | GPU 延迟结果超过 CPU 参考阈值 | 4 |
| `E_OUTPUT_IO` | PNG 写入失败 | 5 |
| `E_INTERNAL` | 未预期内部异常 | 1 |

错误检查均由输入场景内容或真实运行环境触发，没有为测试改源码制造失败的开关。

## 测试与验收

运行全部测试：

```bash
python -m pytest
```

运行完整验收并生成报告：

```bash
python scripts/run_acceptance.py
```

验收脚本会生成场景、执行两次无头渲染和一次隐藏窗口渲染、比较 PNG 像素、检查
CPU 参考误差、G-Buffer、tile 列表、景深直方图和四类错误路径，然后调用 pytest。

### 本次实际验证结果

报告：`artifacts/acceptance/report.json`

- Python 测试：`13 passed`
- 无头与隐藏窗口 PNG 最大通道差：`0`
- 重复无头渲染像素 SHA-256：
  `510c958a2f96cdce6827051e6c2709b171bd4c10b7df7536407f45e4bf81da47`
- 几何 SHA-256：
  `8bec5b7b4e081a9b0100bb89ed7d50f38801fb69d04ce20bd1514d5bd584818b`
- 纹理 SHA-256：
  `819ed2eacff3e0cb2f86f0453bafb79a03f562ef841a448f69ea0a18e88d8723`
- 160×90 验收场景 G-Buffer：7323 个几何像素，正面/背面像素均存在，6 个材质 ID
- Tile：60 个，16×16，平均 `2.8167` 个光源/tile，最大 `6`
- CPU 全屏参考：最大绝对误差 `0.001671`，阈值 `0.01`
- 非法输入：`E_MISSING_ATTACHMENT`、`E_BAD_LIGHT`、`E_ZERO_AREA_GEOMETRY`、
  `E_WINDOW_CREATE` 均返回非零退出码并写入相同业务码
- 环境：WSL2 Ubuntu、Mesa llvmpipe OpenGL 4.5；无头走 EGL surfaceless，窗口走
  GLFW X11。当前环境不是原生 Windows 桌面，Windows 路径使用相同的
  ModernGL/GLFW API，但本次未在原生 Windows 10/11 上执行。

示例帧：`artifacts/acceptance/headless_a/frame_0000.png`

## 项目结构

```text
deferred_renderer/
  cli.py          CLI、输出与错误写入
  scene.py        JSON 校验、程序化几何/纹理、资产哈希
  renderer.py     GL 上下文、FBO、分块渲染、延迟光照与统计
  culling.py      深度视锥、精确球体相交、tile 列表、CoC 直方图
  reference.py    CPU 全屏逐像素 BRDF 参考
  shaders/        G-Buffer、延迟、DOF、色调映射、窗口 blit
tests/            单元与端到端测试
scripts/          完整验收脚本
scenes/           确定性默认场景
artifacts/        实际运行输出与验收报告
```

