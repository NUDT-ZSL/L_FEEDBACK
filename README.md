# Offline DOF Pipeline

从零实现的确定性离线景深（Depth of Field）渲染器。输入为程序生成的彩色序列、同尺寸
`float32` 线性深度图和焦点关键帧 JSON，输出 PNG 序列、逐帧统计报告和可选的切片一致性
验证报告。运行时不下载或读取任何外部图像资产。

## 环境

- Python >= 3.10（固定 NumPy 1.24.x 时建议 3.10/3.11）
- NumPy 1.24.x
- opencv-python 4.8.x（PNG 编解码）

Windows PowerShell 示例：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e . --no-deps
.venv\Scripts\Activate.ps1
```

Linux/WSL 示例：

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps
source .venv/bin/activate
```

不安装包也可以直接使用 `python -m dofpipe ...`。安装包后会提供 `dof-pipeline` 命令。

## 快速开始

```bash
python -m dofpipe generate --output samples/showcase --width 240 --height 135 --frames 8 --fps 12 --seed 12345
python -m dofpipe render --config samples/showcase/config.json --input samples/showcase --output out/showcase
python -m dofpipe verify-slices --config samples/showcase/config.json --input samples/showcase --json-output out/showcase/slice_verification.json
python -m dofpipe export-kernel --blades 0 --radius 10 --output out/showcase/circle.npy
python -m dofpipe export-kernel --blades 6 --radius 10 --output out/showcase/hex.json
```

完整自动验收：

```bash
python -m unittest discover -s tests -v
python scripts/acceptance.py --root validation/acceptance
```

最后一条命令依次验证：程序生成样本、两次渲染、PNG 字节哈希、去除耗时字段后的报告等价性、
切片/全图一致性、圆形与多边形核归一化、乱序焦点轨道的非零退出、失败报告及已生成帧保留。
实际结果见 [VERIFICATION.md](VERIFICATION.md)。

## 输入约定

生成器写出以下文件，也可以用自己的确定性输入替换：

- `color_{frame:04d}.png`：8 位灰度、BGR 或 BGRA；渲染时 BGRA 去掉 alpha。
- `depth_{frame:04d}.npy`：`float32`、二维、与彩色图同尺寸。约定值在 `[0, 1]`，
  `physical_depth = near_plane_m + value * (far_plane_m - near_plane_m)`。
- `focus_track.json`：时间戳（秒或帧）和焦点距离（米）。

焦点轨道示例：

```json
{
  "time_unit": "seconds",
  "keyframes": [
    {"time": 0.0, "focus_distance": 1.8},
    {"time": 0.25, "focus_distance": 3.8},
    {"time": 0.5, "focus_distance": 9.0}
  ]
}
```

`time_unit` 也可为 `"frames"`。关键帧时间必须严格递增，且位于输出序列的时间范围内。
输出帧时刻为绝对 `frame_index / frame_rate`，因此 `frame_start` 会同时影响文件名和
轨道对齐的时间范围。
轨道过早结束（或过晚开始）时，缺失帧沿用最近端点关键帧值，并在报告中写入
`track_ends_before_sequence` / `track_starts_after_sequence` 警告。

深度兜底规则固定为：`NaN`、`Inf`、`raw <= 1e-6` 视为无效，映射到
`invalid_depth_fallback_m`（默认远平面）；`raw > 1` 截断到 1。无效深度计数逐帧写入报告。

## 配置

`generate` 会生成完整可运行配置。关键参数如下：

```json
{
  "lens": {
    "focal_length_mm": 85.0,
    "f_number": 1.2,
    "sensor_width_mm": 36.0,
    "near_plane_m": 0.5,
    "far_plane_m": 20.0,
    "max_blur_radius_px": 10.0,
    "invalid_depth_fallback_m": 20.0,
    "depth_tolerance_m": 0.001
  },
  "aperture": {
    "blades": 6,
    "rotation_deg": 0.0,
    "supersample": 8
  },
  "render": {
    "depth_slices": 8,
    "gather_rounds": 2,
    "png_compression": 3,
    "output_pattern": "frame_{frame:04d}.png"
  }
}
```

`render --slices N` 和 `render --gather-rounds N` 可覆盖配置。`blades=0` 为圆形，
正整数为对应边数的正多边形。

## 算法说明

### 焦点插值

使用 Fritsch-Carlson 单调 PCHIP 三次 Hermite 插值。每个区间的结果额外夹在相邻关键帧值之间，
因此不会过冲；查询点等于关键帧时间时直接返回原始端点值。

### 薄透镜弥散圆

物距 `s`、对焦距离 `s_f`、焦距 `f`、F 值 `N` 时，传感器上弥散圆直径为：

```text
c = f^2 * |s - s_f| / (N * s * (s_f - f))
```

再用传感器宽度和图像宽度换算为像素。最终半径截断到
`[0, max_blur_radius_px]`。对焦距离不大于焦距、非有限值或非法镜头参数会给出明确异常，
不会进入除零路径。

### 光圈核

核矩阵为 `(2*ceil(radius)+1)^2`，按固定网格超采样圆形/正多边形并归一化，权重和为 1。
圆形核关于中心、坐标轴和对角线近似各向同性。所有形状归一到相同的有效通光能量；
多边形的原始几何支撑面积小于同半径圆，符合多边形遮光面积的比例。

### 散射、遮挡与切片

每个源像素按其弥散圆半径将颜色 `color * kernel_weight` 散布到邻域，同时累积
`weight_sum`（不透明度）、`depth_sum` 和最近源深度。目标像素执行可见性测试：

```text
source_depth <= destination_depth + depth_tolerance_m
```

因此背景样本不能进入更近的前景，前景散焦可以覆盖背景。深度范围按配置切成多个切片，
每片独立累积，再严格从远到近合并；合并只降低最近深度，不会反转遮挡顺序。
`gather_rounds` 将光圈采样半径分成多轮 gather；轮数变化只改变浮点累加顺序。

### 失败与部分统计

每完成一帧都会原子更新 `report.partial.json`。焦点时间戳乱序或越界时，异常携带
`frame_index` 和具体原因，CLI 以状态码 2 退出。已经生成的 PNG 保留，同时写出
`report.partial.json` 和状态为 `failed` 的 `report.json`。

## 输出与报告

成功时输出：

- `frame_XXXX.png`：8 位 BGR PNG。
- `report.json`：`status="completed"`、`completed=true`、`error=null`。

报告固定字段：

- 顶层：`schema_version`、`status`、`completed`、`error`、`parameters`、`warnings`、
  `total_frames`、`expected_total_frames`、`frames`。
- 每帧：`frame_index`、`sequence_index`、`output_file`、`focus_distance_m`、
  `coc_min_px`、`coc_max_px`、`coc_mean_px`、对应直径字段、
  `aperture_blades`、`max_blur_radius_px`、`invalid_depth_pixels`、
  `clamped_depth_pixels`、`render_time_ms`。

`coc_*_px` 是像素半径；同时提供 `coc_*_diameter_px`。`render_time_ms` 是该帧加载、渲染和
PNG 编码的实际墙钟时间。确定性比较时只屏蔽该字段。JSON 使用固定插入顺序、UTF-8、
标准 JSON 数值，不允许 `NaN`。

## 确定性与平台

- 样本生成使用固定 `numpy.random.default_rng(seed)`。
- 渲染固定 OpenCV 单线程和 RNG 种子；核采样、偏移顺序、切片合并顺序均由代码固定。
- PNG 使用固定压缩级别，文件不含时间戳元数据。
- 路径、原子替换和 JSON I/O 只使用跨平台 Python API，可直接在 Windows Python 3.10+
  运行；任务验证环境为 WSL2/Linux。
