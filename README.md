# 纯软件确定性级联阴影图（CSM）离线管线

本项目是一套**不依赖 GPU、图形驱动或任何图形界面**的离线级联阴影图管线，
仅使用 Python 3.9+ 与 NumPy 实现。所有随机数、矩阵、光栅化与指标计算都采用
固定顺序的确定性运算，同一份种子与参数在任意支持的机器上重复运行，产物逐位一致。

## 1. 环境准备

```bash
python3 -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
python -m pip install -r requirements.txt
```

本次验证环境：CPython 3.14.4 + NumPy 2.5.4（Linux 开发机）。代码仅使用
Python 3.9 兼容语法与稳定的 NumPy API，可直接在无界面的 Windows 机器运行：

```bat
python -m csm_pipeline make-path --seed 20261115 --output path.json
python -m csm_pipeline run --seed 20261115 --camera-path path.json ^
  --cascades 4 --shadow-resolution 256 --width 320 --height 240 ^
  --output-dir out
```

## 2. 模块结构

| 模块 | 职责 |
| --- | --- |
| `csm_pipeline/scene.py` | 固定 PCG64 种子生成地面、立方体、球体、光方向、相机关键帧与几何指纹 |
| `csm_pipeline/geometry.py` | 确定性平面/立方体/UV 球体三角网格，顶点顺序固定 |
| `csm_pipeline/matrices.py` | 右手系透视/正交/视矩阵，NDC 深度统一映射到 `[0,1]` |
| `csm_pipeline/camera_path.py` | 相机路径 JSON 加载、严格校验、关键帧线性插值 |
| `csm_pipeline/csm.py` | PSSM 级联划分、光源正交矩阵、方形纹素与纹素吸附 |
| `csm_pipeline/rasterizer.py` | 纯 NumPy 软件光栅化：深度缓冲、世界坐标/法线 G-buffer |
| `csm_pipeline/shadows.py` | 级联选择、法线偏移、斜率自适应深度偏移、PCF、级联淡入淡出 |
| `csm_pipeline/metrics.py` | PSNR、NCC、最大绝对误差、覆盖率、平均偏移、PGM/校验和 |
| `csm_pipeline/depth_archive.py` | 自定义二进制深度纹理存档与头部/截断校验 |
| `csm_pipeline/pipeline.py` | 端到端编排、配置校验、产物与 JSON 报告 |
| `csm_pipeline/cli.py` | 配置驱动命令行入口与统一错误报告 |

## 3. 关键工程设计

### 确定性

- 随机源固定为 `numpy.random.Generator(PCG64(seed))`，所有抽样顺序固定。
- 几何、矩阵使用 float64；深度缓冲与阴影因子使用固定字节序的 float32。
- 不使用集合/哈希遍历、线程调度或任何时间相关输入。
- 报告写入 `sort_keys=True` 的 JSON，深度/因子以小端 `<f4` 字节落盘。
- 每次输出附带几何 SHA-256 指纹与各数组 SHA-256 校验和，便于跨机比对。

### 级联与纹素吸附

- 采用 PSSM `lambda` 混合（默认 0.5）生成对数/均匀混合的级联分割。
- 每级联光源正交视锥的 XY 包围盒强制为**正方形**，保证纹素为方形。
- 光源眼点放置在场景包围盒远侧，使全部几何都位于光源正前方（正的近平面）。
- 近/远平面基于完整场景 8 角点包围盒，避免地面等视锥外几何被裁剪。
- 纹素吸附在光源视图坐标下把视锥原点对齐到整数纹素网格；吸附只改变平移，
  不改变旋转、尺度与深度范围。吸附前/后两套矩阵共用同一级联索引。

### 阴影采样与偏移

- 相机视口软件光栅化产出世界坐标与法线 G-buffer。
- 世界空间**法线偏移**按级联纹素尺寸缩放，缓解自阴影粉刺。
- **斜率自适应深度偏移**：按法线与光方向夹角的 `1/cos`（带上下限）放大 NDC 偏移。
- PCF 核大小可配置（奇数），贴图边界外采样按“无遮挡（亮）”处理且绝不越界。
- 级联边界按屏幕/视图深度距离在 `blend_fraction` 带内线性淡入淡出；
  最后一级不向不存在的下一级混合。选择、吸附、混合共用同一套级联索引。

### 重投影比对

对同一场景分别用**吸附前**与**吸附后**级联矩阵渲染阴影因子图，输出：

- `shadow_factor_raw.npy` / `shadow_factor_snapped.npy`
- `shadow_diff.npy`（绝对差异）与可直接查看的 `shadow_diff.pgm`
- PSNR、最大绝对误差、阴影区域归一化互相关（NCC）

当两张图完全一致（零误差）时，PSNR 明确定义为常量 **99.0 dB**
（`metrics.PSNR_IDENTICAL_DB`），避免数学上的无穷大。

### 深度纹理存档格式

小端二进制，32 字节头部 + float32 像素载荷：

```
magic b"CSMD" | version(u32) | cascade_count(u32) | width(u32)
height(u32) | pixel_dtype(u32=1 float32) | reserved(u32) | reserved2(u32)
payload = cascade_count * height * width 个 float32
```

读取时校验魔数、版本、维度、保留字段与实际字节数；头部损坏返回
`E_ARCHIVE_HEADER`，载荷截断返回 `E_ARCHIVE_TRUNCATED`，文件缺失返回
`E_ARCHIVE_MISSING`，并以非零退出码结束。

## 4. 错误码与退出码

| 错误码 | 含义 | 退出码 |
| --- | --- | --- |
| `E_CONFIG` | 配置非法（级联数、分辨率、近远平面、PCF 核等） | 2 |
| `E_PATH_MISSING` | 相机路径/存档文件缺失 | 2（路径）/3（存档） |
| `E_PATH_FORMAT` | 路径 JSON 损坏或字段非法 | 2 |
| `E_ARCHIVE_HEADER` | 存档魔数/版本/头部字段错误 | 3 |
| `E_ARCHIVE_TRUNCATED` | 存档载荷字节数与头部不符 | 3 |
| `E_INVALID_MATRIX` | 上游矩阵非有限 | 4 |
| `E_INVALID_DEPTH` | 深度图越界或含非有限值 | 4 |
| `E_UNKNOWN` | 其他受控 I/O 异常 | 1 |

任何失败都会在输出目录写出含 `status/error_code/error` 的 `report.json`，
进程返回非零退出码且不产生伪造指标。

## 5. 命令行

```bash
# 生成确定性相机路径
python -m csm_pipeline make-path --seed 20261115 --output path.json

# 运行完整管线
python -m csm_pipeline run \
  --seed 20261115 --camera-path path.json \
  --cascades 4 --shadow-resolution 256 \
  --width 320 --height 240 \
  --camera-near 0.1 --camera-far 40 \
  --frame-time 1.0 --pcf-kernel 3 \
  --blend-fraction 0.15 \
  --normal-bias-scale 0.02 --depth-bias 0.0002 \
  --output-dir out

# 校验并读取深度存档
python -m csm_pipeline read-archive --archive out/shadow_depth_snapped.bin
```

## 6. 测试与验证

```bash
python -m unittest tests.test_pipeline -v
```

测试覆盖：场景位级一致性、矩阵 NDC 约定、级联分割单调性、纹素吸附整数网格、
光栅化深度范围、PSNR 零误差常量、存档往返/截断/坏魔数、端到端复现与报告一致性、
PCF 边界、级联淡入淡出、非法配置与损坏路径的错误码。

## 7. 产物清单

| 文件 | 内容 |
| --- | --- |
| `report.json` | 环境、配置、场景指纹、各级联统计、指标、校验和 |
| `shadow_depth_snapped.bin` | 吸附后多级联 float32 深度图二进制存档 |
| `shadow_factor_raw.npy` | 吸附前阴影因子（0 全影 / 1 全亮） |
| `shadow_factor_snapped.npy` | 吸附后阴影因子 |
| `shadow_diff.npy` / `.pgm` | 两图绝对差异（数组与 8 位可视化） |
| `shadow_factor_snapped.pgm` | 阴影强度可视化（黑=亮，白=阴影） |

