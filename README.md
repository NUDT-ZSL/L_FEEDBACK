# 漆面分层评审渲染器

一个面向汽车造型评审的 Windows 桌面工具。几何、HDR 环境、微表面颗粒和光照均由代码确定性生成，不读取外部模型或贴图。应用使用 NumPy 在 CPU 上完成三角形光栅化与分层 BRDF 着色，使用 PyQt5 提供实时预览、相机交互和离屏出图控制。

## 运行

要求 64 位 Python 3.9+，并确保 `python` 与 `pip` 在 PATH 中。

```powershell
python -m pip install -r requirements.txt
python main.py
```

Windows 也可以双击 `run_windows.bat`。不需要 GPU、OpenGL 扩展、模型文件或环境贴图文件。

## 主要能力

- 程序化 4×4 控制点 Bezier 引擎盖曲面，默认 96×96 细分、18,432 个三角形；解析切线/法线用于平滑着色。
- 软件三角形光栅化，包含屏幕包围盒、透视正确重心插值、重心 UV/法线/切线插值和深度缓冲。
- 程序化 512×256 经纬度 HDR 环境，含太阳亮斑、大面积柔光箱和冷暖轮廓光；初始化时生成 8 级预过滤环境。
- 线性 HDR 分层 BRDF：
  - 清漆层使用 GGX 分布、Smith 几何遮蔽和 Schlick Fresnel，并采样预过滤环境的镜面级；
  - 底色层由漫反射与金属反射按金属度混合；
  - 两层用清漆 Fresnel 反射率和厚度透射率耦合；
  - 渲染完成后统一做 ACES 色调映射、sRGB 伽马和 8 位量化。
- 1024×1024 确定性颗粒法线图。金属颗粒使用随机圆盘内的各向同性微表面倾斜；珠光颗粒按颗粒法线与视线夹角，在基础色相和旋转色相之间混合。
- 素色漆、金属漆、珠光漆三个预设。切换时参数面板、预览和窗口标题同步更新。
- 拖拽旋转相机，滚轮缩放；参数变化由 Qt `valueChanged` / 信号直接触发预览任务，不使用定时轮询。
- 后台工作线程输出 1920×1080 PNG，主线程保持响应并显示进度。切换预设或关闭窗口时取消在途任务，等待线程结束后再释放预过滤环境和图像资源。
- 输出目录内同时写入 PNG 和 `render_log.jsonl`，日志含参数哈希、材质哈希、相机哈希、渲染统计、环境预过滤信息及实际 NumPy/Python/Pillow 版本。

## 参数与预设

面板提供以下可实时调整参数：

- 基础色 RGB
- 金属度
- 粗糙度
- 清漆层厚度
- 颗粒密度
- 珠光色相偏移
- 曝光
- 颗粒随机种子

内置预设位于 `paint_review/params.py`：

| 预设 | 类型 | 金属度 | 粗糙度 | 颗粒 |
| --- | --- | ---: | ---: | ---: |
| 素色漆 | solid | 0.00 | 0.24 | 关闭 |
| 金属漆 | metallic | 0.88 | 0.22 | 开启 |
| 珠光漆 | pearl | 0.42 | 0.18 | 开启 |

## 确定性设计

1. Bezier 控制点、环境亮斑位置、环境纹理和颗粒纹理均由固定公式生成。
2. 颗粒使用坐标与 `grain_seed` 的整数哈希，不使用依赖平台状态的全局随机流。
3. 渲染全部在线性 `float64` 空间中完成，最后一步才做 ACES 与 sRGB 变换。
4. 环境预过滤只在渲染器初始化时执行；粗糙度变化只选择已缓存级别，不会逐帧重建。
5. 参数和相机使用规范化 JSON 的 SHA-256。材质哈希只含材质参数；出图哈希额外包含相机和分辨率。
6. 输出文件名为：

```text
paint_review_<本地时间戳>_<出图哈希前12位>.png
```

同目录 `render_log.jsonl` 每行是一份完整 JSON 记录，包含 `params_hash`、`material_hash`、`camera_hash`、`stats`、`environment` 和 `software`。

## 关键模块

| 文件 | 作用 |
| --- | --- |
| `paint_review/geometry.py` | Bezier 曲面求值、切线与法线、网格生成 |
| `paint_review/environment.py` | 程序化 HDR 环境、经纬度采样、8 级预过滤 |
| `paint_review/grains.py` | 确定性金属/珠光颗粒法线图与缓存 |
| `paint_review/renderer.py` | 光栅化、分层 BRDF、IBL、色调映射和统计 |
| `paint_review/export.py` | 后台出图线程、确定性 PNG 和 JSONL 日志 |
| `paint_review/preview.py` | 仅保留最新请求的预览工作线程 |
| `paint_review/ui.py` | PyQt5 面板、预览画布、相机和生命周期控制 |
| `main.py` | 桌面应用入口 |

## 着色摘要

环境预过滤级别由 `roughness` 选择。底色层镜面按反射向量 `R` 从对应粗糙度级别采样，清漆镜面从第 0 级采样。

```text
F_base = F0 + (1 - F0) * (1 - N·V)^5
L_base = L_prefilter(R_base, roughness) * F_base
       + (1 - metallic) * base_color * L_ibl(N) * (1 - F_base)

F_coat = 0.04 + 0.96 * (1 - N·V)^5
L_coat = L_prefilter(R_coat, 0.055) * F_coat * 0.72
       + GGX_direct(N, V, sun, roughness=0.055)

L_out = L_coat + (1 - F_coat) * exp(-0.135 * thickness) * L_base
```

珠光颗粒的有效色相混合权重由颗粒是否激活、颗粒法线与视线夹角共同决定；未激活区域保持底色。

## 测试与验证

运行全部自动化测试：

```powershell
python -m unittest discover -s tests -v
```

测试覆盖：

- 参数/材质/相机哈希规范化；
- 96×96 最小曲面细分与向上法线；
- HDR 多级预过滤及独立构建一致性；
- 1024×1024 颗粒图确定性；
- 两个独立渲染器输出逐像素一致；
- PNG 编解码后逐像素一致；
- 光栅化中途取消及资源释放；
- GGX 数值有限性；
- 后台 `ExportJob` 写出 PNG/JSONL、日志软件版本和重复出图像素一致；
- 切换预设时取消在途出图并重建资源；
- 关闭窗口时取消在途出图且不残留线程。

本交付环境的实际结果（2026-11-16，Python 3.14.4，NumPy 2.5.4，Pillow 13.0.0，Linux/WSL2 离屏验证）：

```text
Ran 11 tests in 4.286s
OK
```

完整 1920×1080 验证：

```powershell
python tools/verify_delivery.py
```

生成结果在 `artifacts/verification/`，汇总在 `artifacts/verification_summary.json`。本次实测：

| 预设 | 渲染耗时 | 光栅耗时 | 覆盖像素 | 环境级别 | 颗粒图 |
| --- | ---: | ---: | ---: | ---: | --- |
| 素色漆 | 2252.2 ms | 937.2 ms | 743,220 | 8 | - |
| 金属漆 | 2442.5 ms | 925.5 ms | 743,220 | 8 | 1024×1024 |
| 珠光漆 | 2315.7 ms | 980.1 ms | 743,220 | 8 | 1024×1024 |

金属漆由第二个独立渲染器重新渲染 1920×1080，像素 SHA-256 均为 `16918276de4c9ea8...`，`independent_repeat_pixel_identical = true`。4 次出图对应 `render_log.jsonl` 的 4 条记录，日志中实际记录的 NumPy 版本为 `2.5.4`。

Linux 离屏环境仅用于自动化验证；应用本身只使用 PyQt5、NumPy 和 Pillow 的跨平台接口，目标运行环境为 Windows 桌面。

## 线程与资源生命周期

- 预览和离屏出图均在独立 `QThread` 中执行，主线程只更新 UI。
- 预览线程维护“最新请求覆盖旧请求”的条件变量队列；新参数会取消旧帧，不轮询。
- 出图线程通过 `threading.Event` 在光栅化、背景采样和着色分块处检查取消。
- 切换预设时：禁用面板、请求取消、等待 `QThread.finished`，随后关闭旧渲染器、清空颗粒缓存和 8 级预过滤数据，再创建新渲染器并恢复预览。
- 关闭窗口时若出图仍在进行：先忽略关闭事件并显示取消提示；线程结束后释放资源，再真正关闭窗口。

## 性能说明

- 预览为 640×360，默认同参数下约 1 秒（本机 Python 3.14 + NumPy 单线程）。
- 1920×1080 出图约 2.2–2.5 秒。
- 主线程不参与出图像素计算，因此出图时参数面板和相机交互仍然可用。
- 若机器差异导致性能不同，仍不影响同一 NumPy/硬件浮点组合下的像素复现；跨机器比对建议同时归档日志中的 NumPy 版本和渲染统计。

