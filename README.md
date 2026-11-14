# PCSS 多光源阴影渲染器

一个从零实现的 OpenGL 4.6 Core 实时阴影演示程序。场景、材质、网格全部由代码确定性生成，不依赖外部模型或纹理。程序默认创建 1280x720 可交互窗口，并在标准输出记录光源、分辨率、级联、阴影图绑定和 GPU 计时状态。

## 功能对应

- **方向光 CSM**：3 级级联，实用分割（lambda 0.72），级联投影按纹素对齐，级联间有 12% 投影重叠和 16% 视深淡入淡出。
- **点光源立方体阴影**：2 盏点光源，各自 6 个立方体面在 GPU 上写入线性距离深度，运行时只重绑定当前启用的点光源资源。
- **PCSS**：每像素先做 16 次泊松圆盘遮挡物搜索，再根据平均遮挡深度计算半影宽度，最后做 32 次泊松圆盘过滤，共 48 次名义采样。
- **自适应分割**：分割距离由相机近远平面、级联数和 lambda 计算；相机平滑移动时使用光空间中心纹素吸附抑制级联闪烁。
- **动态分辨率**：1024、2048、4096 三档。切换时销毁并重建方向光数组纹理、点光立方体纹理和 FBO，场景与相机状态保持不变。
- **动态偏移**：法线偏移和深度偏移均根据当前阴影图纹素在光源空间中的世界尺寸缩放；方向光和点光共用同一组偏移参数。
- **多光源管理**：1 个方向光加 2 个点光源可同时工作；点光源通过紧凑活动槽位绑定到固定 shader 数组，切换不触发 shader 编译。
- **GPU 统计**：`GL_TIME_ELAPSED` 查询对象延迟两帧轮询，不调用同步等待；屏幕叠加和标准输出同时显示阴影图生成耗时。
- **确定性场景**：固定种子 `0x5EED2026`，程序化生成地面、立方体、球体、圆柱体与固定材质。

## 代码结构

- `src/main.cpp`：窗口、交互、auto-test、灯光槽位绑定、屏幕叠加与截图。
- `src/shadow_renderer.*`：CSM 拟合/吸附、方向光数组阴影图、点光立方体阴影图、FBO 重建。
- `src/shaders.h`：场景光照、PCSS、级联混合和叠加文字 shader。
- `src/camera.*`：平滑轨道相机和实用分割计算。
- `src/scene.*` / `src/mesh.*`：确定性场景与程序化网格。
- `src/gpu_timer.*`：延迟查询 GPU 计时器。
- `src/text_overlay.*`：不依赖字体文件的屏幕文字渲染。

## 构建

### Windows 10/11

前置条件：

- Visual Studio 2022 或更高版本，安装“使用 C++ 的桌面开发”工作负载。
- CMake 3.20+、Git。
- 支持 OpenGL 4.6 Core 的显卡驱动。

首次配置会通过 CMake `FetchContent` 拉取 GLFW 3.3.8 和 GLM 0.9.9.8。GLAD 2 生成的 OpenGL 4.6 Core loader 已放在 `third_party/glad`。

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
.\build\bin\Release\pcss_shadow_renderer.exe
```

若 Visual Studio 使用其他生成器，请按生成器输出目录调整可执行文件路径。

### Linux 验证环境

GLFW 在 Linux 需要 X11 开发头文件和 OpenGL 开发文件：

```bash
sudo apt install build-essential cmake git libx11-dev libxrandr-dev \
  libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bin/pcss_shadow_renderer
```

## 运行与交互

常用命令行参数：

```text
--auto-test             执行确定性的状态切换、PNG 截图和报告输出后退出
--frames N              auto-test 帧数上限
--width N --height N    窗口/帧缓冲尺寸
--resolution N          初始阴影图分辨率：1024、2048、4096
--light-size F          初始点光源直径（方向光直径为其 1.8 倍）
--capture-dir PATH      auto-test 输出目录
--no-vsync              关闭垂直同步
--hide-overlay          隐藏屏幕叠加，便于截取纯场景图
--allow-gl45            仅用于 4.5 软件环境验证；正常目标仍为 OpenGL 4.6
```

交互按键：

| 按键 | 功能 |
| --- | --- |
| `1` / `2` | 启用或禁用点光源 0 / 1 |
| `[` / `]` | 减小 / 增大光源尺寸 |
| `-` / `=` | 阴影图 1024 → 2048 → 4096 循环 |
| `C` | 切换级联着色可视化 |
| `O` | 切换相机自动环绕 |
| `L` | 切换点光源动画 |
| `V` | 显示 / 隐藏屏幕叠加 |
| `S` | 保存当前画面到 `artifacts/screenshots/` |
| `R` | 重置相机、光源和级联可视化 |
| `G` | 立即输出一条运行时状态日志 |
| 鼠标左键拖动 | 轨道旋转相机 |
| 滚轮 | 缩放相机 |
| `W/A/S/D` | 平移相机目标 |
| `Esc` | 退出 |

## PCSS 与偏移实现

方向光 PCSS 半影宽度：

```text
p = lightSize * (zReceiver - zBlocker) / zBlocker
radiusUv = p / (2 * cascadeRadius)
```

点光源 PCSS 半影宽度：

```text
p = lightSize * (distanceReceiver - distanceBlocker) / distanceBlocker
radiusUv = p / (2 * distanceReceiver)
```

因此半影宽度与光源尺寸和遮挡物到接收面的距离成正比。遮挡物搜索半径取光源直径的两倍，并限制在阴影图坐标的合理范围内。

方向光每级联纹素世界尺寸为 `2 * cascadeRadius / resolution`。点光源在接收点处的纹素世界尺寸为 `2 * distanceToLight / resolution`。法线偏移、常量深度偏移和斜率深度偏移都以该世界纹素尺寸为基准，所以两类光源使用同一套视觉参数。

## 级联稳定化

1. 用实用分割公式计算各级视锥片段的近远距离。
2. 对每个片段求包围球，构造稳定的方向光视图基。
3. 将级联中心投影到光空间 XY 平面并吸附到整数纹素。
4. 相邻级联投影故意扩展 12% 重叠；着色时在边界前 16% 区间完成 smoothstep 混合。

## 日志与 GPU 统计

标准输出包含：

```text
[STATE] ... shadow_resolution ... camera_preserved=1 yaw=... pitch=... distance=...
[STATE] ... light_size ... dir_diameter=... point_diameter=...
[BIND] dir_shadow=texture_unit_0 layers=3 slot0=point0(unit1) ...
[STATUS] ... shadow_gpu_ms=... pcss_taps=48 center_cascade=...
[CAPTURE] ... mean_luma=... checksum=...
```

阴影生成本身用一个 `GL_TIME_ELAPSED` 查询包围。每帧只检查更早两帧查询的 `GL_QUERY_RESULT_AVAILABLE`，可用时才读取结果，渲染循环内不会等待 GPU。只有 auto-test 结束时才执行一次 `glFinish` 收集最终样本。

## 自动验证

Linux：

```bash
ALLOW_GL45=1 ./scripts/run_validation.sh
```

Windows PowerShell：

```powershell
.\scripts\run_validation.ps1
```

`auto-test` 会生成以下内容到 `artifacts/auto_test/`：

- 1024 分辨率基准图；
- 小 / 大光源尺寸对比图；
- 2048 分辨率图；
- 相机平滑移动期间的级联调试图；
- 仅方向光、单点光资源切换图；
- 4096 分辨率图和最终双点光图；
- `validation_report.json` 与标准输出日志；
- 仅方向光模式下方向光数组纹理占用率和若干世界坐标投影探针日志。

## 本机实际验证结果

验证运行环境为 WSL2 Ubuntu 26.04 + Mesa 26 llvmpipe。该软件上下文最高为 OpenGL 4.5，因此验证使用显式 `--allow-gl45`；Windows/正常目标路径仍首先要求 OpenGL 4.6 Core。渲染使用的 API 不高于 4.5，4.6 上下文可直接运行。

已验证：

- 1024、2048、4096 三档阴影资源均成功重建；4096 档估算显存约 960 MiB；
- 方向光 3 级 CSM 与 2 个点光立方体阴影同帧生成；
- 点光 0/点光 1 单独禁用和交叉切换时，活动槽位与纹理单元绑定正确；
- 小光源到大光源截图出现可见半影扩展，图像差分和纹理占用探针写入日志；
- 4096 自动序列在 llvmpipe 上约 7.1 秒完成 84 帧，GPU 查询样本正常回读（平均阴影生成 GPU 时间约 22.5 ms，仅作链路验证）；
- 小/大光源对比的平均灰度差为 0.411%，10.151% 的降采样像素发生变化，证明 PCSS 半影不是静态贴图或固定过滤核；
- 全流程无 shader 重编译错误，点光切换未重新编译场景着色器。

本次运行产物：

- `artifacts/auto_test/11_final_both_points.png`：最终双点光效果；
- `artifacts/auto_test/02_pcss_small_1024.png` / `03_pcss_large_1024.png`：PCSS 半影对比；
- `artifacts/auto_test/05_cascade_orbit_a.png`：级联可视化；
- `artifacts/auto_test/07_directional_only.png`：仅方向光；
- `artifacts/auto_test/10_resolution_4096.png`：4096 阴影图；
- `artifacts/auto_test/overlay/01_base_1024.png`：屏幕统计叠加；
- `artifacts/auto_test/stdout.log` / `validation_report.json`：运行日志与机器可读结果。

llvmpipe 的 GPU 毫秒数仅用于证明查询链路和非阻塞回读有效，不代表硬件显卡性能。极端大光源尺寸下固定 48 taps 的 PCSS 边缘会保留一定随机采样噪点，这是实时无降噪 PCSS 的已知代价；程序通过可调光源尺寸和 4096 档分辨率提供质量取舍。
