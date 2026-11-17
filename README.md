# 确定性骨骼动画 + 级联阴影离屏渲染器

这是一个从零构建的 C++17 / OpenGL 4.3 离屏渲染程序。程序内部生成低模角色、三级骨骼链、
地面和三个静态遮挡体，以固定相机轨迹逐帧采样骨骼动画、执行线性混合蒙皮（LBS），并把
定向光两级级联阴影、两个点光源和主视图颜色/深度结果渲染到帧缓冲附件。

主视图输出可直接比对的 PNG：

- `frame_<id>_color.png`：RGBA8 颜色图。
- `frame_<id>_depth.png`：16 位灰度深度图，样本为窗口深度 `[0,1]`，0 表示近平面。

每帧还输出级联划分、阴影矩阵、覆盖像素比例、蒙皮矩阵和耗时统计。所有场景输入均来自源码
中的确定性常量与程序化生成逻辑；运行时不读取模型、纹理、网络或其他外部资源。

## 目录

```text
CMakeLists.txt              构建入口，自动使用仓库内的 GLFW/GLM/GLAD/stb
src/
  animation.*               关键帧、四元数、姿势、LBS 与法线数学
  scene.*                   程序化角色/遮挡体/地面/灯光/材质
  cascade.*                 相机视锥 + 可见包围体驱动的 CSM 拟合
  renderer.*                OpenGL 资源、阴影/深度/颜色通道、回读
  gl_context.*              Windows GLFW 与 Linux OSMesa 上下文
  shaders.h                 内嵌 GLSL 430 core 着色器
  image_io.*                颜色 PNG、16 位深度 PNG、FNV 哈希
  main.cpp                  CLI、逐帧模式、JSON 统计、错误码
tests/core_tests.cpp        无 GPU 的确定性数学/网格/级联测试
scripts/validate.sh         Linux 可重复验证脚本
scripts/validate.ps1        Windows 可重复验证脚本
docs/TECHNICAL.md           实现细节与取舍
VALIDATION.md               本次实际验证记录
```

## Windows 构建

前提：

- Windows 10/11 x64。
- Visual Studio 2022，安装“使用 C++ 的桌面开发”工作负载。
- CMake 3.20 或以上。
- 支持 OpenGL 4.3 Core 的显卡与驱动。

GLFW 3.4、GLM 1.0.1、GLAD（OpenGL 4.3 core）和 `stb_image_write` 已放在 `third_party/`，
构建过程不下载依赖。Windows 上下文通过隐藏的 GLFW 窗口获得，渲染目标始终是自建 FBO。

```bat
cmake --preset windows-vs2022
cmake --build --preset windows-release
ctest --preset windows-release-tests
```

`CMakePresets.json` 使用较新的 schema；若本机 CMake 只有 3.20，可直接用等价命令：

```bat
cmake -S . -B build\windows-vs2022 -G "Visual Studio 17 2022" -A x64
cmake --build build\windows-vs2022 --config Release
ctest --test-dir build\windows-vs2022 -C Release --output-on-failure
```

也可以直接运行：

```bat
scripts\build_windows.bat
```

可执行文件默认位于：

```text
build\windows-vs2022\bin\Release\skinned_csm_renderer.exe
```

## 运行

```bat
build\windows-vs2022\bin\Release\skinned_csm_renderer.exe ^
  --output out\demo ^
  --start-frame 0 ^
  --frames 4 ^
  --width 640 ^
  --height 360 ^
  --shadow-size 512 ^
  --cascades 2 ^
  --regenerate-frame 2 ^
  --regenerate-variant 1
```

Windows 下 `--context auto` 使用 GLFW。显式写出 `--context glfw` 也可。
`--regenerate-frame 2 --regenerate-variant 1` 会在同一进程内、渲染第 2 帧之前重建角色
网格并重传顶点/索引缓冲，后续帧继续使用新网格，用来验证热重建路径。
代码接口 `DeterministicScene::rebuildCharacter(const MeshParams&)` 还接受体宽、整体高度、
头部比例和臂展等连续参数，因此无需重启进程即可接入新的网格参数。

常用参数：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--start-frame N` | 第一帧编号 | `0` |
| `--frames N` | 帧数 | `1` |
| `--fps F` | 动画/相机帧率 | `24` |
| `--width/--height` | 主视图分辨率 | `960x540` |
| `--cascades N` | 级联数，允许 `2..4` | `2` |
| `--shadow-size N` | 每级阴影图分辨率，`256..4096` 的 2 次幂 | `1024` |
| `--split-lambda F` | 均匀/对数级联划分混合系数 | `0.25` |
| `--mesh-variant N` | 初始程序化网格变体 | `0` |
| `--regenerate-frame N` | 在该帧前重建角色网格 | 禁用 |
| `--regenerate-variant N` | 热重建使用的网格变体 | `0` |
| `--context auto\|glfw\|osmesa` | 上下文后端 | `auto` |
| `--test-error none\|shader\|framebuffer` | 仅用于故障注入测试 | `none` |

完整参数：`skinned_csm_renderer --help`。

## 输出

每次运行指定独立输出目录：

```text
<output>/
  frame_000000_color.png
  frame_000000_depth.png
  frame_000001_color.png
  frame_000001_depth.png
  stats/
    frame_000000.json
    frame_000001.json
    summary.json
```

`stats/frame_*.json` 包含：

- 分辨率、相机参数、可见投射体包围盒。
- 每个骨骼的相邻关键帧时间、插值比例、首末帧钳制状态、世界矩阵与蒙皮矩阵。
- 每级级联的近/远视深、光源空间中心、半径、世界单位 texel 大小、阴影矩阵和阴影图占用率。
- 灯光参数和三个材质参数。
- 阴影、主视图深度预通道、颜色通道、回读和整帧耗时。
- 像素 FNV-64 与 CPU 蒙皮位置哈希，便于快速回归。

耗时字段必然受调度影响；像素数据不依赖耗时。

## 退出码

| 退出码 | 含义 |
|---:|---|
| `0` | 成功，GPU 资源在退出前释放 |
| `1` | 未分类内部错误 |
| `2` | 命令行用法错误 |
| `3` | 输出目录不可写或图像/统计文件写入失败 |
| `4` | GLFW/OSMesa 上下文或 OpenGL 版本/函数加载失败 |
| `5` | 着色器编译或链接失败 |
| `6` | 主视图或阴影帧缓冲不完整 |
| `7` | 级联配置非法 |
| `8` | 场景、网格、权重或骨骼数据非法 |

所有预期错误都写入 `stderr`，格式为：

```text
error[E_SYMBOL]: readable message
```

故障注入示例：

```bat
skinned_csm_renderer.exe --output out\shader_error --test-error shader
```

## 确定性验证

同一可执行文件、同一 GPU/驱动、同一组参数下：

- 不使用随机数、系统时间、帧间累积状态或外部文件。
- 关闭混合、MSAA、抖动和依赖实现的采样模式。
- 相机、关键帧、灯光、材质、网格参数全部由整数帧号解析计算。
- 阴影采样使用固定 3x3 PCF 内核；阴影图使用独立纹理，采样边界为白边界。
- PNG 压缩参数固定；深度 PNG 由仓库内确定性编码器写出，不含时间戳。

因此同一参数重复运行的 PNG 应逐字节一致。跨不同 GPU、驱动或 Mesa/Windows 实现不保证
像素级一致，这是 OpenGL 光栅化实现差异的正常边界。

Windows 验证：

```powershell
scripts\validate.ps1
```

Linux/WSL 无窗口验证（使用 OSMesa/llvmpipe，仍运行同一套 GLSL 和渲染代码）：

```bash
OSMESA_ROOT=/path/to/osmesa/prefix ./scripts/validate.sh
```

本次实际验证结果、SHA-256 和错误码见 [VALIDATION.md](VALIDATION.md)。实现细节见
[docs/TECHNICAL.md](docs/TECHNICAL.md)。

## 已有边界

- 两个点光源参与正向着色，但不生成点光源阴影；任务要求的是定向光 CSM。
- 深度 PNG 保存窗口深度，不做线性化；`stats` 保留主视图深度附件和相机近远平面。
- 不实现窗口显示或交互；这是离屏逐帧批处理程序。
- Windows 的 GLFW/WGL 路径由条件编译提供；本工作区的实际运行记录来自 Linux OSMesa，
  因为当前容器没有 Windows 图形会话。Windows 实机可按上面的验收脚本复测。
