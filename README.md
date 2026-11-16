# UID166 DirectX 11 角色与环境渲染基准

这是一个从零实现的 Windows 原生 DirectX 11 C++20 项目，用于同时展示：

- 内置确定性生成并序列化到本地 JSON 的 5000 物体城市场景；
- 写入本地二进制 `.skan`、由渲染端读回并在顶点着色器中蒙皮的骨骼角色；
- BVH 空间划分、视锥剔除、延迟一帧的遮挡查询和快速相机保守路径；
- 3 级稳定化 CSM、点光源立方体阴影、聚光灯 2D 阴影；
- Windows 线程池并行命令记录、主线程批合并；
- GPU 帧时间驱动的动态分辨率与阴影质量调节；
- 固定步长、无窗口、可重复的 120 帧 JSON 基准。

实时模式使用 Win32 窗口、D3D11 硬件设备、实际 GPU 时间戳查询和硬件遮挡谓词查询。基准模式不创建窗口，使用同一套场景、BVH、剔除、阴影策略和命令记录代码，输出逐帧完整观测数据。

## 构建

### Visual Studio 2022

需要“使用 C++ 的桌面开发”工作负载和 Windows 10/11 SDK。

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

生成：

```text
build\Release\uid166.exe
build\Release\uid166_core_tests.exe
```

也可以直接使用：

```powershell
cmake --preset windows-vs2022
cmake --build --preset windows-release
```

### 跨平台核心验证

核心库、基准和测试不依赖 Windows API，可在 Linux/GCC 或 Clang 下构建：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --build build --target test
```

## 运行

### 实时窗口

```powershell
.\build\Release\uid166.exe --width 1600 --height 900
```

首次运行会在 `artifacts\generated` 下生成并重新读取：

```text
uid166_scene.json
uid166_character.skan
```

窗口每帧向 stdout 输出一行 JSON 统计。按 `Esc` 退出。测试或 CI 可使用 `--live-frames N` 自动退出：

```powershell
.\build\Release\uid166.exe --live-frames 120 --width 1280 --height 720 --reuse-data
```

### 5000 物体确定性基准

```powershell
.\build\Release\uid166.exe --benchmark --frames 120 --objects 5000 --seed 166 ^
  --data-dir artifacts\generated --output artifacts\uid166_benchmark.json
```

输出文件包含：

- 每帧视锥可见、遮挡剔除、最终可见物体的数量和完整 ID 集合；
- 3 个 CSM 级联的分割距离、阴影矩阵、更新时间统计；
- 每个点光源的 6 个立方体面矩阵、每个聚光灯的 1 个矩阵；
- 局部光源的更新/脏/可见/预算延迟标志和来源位；
- 实际 CSM/局光源更新数、批合并后的绘制调用数；
- 命令记录总耗时、记录预算、超限标志和每个线程负载；
- GPU 帧时间模型值、渲染比例、阴影比例、最近 120 帧调节历史；
- 场景/动画完整性哈希、逐帧哈希和总统计哈希。

### 只生成数据

```powershell
.\build\Release\uid166.exe --generate-only --data-dir artifacts\generated
```

### 复用已有数据与错误恢复验证

```powershell
.\build\Release\uid166.exe --benchmark --frames 1 --reuse-data ^
  --data-dir artifacts\generated --output artifacts\one_frame.json
```

缺失、损坏或 schema 错误的数据会写入 stderr，例如：

```text
[ERROR] code=2001 name=SCENE_MISSING message=file not found: ...
[ERROR] code=2103 name=ANIMATION_CORRUPT message=invalid animation payload size
```

Windows 进程返回完整业务码（如 `2001`、`2103`）；POSIX 主机会把退出码截断到 8 位，但 stderr 中的完整业务码不变。

## 命令行

```text
--benchmark / --headless   无窗口固定帧基准
--frames N                 基准帧数，默认 120
--objects N                生成物体数，默认 5000
--seed N                   生成器种子，默认 166
--data-dir DIR             场景/动画数据目录
--output FILE              基准 JSON 路径
--reuse-data               不重新生成，只加载已有文件
--record-budget-us N       并行记录阶段预算，默认 1800 微秒
--workers N                记录 worker 数，默认 4
--no-gpu-load              关闭基准中的确定性负载尖峰模拟
--width N --height N       实时窗口客户区尺寸
--live-frames N            实时模式渲染 N 帧后自动退出；0 表示持续运行
--generate-only            仅生成并回读数据后退出
```

## 代码结构

```text
src/core/        平台无关生成、BVH、剔除、阴影、线程记录、基准
src/win/         Win32 主循环、D3D11 渲染器、HLSL
tests/           1447 项核心断言
tools/           基准 JSON 与错误码验证脚本
artifacts/       实际生成的场景、动画和基准输出
docs/            架构与实际验证记录
```

详细设计见 `docs/DESIGN.md`，实际命令与结果见 `docs/VALIDATION.md`。

## 验证摘要

- Linux 核心测试：1447 checks / 0 failures。
- Linux 与 Windows 的 120 帧/5000 物体基准 `frameStatsHash` 均为 `-5703625736082534301`。
- 两份场景 JSON 和角色 `.skan` SHA-256 完全一致。
- 固定相机窗口第 46–90 帧共 45 帧，三个 CSM 矩阵逐元素一致。
- 4 个局部光源每帧最多更新 2 个，120 帧共 240 次局部阴影更新。
- 自适应质量从默认 1.0 降到 0.8，负载降低后恢复 1.0。
- 真实 Windows D3D11 实时模式连续渲染 120 帧，退出码 0，GPU 时间戳与硬件遮挡查询均产生数据。

## 说明

- 基准模式的 GPU 帧时间使用确定性模型，以便无 GPU 环境也能稳定验证质量调节；实时模式使用 D3D11 timestamp 查询。
- 基准中的遮挡结果是基于上一帧低分辨率深度层次和延迟查询状态的确定性模型；实时模式额外使用真实 `D3D11_QUERY_OCCLUSION_PREDICATE`。
- CSM 稳定化使用固定参考纹理尺寸，因此动态阴影分辨率变化不会破坏固定相机/光源下的矩阵稳定性。
- 角色动画浮点在写盘时量化到 `1e-6`，帧统计哈希中的矩阵量化到 `1e-4`，从而让 Linux 与 Windows 的关键哈希一致。

