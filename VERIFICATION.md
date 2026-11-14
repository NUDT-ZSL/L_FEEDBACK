# 实际验证记录

验证环境：

- 日期：2026-11-15
- 系统：Ubuntu 26.04 / WSL，本地 Mesa OpenGL
- Python：3.14.4
- NumPy：2.5.4
- PyOpenGL：3.1.10
- OpenGL：EGL pbuffer，Mesa `llvmpipe (LLVM 21.1.8, 256 bits)`,
  `4.5 (Core Profile)`，shader 使用 `#version 330 core`

## 自动化测试

命令：

```bash
.venv/bin/pytest -q
```

结果：

```text
.............................                                            [100%]
29 passed in 2.31s
```

## 确定性样本

```bash
.venv/bin/python samples/generate_samples.py
.venv/bin/python -m skin_bake bake samples/generated/scene_loop.json -o bakes/loop --quiet
```

样本属性：

- 15x9 布料网格，135 顶点，224 三角形
- 4 骨骼层级，每顶点 4 个非零混合权重
- 顶点 7 使用零长度切线，触发球面采样回退
- 24 fps、2 秒、48 个循环动画帧
- 位移输入为 `delta`，使用 Kahan 补偿累积

烘焙报告关键值：

```text
status                 ok
inputFileSha256        67e9ca5b2d7129012225a3704f571448378685eb00fdcd20cea6a97fc3fa43a2
sceneContentHash       d04af3aa8a96e591d9d74e609829e9b50913a98b0084cf6a16765cbd97c8fe4f
baseSampleRate         60.0
actualSampleRate       32.0
sampleCount            64
downsampleFactor       0.5333333333333333
atlasSize              [128, 128]
frameCount             48
fallbackVertices       [7]
maxProjectionError     4.49e-16
errorTolerance         0.03
loopClosureError       0.0
maxFloat16AbsError     6.10177e-05
```

135 个 padded block 全部位于 128x128 图集内；测试逐对检查无重叠、payload
越界、块外零填充和边缘 texel 扩张。纹理读取重新计算了 payload SHA-256。

## 逐帧预览

软件对照图：

```bash
.venv/bin/python -m skin_bake check samples/generated/scene_loop.json bakes/loop \
  --frame 12 --output previews/software_frame12.png
```

```text
{"atlasSize": [128, 128], "backend": "software-reference", "coveredPixels": 77337,
 "decodedDisplacementAbsMax": 0.13916015625, "fallbackVertices": [7],
 "frame": 12, "frameCount": 48, "frameTime": 0.5, "sampleIndex": 16,
 "preview": "previews/software_frame12.png", "status": "ok"}
```

OpenGL 实际渲染：

```bash
.venv/bin/python -m skin_bake view samples/generated/scene_loop.json bakes/loop \
  --frame 12 --headless-check previews/egl_frame12.png --width 640 --height 480
```

```text
{"backend": "egl-offscreen-opengl", "frame": 12, "glRenderer":
 "llvmpipe (LLVM 21.1.8, 256 bits)",
 "glVersion": "4.5 (Core Profile) Mesa 26.0.8-1ubuntu0.3",
 "nonBackgroundPixels": 48264, "preview": "previews/egl_frame12.png",
 "sampleIndex": 16, "status": "ok"}
```

软件和 OpenGL 两条路径均在帧 12 采样第 16 个曲线 texel，输出非空且图像
不同帧之间发生变化。帧 0/12/24/36 的四联对照图位于
`previews/contact_sheet.png`。Windows 交互查看器使用同一 `GLRenderer`，但上下文由
Win32/WGL 创建；当前环境是 Linux，无法在本机启动 Windows 窗口流程。

