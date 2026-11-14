# 实际验证记录

验证日期：2026-11-15（Asia/Shanghai）

## 环境

| 项目 | 值 |
|---|---|
| 宿主 | WSL2 Ubuntu 26.04 |
| 编译器 | GCC 15.2.0 |
| CMake | 3.30.5 |
| 图形栈 | WSLg / Mesa 26.0.8 |
| OpenGL | 4.5 (Core Profile), GLSL 4.50 |
| Renderer | llvmpipe (LLVM 21.1.8, 256 bits) |
| Windows/MSVC | 当前工作区未安装，未执行原生 Windows 运行 |

说明：工程已提供 Visual Studio 2022 x64 preset、vcpkg manifest 和 Windows 脚本。当前机器只有 Linux/WSL 工具链，因此这里记录的是同一 C++/OpenGL 代码路径的实际构建与 GPU（软件光栅器）运行结果，而不是对 MSVC 执行结果的虚构声明。

## 构建与核心测试

```text
cmake --build build -j2
./build/bin/lf_core_tests artifacts/core-test-report.json
```

结果：

```text
PASS: 9570 checks, 0 failures
```

覆盖内容：

- 12 张纹理和 `lighting.json` 两次生成逐字节一致；
- 权重越界、重复灯光 ID、缺失纹理引用、错误层序、非法锥角、非法向量均返回 `E_CONFIG_INVALID`；
- 热重载非法配置返回 `E_CONFIG_RELOAD_INVALID`，旧灯光和旧材质保持不变；
- 快照 v2 往返字段完全一致；
- v1 自动迁移并产生 8 条字段迁移日志；
- v99 返回 `E_SNAPSHOT_VERSION_UNSUPPORTED`；
- 截断 JSON、空灯光数组返回 `E_SNAPSHOT_CORRUPT`；
- 3135 顶点、5400 三角形、18 骨骼，蒙皮权重归一化，walk 姿态改变蒙皮矩阵；
- 三层 CPU BRDF 均有限、非负且结果互不相同。

报告：`artifacts/core-test-report.json`

## OpenGL 像素验收

命令：

```text
./build/bin/lf_character_engine --verify \
  --data-dir artifacts/runtime --width 640 --height 480
```

结果：`allPassed = true`

| 检查 | 结果 |
|---|---|
| 点光/方向光/聚光均存在 | true |
| CPU 三层 BRDF 有限 | true |
| 角色渲染可见且非均匀 | true |
| 快照保存成功 | true |
| 修改状态后像素哈希改变 | true |
| 保存/恢复逐像素一致 | true，`differingBytes = 0` / `307200` |
| 三层 BRDF 渲染图像不同 | true |

保存帧、修改后帧和恢复帧哈希：

```text
saved     = 68b3c16dc971248b
mutated   = 5305e0071259b705
restored  = 68b3c16dc971248b
```

纯层权重图像哈希：

```text
diffuse      = 19208e78f3166349
ggx_specular = b6c2b5642ced8d94
cloth_sheen  = 438d2a3cc4e2f4b6
```

报告：`artifacts/runtime/artifacts/verification.json`

可视帧：

- `artifacts/runtime/artifacts/snapshot_saved_frame.png`
- `artifacts/runtime/artifacts/snapshot_restored_frame.png`
- `artifacts/runtime/artifacts/layer_diffuse.png`
- `artifacts/runtime/artifacts/layer_ggx_specular.png`
- `artifacts/runtime/artifacts/layer_cloth_sheen.png`

## 运行时故障注入

### 非法配置热重载

程序运行中把点光颜色改为 `[2.0, -0.5, 0.1]`：

```text
[ERROR] [E_CONFIG_RELOAD_INVALID:1102] configuration reload rejected;
previous valid state retained: E_CONFIG_INVALID: configuration validation failed:
$.lights[0].color: vector component outside allowed range
```

程序继续运行到正常窗口关闭，进程退出码 0。

### 着色器编译失败

向 `gbuffer.vert` 注入非法 GLSL：

```text
[ERROR] [E_SHADER_COMPILE:1004] shader compile failed (.../gbuffer.vert):
0:49(1): error: syntax error, unexpected NEW_IDENTIFIER, expecting end of file
```

进程走统一异常处理并以业务码 1004 退出（Linux 8-bit shell 显示为 236）。

### 纹理损坏降级

把 `cloth_basecolor.tga` 截断为 6 字节：

```text
[ERROR] [E_TEXTURE_LOAD:1006] texture header is truncated:
.../cloth_basecolor.tga; using deterministic fallback texture
```

程序继续完成渲染，进程退出码 0。

### 窗口创建失败

窗口创建/上下文失败路径由 `E_RENDER_WINDOW_CREATE`、`E_RENDER_CONTEXT_CREATE` 和最外层统一异常处理覆盖。当前 WSLg/GLFW 在指向无效 X display 时会阻塞连接，未取得可重复的自动化退出记录，因此本文件不把它列为已完成的实际运行项。

## 结果文件

- `artifacts/core-test-report.json`
- `artifacts/runtime/artifacts/verification.json`
- `artifacts/runtime/artifacts/*.png`
- `artifacts/runtime/logs/lf_engine.log`
- `artifacts/fault_reload/logs/lf_engine.log`
- `artifacts/fault_texture/logs/lf_engine.log`

