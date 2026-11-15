# 实际验证记录

验证日期：2026-11-16  
当前执行环境：Ubuntu 26.04 WSL，x86-64，24 核，约 7.6 GiB 内存  
工具链：GCC 15.2.0，CMake 3.30.5，GLFW 3.4，Mesa llvmpipe，OpenGL 4.5 Core

本环境没有 MSVC，因此未在 Windows 上重新编译；Windows 专用分支已配置
`psapi` 链接和 vcpkg/GLFW 查找，构建步骤见 `README.md`。以下均为非容器环境中的
实际构建和运行结果。

## 构建与 CPU 测试

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH=<local-glfw-prefix> -DSUBDIV_FETCH_GLFW=OFF
cmake --build build -j 8
./build/subdiv_tests
```

结果：

```text
[PASS] all deterministic CPU tests passed
```

额外以 ASan/UBSan 重新编译并运行 CPU 测试：

```text
[PASS] all deterministic CPU tests passed
```

没有 sanitizer 报告或泄漏报告。

## 图形上下文与渲染

在 X11 + Mesa llvmpipe 下获得：

```text
[gl] context=4.5 renderer=llvmpipe (LLVM 21.1.8, 256 bits) \
version=4.5 (Core Profile) Mesa 26.0.8-1ubuntu0.3
```

级别 0：

```text
[stats] level=0 original_triangles=48 subdivided_quads=0 subdivision_ms=0.008 \
shader=compiled+linked vertices=35 gpu_triangles=48 boundary_edges=20 \
crease_edges=33 traceable_vertices=35 rss_mib=138.7
```

级别 1：

```text
[stats] level=1 original_triangles=48 subdivided_quads=144 subdivision_ms=0.172 \
shader=compiled+linked vertices=165 gpu_triangles=288 boundary_edges=40 \
crease_edges=66 traceable_vertices=165 rss_mib=141.1
```

级别 3：

```text
[stats] level=3 original_triangles=48 subdivided_quads=2304 subdivision_ms=2.768 \
shader=compiled+linked vertices=2385 gpu_triangles=4608 boundary_edges=160 \
crease_edges=264 traceable_vertices=2385 rss_mib=139.4
```

几何指纹在不同进程和多次构造中一致：

```text
[mesh] seed=20251029 vertices=35 triangles=48 edges=82 \
fingerprint=0xdad9c0b8c17ace49
```

## 资源切换压力测试

命令：

```bash
./build/subdivision_preview --scene scene.json --stress-switches 1000
```

结果：

- 完成 1001 次统计（初始级别加 1000 次切换）。
- 进程退出码 `0`。
- 日志中没有 `GL_INVALID_*`、`fatal` 或 `warning`。
- 首轮 RSS 为 `138.9 MiB`。
- 第 1、201、401、601、801、1001 轮 RSS 稳定在 `142.9 MiB`。
- 首次进入 4 级时因分配器高水位短暂升至约 `149.2 - 149.4 MiB`，离开 4 级后
  回落到 `142.9 MiB`；后续每次进入 4 级均保持同一水平，不随循环次数累积。
- 最终退出 RSS 为 `142.9 MiB`，没有随切换次数线性增长。

每轮切换后都执行 `glGetError` 排空检查；任何错误会使程序写入 stderr 并返回 `7`。

## 失败路径

### 无可用图形显示

```text
DISPLAY=:199 WAYLAND_DISPLAY=wayland-99 ./build/subdivision_preview --frames 1
```

退出码：`2`

```text
[fatal] graphical display preflight failed: no reachable X11 or Wayland display
(DISPLAY=':199', WAYLAND_DISPLAY='wayland-99')
```

### 无法创建 OpenGL 4.1 上下文

通过 Mesa 限制可用版本：

```text
MESA_GL_VERSION_OVERRIDE=2.1 ./build/subdivision_preview --frames 1
```

退出码：`3`

```text
[glfw] error 65543: GLX: Failed to create context: GLXBadFBConfig
[fatal] GLFW window or OpenGL context creation failed
```

### 着色器编译失败

通过测试注入 `SUBDIV_FORCE_SHADER_ERROR=1` 强制失败：

退出码：`5`

```text
[fatal] shader compilation/link failed:
forced shader failure via SUBDIV_FORCE_SHADER_ERROR
```

### 场景文件缺失

```text
./build/subdivision_preview --scene does_not_exist.json --frames 1
```

退出码：`0`，程序使用全部内置默认值继续运行，并逐字段打印警告，例如：

```text
[scene] warning: cannot read scene file 'does_not_exist.json': unable to open file; using all built-in defaults
[scene] warning: field 'seed' missing; using default 20251029
[scene] warning: field 'material.roughness' missing; using default 0.360000
```

## 截图

以下截图均由最终程序直接从帧缓冲读取并写入 PPM，再无损转为 PNG：

| 文件 | 内容 |
| --- | --- |
| `artifacts/level0_shaded.png` | 未细分三角网格，明显低多边形轮廓 |
| `artifacts/level0_face_normals.png` | 未细分的面法线模式，三角面分块清晰 |
| `artifacts/level3_shaded.png` | 3 级细分，平滑轮廓和表面着色 |
| `artifacts/level3_vertex_normals.png` | 顶点法线颜色模式 |
| `artifacts/level3_face_normals.png` | 面法线颜色模式，可见四边形分块 |
| `artifacts/preview_level1.png` | 默认场景初始 1 级细分 |

截图尺寸均为 `1280 x 760`。

