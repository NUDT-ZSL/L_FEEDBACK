# Validation

验证日期：2026-11-15  
环境：WSL2 / Ubuntu 26.04，GCC 15.2，CMake 3.30.5，Mesa 26.0.8（llvmpipe），EGL 1.5。

## CPU 构建与确定性

```text
$ cmake --build build/cpu -j 4
[100%] Built target character_renderer

$ character_renderer --backend cpu --render
rendered backend=cpu output=build/acceptance/output/render.png
reference=build/acceptance/output/reference_cpu.png (created)
elapsed=7.43 sec

$ character_renderer --backend cpu --verify
verify=ok backend=cpu tolerance=1/255
output=build/acceptance/output/verify.png
reference=build/acceptance/output/reference_cpu.png (matched)
elapsed=2.44 sec

$ character_renderer --backend cpu --verify
verify=ok backend=cpu tolerance=1/255
```

生成样本统计：

```text
texture_count=52
material_count=5
viewport=640x480
seed=0x51A7C0DE9E3779B9
```

CPU 输出与参考图 SHA-256：

```text
e2381a05626d2e93abc0c31dd71ebc737d27c70b728e72fc81e79c8863e2c6c4  render.png
e2381a05626d2e93abc0c31dd71ebc737d27c70b728e72fc81e79c8863e2c6c4  reference_cpu.png
```

图像：`docs/render_cpu.png`。

## OpenGL 4.3 离屏渲染

使用与 Windows GLFW 路径相同的 `renderOpenGLDeferredWithCurrentContext` 核心，通过 Mesa surfaceless EGL 创建 OpenGL 4.3 core 上下文，实际执行 MRT G-Buffer 与全屏光照 pass。

```text
EGL 1.5
wrote build/acceptance/output/gl_render.png
cpu_vs_gl_max_channel_diff=211
channels_over_1=11083
elapsed=0.37 sec
```

同一场景连续两次 OpenGL 输出：

```text
6101cdbeab5043e5e64cc95ccc43d601142a4e6156519b129c010a95f8d70bb6  gl_render_first.png
6101cdbeab5043e5e64cc95ccc43d601142a4e6156519b129c010a95f8d70bb6  gl_render.png
```

图像：`docs/render_opengl_llvmpipe.png`。CPU 与 OpenGL 视觉结构一致；跨后端数值不要求逐像素相同，`--verify` 在单个后端内执行。

Windows MSVC + GLFW 的上下文创建路径已在源码中实现，但本次验证环境为 WSL/Linux，未直接运行 `character_renderer.exe`。OpenGL 核心、函数加载器和 GLSL 均已实际编译并在 llvmpipe 上执行。

## 集成验收

执行：

```text
$ cmake -DTEST_EXE=... -DTEST_ROOT=... -P tests/run_integration.cmake
-- integration tests passed
```

覆盖并通过的项目：

- 首次渲染生成 52 张纹理、5 个材质 JSON、G-Buffer 渲染图与参考图。
- 两次独立进程 `--verify` 均匹配 `1/255` 容差。
- 创建快照 2/3，`--diff 2 3` 报告 `key.intensity` 从 28 变为 40。
- 回滚和差异前后，快照 2/3 的 SHA-256 不变。
- `--merge 2 3` 将当前 `key.intensity` 更新为 40。
- 当前值改为 35 后合并同一差异，stderr 输出 `E_SNAPSHOT_CONFLICT`，当前状态文件 SHA-256 不变。
- 回滚不存在版本 999，stderr 输出 `E_SNAPSHOT_NOT_FOUND`。
- 差异正确报告新增光源 `extra` 与后续删除。
- 非法 `blendMode` 与缺失纹理分别输出 `E_MAT_PARSE`，有效材质继续加载且退出码为 0。
- PNG 损坏输出 `E_TEX_LOAD`，退出码 1。
- snapshots 目录只读时创建快照输出 `E_IO_DENIED`，退出码 1。
- 纹理文件无读取权限时输出 `E_IO_DENIED`，退出码 1。
- 无提交日志记录的 `snapshot_000099.json` 不被恢复。
- 删除当前状态文件后，从最后已提交快照恢复。
- 场景光源已改变时，`--verify` 与旧参考图比较失败并输出 `E_RENDER_NONDETERMINISTIC`。

实际错误输出示例：

```text
E_MAT_PARSE: material 'broken_blend' layer 0 has illegal blendMode 'screen'
E_TEX_LOAD: failed to decode texture '.../m0_l0_skin_base_basecolor.png': unknown image type
E_IO_DENIED: cannot create temporary file '.../snapshot_000004.json.tmp...'
E_SNAPSHOT_CONFLICT: cannot modify light 'key' field 'intensity': current value differs from baseline
E_SNAPSHOT_NOT_FOUND: snapshot version 999 is not committed
E_RENDER_NONDETERMINISTIC: render differs from saved reference: pixel (316,36) channel 0 differs by 5
```

所有业务错误进程退出码均为 `1`，未使用业务码本身作为退出码。

