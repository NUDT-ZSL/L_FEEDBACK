# 实际验证记录

验证日期：2026-11-16。工作区：`/home/liang/feedback/uid96`。

## 环境

- Ubuntu 26.04.1 LTS，WSL2。
- GCC 15.2.0，CMake 3.30.5。
- GLFW 3.3.8（仓库内固定源码）。
- GLEW 2.2.0（仓库内固定源码）。
- Mesa OSMesa，llvmpipe，报告 `OpenGL 4.5 (Core Profile) Mesa 25.1.7-1ubuntu3`。
- 渲染上下文为 GLFW Null + OSMesa，无交互窗口。

构建命令：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DOSMESA_ROOT=/home/liang/deps/osmesa-prefix/usr
cmake --build build -j 4
```

构建结果：`offscreen_renderer` 成功生成，GLSL 文件自动复制到构建目录。

## 验收命令

```bash
./tests/acceptance.sh ./build/offscreen_renderer "$PWD" "$PWD/artifacts/acceptance"
```

最终结果：

```text
{"path": ".../artifacts/acceptance/no_lights.png", "width": 320, "height": 180,
 "nonblack_pixels": 14878, "nonblack_fraction": 0.25829861111111113,
 "maximum_channel": 67}
Acceptance checks passed.
```

CTest 同时通过：

```text
1/1 Test #1: renderer_acceptance .............. Passed
100% tests passed, 0 tests failed out of 1
```

验收分辨率为 320x180，种子 1337，预热 2 帧。

## 确定性与状态恢复

基础管线 `configs/pipeline.json` 的 SHA-256：

| 文件 | SHA-256 |
|---|---|
| 进程 A `run_a.png` | `aea128af63ea6943655836522b4ef9a36a168fa7d970984163eb7258b465f57b` |
| 进程 B `run_b.png` | `aea128af63ea6943655836522b4ef9a36a168fa7d970984163eb7258b465f57b` |
| 保存前 `state_before.png` | `aea128af63ea6943655836522b4ef9a36a168fa7d970984163eb7258b465f57b` |
| 同进程恢复 `.restored.png` | `aea128af63ea6943655836522b4ef9a36a168fa7d970984163eb7258b465f57b` |
| 新进程加载 `state.json` | `aea128af63ea6943655836522b4ef9a36a168fa7d970984163eb7258b465f57b` |

重组管线 `configs/pipeline_reconfigured.json`：

| 文件 | SHA-256 |
|---|---|
| 独立进程直接渲染 | `6659e8c416af74cbf9e724a0da2ce63cdcae75624660070b8bcb7fc76ac3f1ca` |
| 同进程基础管线→重组管线 | `6659e8c416af74cbf9e724a0da2ce63cdcae75624660070b8bcb7fc76ac3f1ca` |
| 重组状态保存/恢复后 | `6659e8c416af74cbf9e724a0da2ce63cdcae75624660070b8bcb7fc76ac3f1ca` |

同进程重组移除了 `shadow` 阶段，并将后处理顺序改为 `color_grade → tonemap → bloom → fxaa`；结果与独立进程完全相同，说明资源重建、阶段禁用和效果顺序都生效。

状态文件包含 `version: 1`、四个阶段及完整后处理参数。验收脚本还写入 `version: 999` 的配置，程序正确以 `unsupported_version` 和非零退出码拒绝。

## 计时

基础管线单帧测量（预热 2 帧后）：

| 项目 | 毫秒 |
|---|---:|
| Geometry | 1.288 |
| Shadow | 0.951 |
| Deferred | 0.568 |
| Postprocess | 3.005 |
| Bloom | 1.604 |
| Tonemap | 0.360 |
| FXAA | 0.624 |
| Color Grade | 0.374 |
| **整帧** | **7.269** |

960x540 基础管线整帧约 18.4 ms（软件 llvmpipe，仅供本机参考）。

## 错误样本

未知阶段：

```json
{"exit_code":4,"message":".../unknown_stage.json: unknown pipeline stage 'volumetric_magic'","type":"unknown_stage"}
```

注入非法 `geometry.frag`：

```json
{"exit_code":6,"message":".../geometry.frag (fragment) compilation failed:\n0:2(14): error: syntax error, unexpected '{'","type":"shader_compile"}
```

两次错误均在 stderr 只输出一个 JSON 对象；测试脚本解析并确认了错误类型、具体信息和进程退出码。

## 无光源环境项

使用 `--no-lights` 测得的 PNG 非黑像素比例为 25.83%，最大通道值 67。画面背景和几何体仍可见，说明延迟着色在光源数量为 0 时输出环境项而非纯黑。

## 未做的验证

- 未使用 GPU timer query，耗时来自 CPU 时钟与 `glFinish()`。
- 未运行 Valgrind/ASan 统计驱动内部泄漏；项目对象均使用 RAII，且每次阶段后检查 GL 错误。
- 未在独立 NVIDIA/AMD Windows 驱动上复跑；代码路径为 GLFW 隐藏窗口 + OpenGL 3.3 Core。

