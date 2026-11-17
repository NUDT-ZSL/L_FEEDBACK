# 实际验证记录

验证日期：2026-11-17（UTC）

本次执行环境：

```text
Linux WSL2 x86_64
Intel Core i7-14650HX
CMake 3.30.5
g++ 15.2.0
OSMesa / Mesa llvmpipe
OpenGL 4.5 Core context（程序要求 4.3）
```

当前工作区没有 Windows 图形会话，因此以下像素/退出码结果是在 Linux OSMesa 路径上实际
执行得到。Windows 的 GLFW/WGL 上下文代码路径通过 `WIN32` 条件编译提供；在 Windows
实机上可执行 `scripts/validate.ps1` 复测同一套渲染核心。

## 构建与 CPU 测试

```bash
cmake -S . -B build/linux \
  -DCMAKE_BUILD_TYPE=Release \
  -DOSMESA_ROOT=/home/liang/deps/osmesa-prefix/usr \
  -DSR_WARNINGS_AS_ERRORS=ON
cmake --build build/linux -j2
build/linux/bin/sr_core_tests
```

结果：

```text
All 193 checks passed
```

覆盖点：

- 四元数最短弧、单位化。
- 不规则关键帧查找、相邻帧线性/球面插值、首末帧钳制。
- 三级骨骼世界矩阵和绑定姿势单位蒙皮。
- 权重归一化、全零权重回退、LBS 位置。
- 非均匀缩放下逆转置法线，与切向保持正交。
- 程序化网格确定性、权重和、索引、最多 4 影响、进程内重建。
- 两级级联划分递增、切片角点在各自光源投影内、矩阵逐位重复一致。
- 16 位深度 PNG 签名/位深/重复编码哈希。

## 两次确定性渲染

参数：

```text
--start-frame 0
--frames 4
--width 640
--height 360
--shadow-size 512
--cascades 2
--regenerate-frame 2
--regenerate-variant 1
--context osmesa
```

两次运行均退出 `0`。所有 8 个 PNG 通过 `cmp` 逐字节比较，SHA-256 完全一致：

```text
4cd1859b76af3f6261824455f518fa7815e048274eb7123be2d1feb244d82599  frame_000000_color.png
14d3f8733db60c4cc31dac79db622fbc88d9f25a31a41671b037fc5a951d98f1  frame_000000_depth.png
7d656d15b2e424ee2774f71eec86c6512af5be7ab2d981ea3def3d86c493ab40  frame_000001_color.png
c64c1ac9d7612f35d76c495f5a76ea9836f220d2d6c4a73b467b34a7bfbb32e8  frame_000001_depth.png
c7020aee91c10d9780411455051178fd9724563b9bd22745d6d3318d73229297  frame_000002_color.png
bdcb2eca4e1fe117d4676a2ecd687e4f449f9ca2b32a268eff810e502895aa04  frame_000002_depth.png
de1c3fdd52a592763c9be5c0df0935915e1c6d697521c6e80b1a83380ce5ed43  frame_000003_color.png
2ceed2a89a2382466352e876961c3598ba199a7d45b9a8b9a5ba9b66f20926b2  frame_000003_depth.png
```

帧 2 的日志确认：

```text
Mesh rebuilt in-process before frame 2 (revision 1, variant 1)
```

`file` 检查：

```text
frame_*.png: PNG image data, 640 x 360, 8-bit/color RGBA
frame_*_depth.png: PNG image data, 640 x 360, 16-bit grayscale
```

## 级联覆盖

帧 2 的统计：

```text
frame 2
cascade 0 view depth: [0.450000, 5.272187]
cascade 0 occupied pixels: 10.7357%
cascade 1 view depth: [5.272187, 12.056339]
cascade 1 occupied pixels: 2.8931%
```

两级都有非空阴影图，说明划分不是只给近级联空白范围。CPU 级联测试还检查了每个切片 8 个
角点在对应光源投影中的 X/Y/Z 全部落在 `[-1,1]`。阴影纹理由独立纹理承载并使用
`CLAMP_TO_BORDER`，不存在图集跨级串采。

## 失败路径

| 场景 | 命令要点 | 实际退出码 | stderr 首行 |
|---|---:|---:|---|
| 未知参数 | `--unknown-option` | 2 | `error[E_USAGE]: unknown option --unknown-option` |
| 输出不可写 | `--output /dev/null/not-a-dir` | 3 | `error[E_OUTPUT_IO]: cannot create output directory: ... (Not a directory)` |
| 着色器编译失败 | `--test-error shader` | 5 | `error[E_SHADER_COMPILE]: shader compilation failed for skinned_forward: ...` |
| 帧缓冲不完整 | `--test-error framebuffer` | 6 | `error[E_FRAMEBUFFER]: ... GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT` |
| 级联配置非法 | `--cascades 1` | 7 | `error[E_CASCADE_CONFIG]: cascade count must be between 2 and 4` |

这些失败路径均返回非零码，且 `stderr` 非空、可读。

## 可重复执行

Linux/WSL：

```bash
./scripts/validate.sh
```

Windows PowerShell：

```powershell
scripts\validate.ps1
```

脚本会构建、运行 CPU 测试、执行两次渲染、逐字节比较 PNG、检查 16 位深度 PNG，并验证上述
错误退出码。Linux 的原始验证记录保存在
`artifacts/validation/validation_summary.txt`。
