# 实际验证记录

验证日期：2026-11-15（Asia/Shanghai）

## 环境

| 项目 | 值 |
|---|---|
| 运行环境 | WSL2 Ubuntu 26.04（x86_64） |
| Python | 3.14.4 |
| moderngl | 5.13.0 |
| NumPy | 2.5.4 |
| PyGLM | 2.8.3 |
| Pillow | 13.0.0 |
| OpenGL context | 4.5 Core Profile |
| GL_VENDOR | Mesa |
| GL_RENDERER | llvmpipe (LLVM 21.1.8, 256 bits) |
| backend | EGL standalone |

本机为无显示软件渲染路径；代码使用 OpenGL 4.1 core 特性，Windows 目标使用同一 shader/渲染路径下的 WGL standalone context。本文的逐字节结果来自同一机器、同一驱动和同一参数。

## 样本和命令

样本场景 SHA-256：

```text
c02dbcb94dfcf7b3df3518c886a5fe659c818a9fc73c77245f1b42416582217f
```

样本规格：`480x270`，7 帧，`0..6 s`，`fps=1`，32 个体积步，`768x768` 光空间 ID/深度图，8 个凸多面体建筑，5 个玻璃面板（其中一个固定背面面板）。

```bash
python -m fogcity generate-sample --path scenes/sample_city.json \
  --width 480 --height 270 --fps 1 --start-time 0 --end-time 6 \
  --steps 32 --shadow-resolution 768

python -m fogcity render --scene scenes/sample_city.json \
  --out validation_outputs/run_a --parallel-frames 1

python -m fogcity render --scene scenes/sample_city.json \
  --out validation_outputs/run_b --parallel-frames 2

python -m fogcity render --scene scenes/sample_city.json \
  --out validation_outputs/run_a --validate-only --parallel-frames 1
```

## 逐帧 PNG 字节一致性

`run_a`（串行）与 `run_b`（2 帧并行）的所有 PNG SHA-256 完全一致：

```text
frame_0000.png  98ba5ccba01e3c393b664dc25bddc8bdec404f4bebeb46c879f4efe36bfc527c
frame_0001.png  7246f5c35a12e69a61dffcc0c535ec02fa7d36654dc4f13b139d8bcd0fa862bb
frame_0002.png  643acaeca46f725c4fa05f95e0f4bf95594f2d9170b90176569356b758679556
frame_0003.png  4bd6090655e86e59c130b95aba0371713181e7fba46ccb6a598ff271ed9b1b8d
frame_0004.png  021bf12a31ed1eda555158eb730d82ae074bb75ce7036ff803bace3efe0e237c
frame_0005.png  b652eebc042089988995851323a5298047032c66663fded1d7bf24d2fb8b2047
frame_0006.png  ae3f6221e90f20f01d6fe06964c792ad9f29f17962364672825d5576c7539525
```

两个 `stats.json` 移除 `generated_at` 后深度相等。逐帧 JSON 无需移除字段即完全一致。

## 只读校验

```text
{"files_verified":15,"frames_verified":7,"mode":"validate-only","ok":true,...}
```

退出码为 0。校验前后对 `run_a` 中全部文件的大小、mtime 和 SHA-256 做快照，结果完全相同，确认校验模式未写文件。

将 `frame_0001.png` 人为追加字节后重新校验：

```json
{"error":{"code":"E_VERIFY_MISMATCH","exit_code":41,...},"ok":false}
```

退出码为 41。

## 非法输入

| 输入 | 退出码 | 业务错误码 |
|---|---:|---|
| 场景文件不存在 | 2 | `E_SCENE_NOT_FOUND` |
| `render.width` 为字符串 | 10 | `E_SCENE_FIELD` |
| 关键帧时间重复 | 12 | `E_KEYFRAME_TIME_DUPLICATE` |
| 关键帧时间倒退 | 11 | `E_KEYFRAME_TIME_ORDER` |
| 校验目标目录不存在 | 40 | `E_VERIFY_OUTPUT_MISSING` |

所有错误均以单行结构化 JSON 写入 stderr，没有 Python traceback 或未捕获异常。

## 自动测试

```text
Ran 11 tests in 2.931s
OK
```

覆盖：

- 重复/倒退关键帧时间错误码。
- 字段非法的业务错误。
- 四元数最短路径 slerp 与归一化。
- 等视深面板按 ID 次序键排序。
- 统计浮点固定量化。
- 单帧失败带帧序号并停止。
- 两次运行 PNG 逐字节一致（串行和 2 并行均一致）。
- 逐帧统计与聚合统计可复算。
- 校验模式只读且不一致时返回 41。
- 样本生成和中间帧位置复算。

## 资源稳定性

在 `96x54`、40 帧循环中，每帧创建并释放整套临时附件。进程 RSS：

```text
frame 0:   139.22 MiB
frame 4:   139.49 MiB
frame 9:   139.61 MiB
frame 19:  139.64 MiB
frame 39:  139.65 MiB
after close: 139.58 MiB
```

达到稳定平台后没有随帧数持续增长。

## 产物

实际验证输出保留在：

- `validation_outputs/run_a/`：串行 7 帧及统计。
- `validation_outputs/run_b/`：2 并行 7 帧及统计。
- `scenes/sample_city.json`：确定性样本场景。

