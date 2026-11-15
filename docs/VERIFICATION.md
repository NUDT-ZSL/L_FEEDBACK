# 实际验证记录

## 环境

本次交付在当前本地环境实际执行，不是伪运行：

```text
平台              WSL2 / Ubuntu 26.04 x86_64
Python            3.11.18
OpenGL            4.5 (Core Profile), Mesa 26.0.8-1ubuntu0.3
渲染器            llvmpipe (LLVM 21.1.8, 256 bits)
EGL 后端          egl
numpy             1.26.4
moderngl          5.10.0
glcontext 发行版  2.3.7（模块内版本字符串为 2.3.6）
```

当前执行环境是 WSL2，因此实际 OpenGL 路径为 EGL；Windows 下的 `auto` 后端选择
`wgl`，并使用同一套上下文版本/核心配置校验。Windows WGL 未能在本机 WSL
会话中实机执行。

## 依赖安装

任务要求的直接安装组合与当前 PyPI 元数据冲突：

```text
moderngl 5.10.0 requires glcontext>=2.5.0,<3
task requires glcontext==2.3.7
```

实际采用的精确版本安装命令：

```bash
python -m pip install --no-deps -r requirements.txt
```

## 成功样本

命令：

```bash
python -m render_check --instances 4 --seed 42 \
  --out acceptance_output/final4b --width 96 --height 96
```

结果：

```text
退出码                    0
成功实例                  4
失败实例                  0
通道文件                  16，均为 float32 / [96,96,4]
实例内两遍最大绝对差      0.0
两个独立进程最大绝对差    0.0（16 个文件逐一比较）
CSV 行数                  4
状态快照                  4，均 matches_expected=true
峰值内存                  约 140 MB
总耗时                    约 0.14 s
上下文                    OpenGL 4.5 Core, Mesa llvmpipe
```

ObjectID 通道中每帧非零 ID 集合与实例编号一致：`{1}`、`{2}`、`{3}`、`{4}`。

## 最大批量

命令：

```bash
python -m render_check --instances 64 --seed 2026 \
  --out /tmp/uid125_final64b --width 256 --height 256
```

结果：

```text
退出码                    0
成功实例                  64
失败实例                  0
通道文件                  256
NPY 输出总大小            256.0 MB
CSV 行数                  64
状态快照                  64
实例内两遍最大绝对差      0.0
峰值进程内存              约 156 MB
总耗时                    约 1.38 s
性能限制                  512 MB / 30 s，均未超限
```

## 自动化测试

命令：

```bash
python -m unittest discover -s tests -v
```

实际结果：`Ran 9 tests in 3.251s - OK`。

## 注错验收

| 注入方式 | 业务码 | 退出码 | 实际结果 |
|---|---|---:|---|
| 材质 JSON 删除 `texture` 字段 | `E_MAT_001` | 2 | 通过 |
| 片段着色器引用未声明符号 | `E_GL_001` | 3 | 通过，且未生成通道文件 |
| 顶点/片段 varying 类型不匹配 | `E_GL_002` | 3 | 通过，且未生成通道文件 |
| `--fault-inject state_residue` | `E_STATE_001` | 4 | 通过，报告 `u_metallic` 差异 |
| `--fault-inject diff` | `E_DIFF_001` | 5 | 通过，报告差值约 `0.01` |
| RGBA8 文件声明为 RGBA32F | `E_TEX_001` | 6 | 通过，stderr 含材质名、期望和实际格式 |
| `--memory-limit-mb 0.001` | `E_PERF_001` | 7 | 通过 |
| `--time-limit-sec 0.000000001` | `E_PERF_001` | 7 | 通过 |
| `--instances 0` / `--instances 65` | `E_ARG_001` | 2 | 通过 |

每个失败子进程均检查了 stderr 业务码、整数退出码和 `summary.json` 结构化错误。

