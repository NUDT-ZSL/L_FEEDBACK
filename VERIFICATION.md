# 实际验证记录

验证日期：2026-11-17。以下结果均来自本机实际执行，不是伪实现或预期值。

## 环境准备

```text
Python 3.14.4
NumPy 2.5.5
SciPy 1.18.1
FFmpeg 8.0.1-3ubuntu2
```

系统 Python 没有预装 pip/ensurepip，因此本机使用已有的 `uv` 创建项目虚拟环境：

```bash
uv venv --python python3.14 .venv
uv pip install --python .venv/bin/python numpy scipy
```

编译检查：

```bash
.venv/bin/python -m compileall -q skillgen tests scripts
# COMPILE_OK
```

## 自动化测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

结果：

```text
Ran 8 tests in 14.264s
OK
```

覆盖内容：

- 缺字段、类型/文件路径错误的结构化报告与非零退出码。
- 参数范围、瞬态残余 ≤5 ms、响度误差 ≤0.5 LU、单声道 16-bit PCM 格式。
- 同一配置 `--force` 重跑：清单相等，所有 PCM 逐字节相同。
- 串行与 4 并发：清单逐字节相同。
- FFmpeg 注入非零退出：4 个变体均记录 `E_ENCODE`，进程继续完成；修复后错误项
  被重试并全部成功。
- `SIGKILL` 强杀后恢复：成功项保留原 PID 被跳过，剩余项继续处理。
- 无瞬态恒定信号与“标记附近无瞬态”的端到端配置均返回 `E_TRANSIENT`，并确认
  跳过响度与编码。
- 顶层单技能配置、Unicode 技能 ID、整数值浮点瞬态标记。

## 演示配置

生成夹具：

```bash
.venv/bin/python scripts/generate_demo_fixtures.py demo
```

配置包含 `fireball`、`frost_nova` 两个技能，各 4 个变体，并发数 2。实际运行：

```bash
.venv/bin/python -m skillgen demo/config.json
```

全局摘要：

```json
{
  "all_success": true,
  "error_codes": {},
  "error_count": 0,
  "skill_count": 2,
  "success_count": 8,
  "variant_count": 8
}
```

8 个输出均为 `16-bit / 44100 Hz / 1 channel / 52920 frames`（目标 1.2 秒）。

| 文件 | 音高（半音） | 伸缩 | 截止（Hz） | 瞬态偏差（ms） | 实测响度（MLFS） | 峰值 |
|---|---:|---:|---:|---:|---:|---:|
| fireball_var000.wav | -1.144087 | 0.908918 | 17231.030 | -1.8141 | -20.0184 | 0.983887 |
| fireball_var001.wav | 2.766905 | 1.050041 | 18912.656 | -1.7007 | -20.0122 | 0.981415 |
| fireball_var002.wav | 2.120500 | 0.916525 | 19874.621 | -3.6961 | -20.0144 | 0.982666 |
| fireball_var003.wav | -2.208064 | 1.081174 | 21477.601 | -2.6304 | -20.0217 | 0.982697 |
| frost_nova_var000.wav | 2.311767 | 1.018757 | 21090.470 | -3.3107 | -20.0092 | 0.980316 |
| frost_nova_var001.wav | -2.887314 | 0.941039 | 9939.863 | -3.6054 | -20.0214 | 0.998993 |
| frost_nova_var002.wav | 0.583515 | 0.919319 | 11634.114 | 0.0000 | -20.0291 | 0.998993 |
| frost_nova_var003.wav | 2.517249 | 0.981710 | 2143.527 | 0.0000 | -20.0000 | 0.376160 |

最大瞬态残余为 `3.6961 ms`（阈值 5 ms）；最大响度误差为 `0.0291 LU`
（阈值 0.5 LU）；最大峰值为 `0.998993`（≤1.0）。状态文件为 8 行。

### 重复运行确定性

执行：

```bash
.venv/bin/python -m skillgen demo/config.json --force
```

结果：

```text
manifest_equal True
state_lines 8
```

两次输出的 8 个 PCM SHA-256 全部相同，例如：

```text
fireball_var000.wav  0956685e040b321db89c4f4e82be4c818e74a3c80969f687387d55e8121de3f3
frost_nova_var003.wav 68f1dd4dc974367441c23cfd71eb5d1852a422e3b850c06895362c265b388569
```

### 串行与并发一致

同一夹具分别生成 `concurrency=1` 和 `concurrency=4`：

```text
manifests_equal True
manifest_sha256 51c9522382c047b4ad6e3091e7845ce9debecca98c4ed831f010ac560b71f126
```

`cmp` 返回 0；自动测试逐项比较了全部对应变体 PCM，字节完全相同。

## 强杀恢复演练

使用一个在真实 FFmpeg 后 `sleep 0.45` 的包装器，单技能 6 个变体。任务在状态
文件出现第 1 个成功记录后被 `SIGKILL`（进程组一并终止）：

```text
after_kill_state_lines 1
after_kill_success [(0, 248716)]
```

以相同配置和输出目录重启后：

```text
final_summary {'all_success': True, 'error_codes': {}, 'error_count': 0,
               'skill_count': 1, 'success_count': 6, 'variant_count': 6}
final_state_lines 6
unique_keys 6
resume_evidence [(0, 248716)] -> [(0, 248716), (1, 248883), (2, 248883),
                                  (3, 248883), (4, 248883), (5, 248883)]
temp_dirs_remaining []
```

变体 0 的 `processed_by_pid` 保持第一次运行的 `248716`，证明恢复时直接复用成功
状态而没有重复处理；其余变体由新进程 `248883` 生成。最终状态每个
`(skill_id, variant_index)` 恰好一行，且每条 `variant` 与 `manifest.json` 对应项
相等。强杀遗留的临时目录也在重启时清理。

## 失败继续与重试

自动测试使用包装 FFmpeg：第一次固定输出 `injected encoder failure` 并返回 9。
结果全局摘要是 `E_ENCODE: 4`，所有变体均有错误记录、没有生成 WAV，且临时目录
为空。创建允许标记后以同一配置重跑，4 个错误项全部重新处理成功，最终状态 4 行、
4 个唯一键、全部 `status=success`。
