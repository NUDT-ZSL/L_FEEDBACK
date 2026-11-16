# skillgen

离线、确定性的技能音效变体生成器。输入 JSON 配置和短 WAV 基音样本，输出一组
16-bit PCM WAV 变体、`manifest.json` 清单和可恢复的 `state.jsonl` 状态文件。

核心处理链路：

1. 严格校验配置、路径、WAV 可解码性、瞬态标记范围和 FFmpeg 可执行文件。
2. 用 `SHA-256(工具版本 | 技能ID | 种子 | 变体序号)` 派生 PCG64 随机序列，
   生成音高偏移、时间伸缩和二阶低通截止频率。
3. 相位声码器/多相重采样完成音高和时长处理，`sosfiltfilt` 实现零相位二阶
   低通；结果裁剪或补零到目标时长。
4. 用能量起振检测首个瞬态，通过与参考标记的采样点差做线性插值平移，迭代到
   ±5 ms 内。
5. 使用 BS.1770 风格 K 加权、400 ms 重叠分块、绝对/相对门限测量积分响度
   （MLFS/LUFS 数值体系），施加增益并迭代校准到 ±0.5 LU。
6. 超峰值时使用平滑软限幅；随后写出临时 float WAV，由 FFmpeg 编码为
   `pcm_s16le`，重新读取 PCM 验证帧数、瞬态和响度，再原子替换到最终路径。

## 环境

- Python 3.10+
- NumPy、SciPy
- FFmpeg 5.0+ 在 `PATH`，或在配置中用 `ffmpeg` 指定路径

```bash
uv venv --python python3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
ffmpeg -version
```

若系统 Python 自带 pip，也可使用 `python3 -m venv .venv` 后执行
`.venv/bin/python -m pip install -r requirements.txt`。

本仓库验证环境为 Python 3.14.4、NumPy 2.5.5、SciPy 1.18.1、FFmpeg 8.0.1。

## 快速运行

```bash
# 确定性生成本地短 WAV 和 demo/config.json
.venv/bin/python scripts/generate_demo_fixtures.py demo

# 生成变体、清单和状态
.venv/bin/python -m skillgen demo/config.json

# 忽略状态并重新生成所有变体
.venv/bin/python -m skillgen demo/config.json --force
```

也可以用已安装的入口：

```bash
uv pip install --python .venv/bin/python -e .
.venv/bin/skillgen demo/config.json
```

正常完成时 stdout 输出同一份清单 JSON；变体级错误不会中止其他变体。默认退出码：

- `0`：批处理完成（清单中可能含变体错误）
- `1`：使用 `--fail-on-variant-error` 且存在变体错误
- `2`：配置/路径/FFmpeg 预检失败
- `3`：内部未预期错误
- `4`：输出目录正被另一个进程占用

## 配置格式

路径均相对配置文件所在目录解析。

```json
{
  "version": 1,
  "output_dir": "out",
  "temp_dir": "tmp",
  "concurrency": 2,
  "ffmpeg": "ffmpeg",
  "skills": [
    {
      "skill_id": "fireball",
      "base_sample": "assets/fireball_base.wav",
      "variant_count": 4,
      "target_duration": 1.2,
      "transient_markers": [5292],
      "target_loudness_mlfs": -20.0,
      "random_seed": 1234
    }
  ]
}
```

必填字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `skill_id` | string | 非空文件名片段；不允许路径分隔符、控制字符或 `.`/`..` |
| `base_sample` | string | WAV 路径 |
| `variant_count` | int | 变体数，>= 1 |
| `target_duration` | number | 输出时长（秒） |
| `transient_markers` | int[] | 基音样本中的瞬态采样点；首项为对齐参考 |
| `target_loudness_mlfs` | number | 目标积分响度 |
| `random_seed` | int | 确定性随机种子 |

`skills` 必填且非空；`output_dir`、`temp_dir`、`concurrency`（1~8）和
`ffmpeg` 可选。兼容常见 camelCase 字段名，以及 `transients`、
`target_loudness`、`seed` 等别名。`target_duration_ms` 会按毫秒换算。若 JSON
顶层直接包含单个技能字段，也可省略 `skills` 数组。

## 错误码

| 错误码 | 行为 |
|---|---|
| `E_CONFIG` | 配置缺字段、类型错误、范围错误、WAV 不可解码或 FFmpeg 不可用；退出码 2 |
| `E_MISSING_FILE` | 配置文件或基音样本不存在；退出码 2 |
| `E_TRANSIENT` | 瞬态未检出，或线性补偿后仍超过 ±5 ms；跳过响度校准与编码 |
| `E_ENCODE` | FFmpeg 启动失败、非零退出、超时或 PCM 帧数异常；继续其他变体 |
| `E_LOUDNESS` | 响度不可测或编码后仍超出 ±0.5 LU |
| `E_PROCESSING` | 变体级未预期处理异常，保留其他变体继续执行 |
| `E_BUSY` | 同一输出目录已有活跃进程 |

配置级错误同时写入 `<output_dir>/error_report.json`，并在 stdout 输出结构化报告：

```json
{
  "version": "1.0",
  "status": "error",
  "error_code": "E_CONFIG",
  "message": "...",
  "issues": [{"code": "E_CONFIG", "path": "skills[0].variant_count", "message": "..."}],
  "exit_code": 2
}
```

## 输出

变体文件名固定为 `{skill_id}_var{序号:03d}.wav`，格式为单声道 16-bit PCM，
采样率继承基音样本，帧数精确等于 `round(target_duration * sample_rate)`。

`manifest.json` 按配置顺序排列技能和变体，不含 PID、耗时、临时路径或完成顺序，
所以串行与并发模式的清单逐项/逐字节一致。每个变体记录：

- 参数：音高半音、时间伸缩、低通截止、增益
- 瞬态：参考点、预测点、补偿前/后偏差、对齐次数
- 响度：目标、调整前、实测、增益
- 峰值、是否软限幅、PCM SHA-256、错误码/错误摘要

全局摘要包含技能数、变体数、成功数、错误数、错误码计数和 `all_success`。

## 恢复与并发

- 每个变体完成后立即向 `<output_dir>/state.jsonl` 追加一条完整记录并 `fsync`。
- 成功记录包含配置指纹、基音哈希、输出 SHA-256；重启时只有指纹匹配、文件存在
  且哈希匹配才跳过。错误记录会重新处理。
- 最终状态文件被原子重写为每个变体恰好一行，顺序与清单一致；强杀留下的半行会
  在加载时忽略。
- 输出目录用 `.skillgen.lock` 排他锁保护；临时目录使用
  `skillgen_tmp_<输出目录哈希>_<技能哈希>_<序号>`，失败/成功均清理，重启会清理
  同一输出目录留下的临时目录。
- 并发只跨技能并行；同一技能的变体顺序处理，便于状态恢复与稳定输出。

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

测试使用 `tests/fixture_factory.py` 确定性生成 44.1 kHz 短样本，覆盖：

- 配置字段缺失/文件不存在及结构化退出
- 参数范围、瞬态 ±5 ms、响度 ±0.5 LU、16-bit PCM 格式
- 同配置 `--force` 重跑 PCM 逐样本一致
- 串行/并发清单逐字节一致
- FFmpeg 注入失败继续处理，修复后可重试错误变体
- `SIGKILL` 强杀后重启，成功记录保留原 PID，未完成项继续处理
- 无瞬态信号返回 `E_TRANSIENT`

实际验证结果见 `VERIFICATION.md`。
