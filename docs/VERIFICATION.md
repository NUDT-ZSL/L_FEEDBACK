# 验证记录

验证日期：2026-11-16。环境：Ubuntu 26.04、Python 3.14.4、numpy 2.5.4、
scipy 1.18.1、soundfile 0.14.0 / libsndfile 1.2.2。

## 自动化测试

执行：

```bash
python -m pytest -q
```

实际结果：

```text
.........                                                                [100%]
9 passed
```

测试由代码在 `tmp_path` 内确定性生成素材，不依赖外部附件：

| 用例 | 实际验证点 |
|---|---|
| BS.1770 校准 | -20 dBFS 峰值/1 kHz 单声道正弦测得 -23.00 LUFS；复制为双声道为 -20.00 LUFS |
| 端到端多采样率 | 生成 44.1k、48k、22.05k、37k 和损坏 WAV；正确转换前三者，37k 与损坏文件被隔离 |
| 缓存 | 同一 44.1k 文件被两个事件引用，`soundfile.read` 仅调用一次，报告 `load_count=1` |
| 对齐/长度 | 首个事件在 100 ms，输出前 4800 帧静音，从第 4800 帧起有信号；最晚事件结束为 2.100 s / 100800 帧 |
| 重叠 | 两个渲染片段在同一轨重叠 0.5 s，不重叠区为 1.0/0.5，重叠区精确为 1.5 |
| 静音/独奏 | JSON 初始静音生效；`--solo sfx` 后仅 SFX 可听；测试音频非独奏区 RMS < 1e-6 |
| 独奏优先 | “静音+独奏”轨道可听，其他非独奏轨不可听 |
| 响度/报告 | 最终整合响度距目标不超过 0.5 LU，报告含真峰值、短时序列和固定参数 |
| 限幅 | 高瞬态混音触发软限幅，报告记录每个区间的轨道、事件、时间、峰值和贡献者；最终真峰值 <= 0 dBTP |
| 异常/致命错误 | 缺失素材、坏文件、非法时间戳返回事件错误但主流程继续；坏 JSON 返回 3 且 stderr 含 `fatal:` |
| 全无效 | 无有效事件时生成 0 帧 24-bit WAV，报告状态 `silent`，不崩溃 |
| 外置清单 | `asset_manifest` 相对时间轴解析，素材片段 `offset_ms`/`duration_ms` 正确选取，输出 0.750 s |

## 命令行冒烟结果

使用三个不同采样率素材与多个合法/非法事件运行默认 `-23 LUFS`：

```text
wrote .../out/mix.wav (3.000s), -23.00 LUFS; report .../out/loudness_report.json
exit=0
```

WAV 实测：

```text
samplerate: 48000 Hz
channels: 2
duration: 3.000 s
format: WAV (Microsoft) [WAV]
subtype: Signed 24 bit PCM [PCM_24]
```

报告关键值：

```text
target.achieved = true
integrated_lufs = -23.0
asset_cache.load_attempts = 3
asset_cache.cache_hits = 1
tracks: sfx audible, amb muted, vo audible
errors: negative timestamp, missing asset, timestamp outside duration
```

高瞬态素材、`--target-lufs -14` 的限幅验证：

```text
target.achieved = true
integrated_lufs = -14.00226
true_peak_dbtp = -0.01
limiter.enabled = true
limiter.trigger_count = 6
每个触发均带 track=..., event_id=hitN, time_s=...
```

## 兼容性检查

对全部源码和测试执行 Python 3.9 `ast.parse(..., feature_version=(3, 9))`，
均通过。运行依赖范围声明在 `pyproject.toml`，入口脚本为
`game-audio-mixer`，也提供 `python mixer.py` 兼容入口。
