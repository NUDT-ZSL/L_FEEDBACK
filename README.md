# 方言配音批量替换工具（dub-replace）

本地命令行工具：按作业清单批量处理过场动画目录，用方言配音替换**目标语言配音声道**，
采用不变音高的 WSOLA 时间伸缩使对白与字幕/口型事件对齐；背景音效与其他语言声道按原
时间轴逐样本保留，输出元数据声明布局（mono / stereo / 5.1）的交错 PCM 16-bit WAV。

## 环境要求（Windows 10/11）

- Python 3.10 或 3.11（已配置 pip）
- FFmpeg 可执行文件，其 `bin` 目录已加入 `PATH`（用于 WAV 校验与方言片段重采样）
- 无需外网；样本由 numpy 确定性生成（48 kHz，PCM 16-bit）

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
# 校验
ffmpeg -version
```

## 快速开始

```powershell
# 1. 生成确定性样本作业（含 stereo / 5.1 / 损坏文件三类动画）
python scripts\generate_sample_job.py

# 2. 运行批处理（退出码 0=全部成功，1=至少一个动画失败，2=作业级致命错误）
python -m dub_replace.cli samples\sample_job\job_manifest.json

# 3. 再次运行：已成功动画自动跳过，失败动画重试
python -m dub_replace.cli samples\sample_job\job_manifest.json
```

也可安装为命令：`pip install -e .` 后使用 `dub-replace <manifest>`。

## 作业清单与元数据

作业清单 `job_manifest.json`：

```json
{
  "job_dir": "out",
  "concurrency": 2,
  "animations": [
    {"animation_id": "cutscene_a", "meta": "cutscene_a/meta.json"}
  ]
}
```

动画目录 `meta.json` 关键字段：`audio.original_track`、`audio.sample_rate`、
`audio.channel_layout`（`mono` / `stereo` / `5.1`）、`audio.channels`（逐声道
`role`：`sfx` / `target_voice` / `other_voice`）、`dialogue`（逐句
`line_id/start_ms/end_ms/text`）、`subtitles`、`lip_events`、`dialect_clips`
（`line_id` 可选、`start_ms` 可选、`path`）、`stretch_limits`（默认 0.5–2.0）。

## 行为约定

- **缺失方言句**：沿用原句音轨，报告 `method=fallback` 并记录 `E_MISSING_AUDIO` 警告。
- **多余方言片段**：带未知 `line_id` 或匹配后仍剩余的片段，按时间轴最近距离丢弃，
  写入 `discarded_clips`（含 `nearest_line_id` 与 `distance_ms`）和警告。
- **伸缩越界**：比例钳制在 `stretch_limits` 内；方言过长则裁尾、过短则尾部补静，
  句子与报告均记录 `E_STRETCH` 事件，绝不静默越界。
- **并发与隔离**：`ThreadPoolExecutor` 按 `concurrency` 并行；FFmpeg 调用加进程内锁且
  使用独立临时文件；单个动画失败只影响自身，其余继续。
- **中断恢复**：状态写入 `out/job_state.json`（临时文件 + 原子替换），带 `version`；
  版本不符或文件损坏时丢弃旧状态。成功项重跑跳过，失败/未完成项重试。
- **报告**：每个动画 `out/reports/<id>.json`，作业汇总 `out/job_summary.json`；
  时间均为毫秒整数，峰值电平 dBFS 保留两位小数（数字静音记为 -120.00）。

## 错误码

| 码 | 触发场景 | 级别 |
| --- | --- | --- |
| `E_MISSING_AUDIO` | 原始/方言 WAV 缺失 | 方言缺失为警告；原始缺失致命 |
| `E_SAMPLE_RATE` | 采样率与元数据不符（方言自动 FFmpeg 重采样并警告；原始音轨致命） | 警告/致命 |
| `E_CHANNEL` | 声道数/布局与声明不一致 | 致命 |
| `E_OVERLAP` | 台词片段时间重叠 | 致命 |
| `E_STRETCH` | 伸缩比超出限制，已裁尾/补静 | 警告 |
| `E_CORRUPT` | WAV 无法解析或 FFmpeg 校验失败 | 致命（跳过该动画） |
| `E_METADATA` / `E_FFMPEG` / `E_INTERNAL` | 元数据非法 / FFmpeg 缺失 / 未预期异常 | 致命 |

每个错误附 `hint` 可操作恢复建议，同时写入 stderr 与失败报告。

## 测试

```powershell
python -m pytest -q
```

覆盖 WSOLA 保音高与长度、伸缩钳制、元数据/重叠/布局校验、端到端混音（仅目标声道
变化）、采样率重采样、fallback/丢弃、错误分类、状态版本与原子恢复、SIGKILL 中断重启、
CLI 退出码与并发隔离。

详见 [TECHNICAL_NOTES.md](TECHNICAL_NOTES.md)。
