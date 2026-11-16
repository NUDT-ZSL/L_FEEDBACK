# 实测记录

验证日期：2026-11-17

## 环境

- 宿主环境：WSL/Linux（代码使用 `pathlib`、`os.replace` 和标准 RIFF PCM，
  路径参数已按含空格路径验证；未在本轮环境中运行原生 Windows）
- Python 3.14.4
- NumPy 2.5.4
- FFmpeg 8.0.1 可用，但 WAV 流程未调用

语法和代码保持 Python 3.9 兼容，未使用 `X | Y` 类型语法或 3.10+
专用结构。

## 确定性样本

命令：

```bash
python -m lipalign generate --seed 314159 --out "/tmp/lipalign-smoke/sample dir"
```

结果：

```text
generated sentences=5 languages=3 seed=314159 duration_sec=12.236 elapsed_sec=0.676
```

生成文件：

```text
timeline.json          15,524 bytes
audio/en.wav        1,174,654 bytes
audio/zh.wav        1,174,654 bytes
audio/es.wav        1,174,654 bytes
background.wav      1,174,654 bytes
manifest.json           1,099 bytes
```

对同一命令在两个不同目录重新生成后，`timeline.json`、三个语种 WAV、
`background.wav` 和 `manifest.json` 的 SHA-256 全部逐文件一致。

## WSOLA 端到端

```bash
python -m lipalign align \
  --timeline ".../sample dir/timeline.json" \
  --audio-dir ".../sample dir/audio" \
  --background ".../sample dir/background.wav" \
  --algorithm wsola \
  --output ".../out dir/multitrack.wav" \
  --report ".../out dir/report.json" \
  --cache-dir ".../cache dir"
```

结果：

```text
aligned sentences=15 failed=0 cache_hits=0 cache_recomputes=15 elapsed_sec=3.499
```

容器校验：

```text
codec=pcm_s16le  sample_rate=48000  channels=4  duration_ts=587305
timeline duration_samples=587305
duration error=0 samples (0 ms)
```

报告状态为 `ok`，15 个句子全部含目标/实际起止、VAD 源起止、伸缩因子、
算法和缓存状态。背景声道 RMS 约 `0.0055`，语音声道均有效。

对同一缓存立即重跑：

```text
aligned sentences=15 failed=0 cache_hits=15 cache_recomputes=0 elapsed_sec=0.063
```

两次生成的多轨 WAV 文件 SHA-256 相同。

## 增量与缓存失效

修改英语 WAV 第一句中间的一个 PCM 采样值，再以同一缓存目录运行三语种对齐：

```text
aligned sentences=15 failed=0 cache_hits=14 cache_recomputes=1
```

报告中唯一重算项为被修改的英语第一句。

保持相同音频与缓存，将 `--algorithm` 从 `wsola` 切换为 `phase_vocoder`：

```text
aligned sentences=15 failed=0 cache_hits=0 cache_recomputes=15 elapsed_sec=0.280
```

以同一缓存目录运行 44.1 kHz 的新样本：

```text
aligned sentences=5 failed=0 cache_hits=0 cache_recomputes=5
```

说明采样率变化会使相关缓存失效。

## 相位声码器

同一 48 kHz 样本以 `--algorithm phase_vocoder` 完成全部 15 句：

```text
aligned sentences=15 failed=0 cache_hits=0 cache_recomputes=15
output: channels=4 sample_rate=48000 duration_ts=587305
```

单元测试中，440 Hz 纯音从 1 秒压缩到 0.5 秒或拉伸到 1.5 秒后，WSOLA
和相位声码器输出主频分别为 439.2 Hz 和 439.2 Hz，均在 2 Hz 误差内，
且输出采样数精确等于目标值。

## 异常输入

### 坏 JSON

退出码 `2`，stderr 一行：

```json
{"code":"E_JSON_INVALID","message":"Invalid JSON at line 1 column 2: Expecting property name enclosed in double quotes: ...","ok":false}
```

### 截断 WAV

退出码 `2`，stderr 一行：

```json
{"code":"E_WAV_CORRUPT","message":"RIFF size exceeds file size: ...","ok":false}
```

### 采样率不匹配

退出码 `2`，stderr 一行：

```json
{"code":"E_RATE_MISMATCH","details":{"actual_rate":44100,"expected_rate":48000,"language":"en"},"message":"Language sample rate 44100 Hz does not match timeline 48000 Hz","ok":false}
```

### 单句缺失

将英语第三句对应区域的 PCM 清零后运行：

```text
stdout: aligned sentences=4 failed=1 cache_hits=0 cache_recomputes=4
stderr: {"code":"E_PARTIAL_FAILURE", ...}
exit: 6
cache entries retained: 4
report status: partial
```

其余 4 句成功，报告、多轨 WAV 和 4 个成功片段缓存均保留。

## 自动化测试

命令：

```bash
python -m unittest discover -v
```

结果：

```text
Ran 14 tests in 13.958s
OK
```
