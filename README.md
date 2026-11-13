# replaymix

离线电竞复盘音轨工具。它读取本地确定性生成的多路单声道 WAV 与事件表，按
48 kHz 内部时间轴重采样、切片、逐段增益并混缩为一条与对局等长的复盘 WAV，
同时输出 JSON 复盘报告和可恢复检查点。

## 环境

- Python 3.11.18
- NumPy 1.26.0
- WAV 读写仅使用标准库 `wave`

本仓库已准备 `.venv`：

```bash
.venv/bin/python -c 'import sys,numpy; print(sys.version); print(numpy.__version__)'
```

## 快速开始

生成确定性多采样率样本并混缩：

```bash
.venv/bin/python -m replaymix demo --out artifacts/demo
```

生成包含故障音轨的样本（采样率不一致、96 kHz、超出对局、文件缺失、帧数错误）：

```bash
.venv/bin/python -m replaymix demo --out artifacts/demo-faults --faults
```

也可以分步执行：

```bash
.venv/bin/python -m replaymix generate --out artifacts/samples
.venv/bin/python -m replaymix mix \
  --events artifacts/samples/event_table.json \
  --out artifacts/run \
  --concurrency 2
```

默认输出：

- `replay.wav`：48 kHz、单声道、PCM16，帧数为 `round(session_ms * 48)`。
- `report.json`：每路状态、错误码、片段增益、零填充、混缩帧数和检查点复用信息。
- `checkpoint.jsonl`：版本化检查点。
- `checkpoint_cache/`：已完成音轨的后增益片段音频缓存。

## 事件表格式

`silence_ms` 和 `gaps_ms` 使用**对局绝对毫秒**。`gaps_ms` 优先于
`silence_ms`；断档和静音都生成显式零段。写入区间之外生成 `pre_roll` /
`post_roll` 零段。

```json
{
  "session_id": "local-scrim",
  "session_ms": 12000,
  "tracks": [
    {
      "track_id": 1,
      "player": "IGL",
      "file": "track_01.wav",
      "sample_rate_hz": 44100,
      "channels": 1,
      "frames": 529200,
      "write_start_ms": 0,
      "write_end_ms": 12000,
      "silence_ms": [[2000, 2700]],
      "gaps_ms": [[6500, 6900]],
      "markers": [{"at_ms": 4800, "type": "teamfight"}]
    }
  ],
  "events": [{"at_ms": 4800, "type": "teamfight"}]
}
```

兼容别名包括 `sample_rate`、`num_frames`、`start_ms`/`end_ms`、
`silence_intervals`、`gap_intervals` 等。输入必须为单声道 PCM；支持的采样率为
8000 到 48000 Hz 的整数。实际采样率、声道数、帧数时长与事件表不一致时，该路
被跳过并记录 `E_TRACK_TIMELINE_MISMATCH` 或 `E_SAMPLE_RATE_UNSUPPORTED`，
其他路继续处理。

## 处理口径

- 时间轴：`frames(ms) = round(ms * 48)`；对局总帧数、片段边界、报告和 WAV
  封装共用这一口径。
- 重采样：非 48 kHz 输入使用固定两抽头线性插值。源位置用
  `i * source_rate / 48000` 的整数有理数计算，避免浮点位置累积差异。自然输出帧数
  与事件区间期望帧数偏差必须不超过 1 帧；不足补零、超出截断，均写入报告。
- 增益：每个有声片段独立计算峰值，目标默认 `-1.0 dBFS`，单段增益限制在
  `[0, 20] dB`。按规格只放大不衰减，因此输入峰值本身高于目标时增益为 0 dB，
  并在片段记录中标记 `input_peak_above_target=true`。
- 混缩：按 `track_id` 升序固定累加顺序，算术相加后限幅到 `[-1, 1]`；因此交换
  输入音轨顺序不改变输出样本。总帧数固定为对局帧数，末尾自动零填充。
- 并发：默认 2，实际并发度取 `min(配置值, 音轨数)`。调度窗口最多同时提交该数量
  的任务，完成结果再按音轨 ID 排序汇总。报告记录实际观测到的最大并发数。

## 错误码

| 错误码 | 含义 |
| --- | --- |
| `E_TRACK_TIMELINE_MISMATCH` | 采样率/声道/帧数/区间与事件表不一致 |
| `E_SAMPLE_RATE_UNSUPPORTED` | 采样率不在 8000..48000 Hz 或非整数 |
| `E_TRACK_EXCEEDS_SESSION` | 音频或写入区间超出对局时长；越界部分不进入混缩 |
| `E_TRACK_FILE_MISSING` | WAV 文件缺失 |
| `E_TRACK_WAV_INVALID` | WAV 损坏、非 PCM 或不支持的位宽 |
| `E_TRACK_PROCESSING_FAILED` | 单路未预期异常；其他路继续 |

## 检查点与恢复

`checkpoint.jsonl` 是 JSON Lines 文件：

1. 首行 header：包含格式、`version`、事件表/配置运行签名。
2. 中间每路一行 `track-row`：片段元数据、增益、状态和缓存引用。
3. 末行 footer：`row_offsets` 记录每路行在文件中的字节偏移，并带行 SHA-256。

每完成一路即原子重写检查点并落盘。再次运行时会校验版本、运行签名、行偏移和行
哈希；任一版本或偏移不匹配即清空旧行与缓存并整份重建。每路输入签名还包含 WAV
内容哈希，因此单个音频变化会使该路重算而不是复用旧增益。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests -v
```

测试覆盖：样本字节级确定性、重采样有理数位置与补零、逐段增益上下限、显式断档
零值、输入顺序不变性、业务错误隔离、检查点复用/版本重建/偏移重建、中断恢复、
以及真实并发上限。

2026-11-14 本机实际结果：

- 10 个单元/集成测试全部通过。
- 12,000 ms 对局输出 576,000 帧，48,000 Hz，单声道 PCM16。
- 干净样本复跑 WAV SHA-256 一致：
  `628b266a8a53f112c6057bdb8608a062c9b0d8ad659c26c0a4a0a29ff603e520`。
- 故障样本中 1-4 路正常处理；5/9 路 `E_TRACK_TIMELINE_MISMATCH`，6 路
  `E_SAMPLE_RATE_UNSUPPORTED`，7 路 `E_TRACK_EXCEEDS_SESSION`，8 路
  `E_TRACK_FILE_MISSING`，混缩仍为 576,000 帧。
