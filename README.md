# cutscene-mixer

离线命令行工具：读取 JSON 描述文件与本地 WAV，将多轨音频按视频帧事件精确对齐到输出时间轴，完成重采样、声道转换、交叉淡化、响度归一化与淡入淡出，输出混音 WAV 与同步报告 JSON。

## 环境

- Windows 10/11 64 位（开发验证于 Linux，纯 Python 跨平台）
- Python 3.9+
- 依赖：`pip install numpy soundfile`（测试另需 `pytest`）

## 使用

```bash
# 生成确定性测试样本（WAV + 三个描述文件）
python generate_samples.py --dir samples

# 运行混音
python -m cutscene_mixer --input samples/descriptor.json \
    --output out/mix.wav --report out/report.json --target-lufs -16
```

- `--input` / `--output` / `--report` 支持相对与绝对路径，输出目录自动创建。
- `--target-lufs` 默认 `-16`。

### 退出码

| 退出码 | 含义 |
|---|---|
| 0 | 全部轨加载成功，混音完成 |
| 1 | 存在加载失败轨（报告记 `AUDIO_LOAD_FAIL`），混音与报告仍写出 |
| 2 | 模式校验失败（stderr 输出 `SCHEMA_INVALID` 及失败字段路径）或输入文件不存在/JSON 非法 |

## 描述文件格式

```json
{
  "frame_rate": 30.0,
  "output": {"sample_rate": 48000, "bit_depth": 16, "channels": 2},
  "crossfade_ms": 10.0,
  "fade_ms": 5.0,
  "gain_limits_db": {"min": -20.0, "max": 10.0},
  "tracks": [
    {"id": "dialog", "path": "dialog.wav",
     "events": [{"id": "line_a", "track_sample": 0, "frame": 0}]}
  ]
}
```

校验规则：`frame_rate > 0`；`sample_rate` 为 8000–192000 的整数；`bit_depth ∈ {16, 24}`；`channels ∈ {1, 2}`；`gain_limits_db.min < max`；每轨 `path` 非空、`events` 非空，`track_sample` 为非负整数、`frame` 为非负数。`crossfade_ms`（默认 10）、`fade_ms`（默认 5）、`gain_limits_db`（默认 -20/+10）可省略。

## 处理要点

- **帧→采样换算**：`abs_sample = round_half_up(frame * sample_rate / frame_rate)`，解析、对齐、报告共用 `cutscene_mixer/timeline.py` 中的同一函数。
- **加载**：线性插值重采样到目标采样率；单声道复制为立体声，立体声保留，多于两声道取前两声道；位深在写出时统一为 PCM_16/PCM_24。加载失败记 `AUDIO_LOAD_FAIL` 并跳过该轨。
- **事件对齐**：每个事件把源音频 `track_sample` 锚定到绝对采样位置；片段为相邻事件锚点之间的源音频。重叠区按 `crossfade_ms` 线性交叉淡化，重叠不足时缩减为实际重叠长度；无重叠则直接拼接（时间轴间隙为静音）。
- **淡入淡出**：每片段首尾线性淡变，长度为 `fade_ms` 换算的采样数；片段不足两倍淡变长度时缩减为片段长度一半，不越界。
- **响度**：简化 ITU-R BS.1770 K 加权（两级标准 48 kHz 双二阶滤波器应用于所有采样率，积分响度不做门控）。各轨增益拉向平均响度（相对差 ≤ ±2 LU），再整体归一化到目标 LUFS；每级增益钳制在 `gain_limits_db`（默认 -20 dB ~ +10 dB）。

## 测试与验证

```bash
python -m pytest tests/ -q
```

实测验证结果（本仓库环境，Python 3.14 + numpy 2.5.5 + soundfile 0.14.0）：

- 28 个测试全部通过（换算、模式校验、重采样/声道、淡化缩减、交叉淡化、响度、端到端、退出码）。
- 确定性样本端到端：混音 48000 Hz / 立体声 / PCM_16，时长 168000 采样（3.5 s）与事件时间轴一致；归一化后整体响度 -16.0 LUFS。
- 起音对齐：sfx 事件锚定帧 60 → 绝对采样 96000，混音中 96000 之前为静音，起音出现在锚点后的淡入窗口内。
- 退出码映射实测：成功 0；含缺失轨 1（仍写出混音与报告）；模式非法 2（stderr 输出 `SCHEMA_INVALID` 及字段路径）。
- 同一输入连续两次运行，混音 WAV 与报告字节级一致（确定性）。
