# 动画音效时间轴处理工具

离线命令行工具，用于把动画事件标记和循环区间应用到一组 WAV 音效上。工具会统一音频格式、对齐事件能量峰值、生成无缝循环片段、执行可选时间伸缩和响度归一化，并输出处理后的 WAV、注入状态字段的时间轴 JSON 以及处理报告。

运行过程只读取本地文件，不访问网络，也不需要 Docker 或其他容器。

## 环境

- Python 3.9 或以上版本
- NumPy 1.21 或以上版本
- 标准库 `wave`（Python 自带）

安装依赖：

```bash
python -m pip install -r requirements.txt
```

PCM 读取、转换和量化由本工具直接实现，不依赖 `audioop`。因此在仍提供 `audioop` 的 Python 3.9-3.12 和已移除该模块的 Python 3.13+ 上都可以运行。

## 快速开始

生成确定性测试样本：

```bash
python generate_samples.py --output-dir samples
```

运行处理：

```bash
python audio_timeline_tool.py \
  --timeline samples/timeline.json \
  --audio-dir samples/audio \
  --output-dir demo_output
```

Windows PowerShell 中可使用相同参数，按需要把续行符替换为单行命令。

## 输入时间轴

时间轴根对象必须包含 `audio` 数组。每个 audio 对象必须包含 `id`、`file`、`events` 和 `loops`，可选 `time_scale`：

```json
{
  "version": 1,
  "audio": [
    {
      "id": "impact",
      "file": "impact_48k.wav",
      "events": [
        {"name": "hit", "time": 0.2}
      ],
      "loops": [
        [0.1, 0.35]
      ],
      "time_scale": 1.0
    }
  ]
}
```

兼容的输入写法：

- 事件可写成数字 `0.2`，也可写成带 `time`（或 `timestamp`）的对象。
- 循环可写成 `[start, end]`，也可写成 `{"start": 0.1, "end": 0.35}`。
- 数字事件和数组循环在输出中会规范为对象，以便注入逐项状态字段；原有时间值不变。
- `file` 为相对路径时，以 `--audio-dir` 为基准。也允许绝对路径。

`time_scale` 是处理后的时长倍率，范围为闭区间 `[0.1, 10.0]`。例如 `2.0` 表示处理后时长变为原来的两倍。事件与循环时间在最终处理时间轴上解释，即先完成 48 kHz 统一、时间伸缩，再按这些秒数查找事件峰值和提取循环。

## 处理规则

### 格式统一

- 输入支持单声道、未压缩 PCM WAV，采样位宽为 8/16/24/32 位。
- 支持 44100 Hz、48000 Hz 及其他有效采样率。
- 非 PCM、压缩 WAV、立体声等不支持的输入记录 `UNSUPPORTED_FORMAT`，跳过音频处理。
- 重采样到 48000 Hz 时先做加窗 sinc 低通抗混叠/抗镜像滤波，再做线性插值。
- 所有输出均为 48000 Hz、单声道、16 位 PCM。

### 事件峰值对齐

对每个事件时间，在前后各 10 ms 范围内计算逐采样点能量（采样值平方），选择能量最大点。多个采样点能量相同时选择离原事件时间最近的点。

- `aligned_time` 为对齐到的精确采样时刻。
- 报告中的 `offset_ms` 为有符号偏移：`(aligned_time - time) * 1000`。
- 事件早于 0 或超出处理后的音频时长时记录 `EVENT_OUT_OF_RANGE`，`aligned_time` 保留原事件时间。
- 音频整体不可处理时记录 `SKIPPED`，并保留原事件时间。

### 循环片段

循环区间从完成时间伸缩和响度归一化后的音频中提取，输出名称为 `{id}_loop_{index}.wav`，索引从 0 开始。

- 片段开头和结尾各应用 20 ms 互补升余弦淡入/淡出窗，首尾采样为零，形成可用于循环的接缝。
- 区间短于 40 ms 时不生成文件，记录 `LOOP_TOO_SHORT`。
- 区间越界时记录 `SKIPPED` 和诊断码 `LOOP_OUT_OF_RANGE`。

### 时间伸缩

使用线性插值重采样实现。因子小于 `0.1`、大于 `10.0`、非数字或非有限值时：

- 记录 `INVALID_TIME_SCALE`；
- 按 `1.0` 处理该音频；
- 时间轴中的原 `time_scale` 字段保持不变。

### 响度归一化

- 使用数字 RMS 近似目标 `-20 LUFS`，目标 RMS 为 `10^(-20/20) = 0.1` 满幅。
- 峰值限制为不高于 `-1 dBFS`，量化后同样严格检查 PCM 峰值。
- 如果纯 RMS 目标要求超过 20 dB 的衰减，则把增益限制为 `-20 dB`，记录 `LOUDNESS_LIMITED`。
- 当峰值限制比 RMS 目标更严格时，采用峰值限制对应的较小增益。

## 输出

默认在 `--output-dir` 生成：

- `processed_timeline.json`：保留原有字段，并为每个事件注入 `aligned_time`、`status`，为每个循环注入 `loop_file`、`loop_status`。
- `report.json`：逐音频报告原始时长、处理后时长、事件对齐偏移、循环结果、响度信息、诊断码和错误码。
- `{id}_processed.wav`：成功处理音频的主输出。
- `{id}_loop_{index}.wav`：成功生成的循环片段。

事件状态只会是 `OK`、`EVENT_OUT_OF_RANGE` 或 `SKIPPED`。循环状态只会是 `OK`、`LOOP_TOO_SHORT` 或 `SKIPPED`。

报告中的主要诊断码包括：

- `EVENT_OUT_OF_RANGE`
- `LOOP_TOO_SHORT`
- `INVALID_TIME_SCALE`
- `LOUDNESS_LIMITED`
- `UNSUPPORTED_FORMAT`
- `AUDIO_READ_ERROR`
- `LOOP_OUT_OF_RANGE`

## 退出码与错误处理

- 时间轴 JSON 格式或结构错误：退出码 `1`，stderr 输出可读错误。
- 输出目录无法创建、不是目录或不可写：退出码 `1`，stderr 输出可读错误。
- 单个音频缺失、读取失败或解码失败：记录 `AUDIO_READ_ERROR`，保留该音频的原事件时间，继续处理其他音频，最终退出码为 `0`。
- 不支持格式同样是逐音频错误，不会使整体退出码变为 `1`。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖：

- 命令行正常处理及输出文件；
- 事件状态、循环状态和原字段保留；
- 精确采样峰值对齐及偏移；
- 44100 Hz 重采样、时间伸缩和时长；
- 48 kHz/单声道/16 位输出与峰值限制；
- RMS 响度目标；
- 循环首尾接缝采样为零；
- 立体声、非 PCM 和缺失文件；
- 非法时间伸缩；
- 时间轴格式错误和不可写输出路径的退出码；
- 样本生成确定性；
- `LOUDNESS_LIMITED` 限制分支。

## 实际验证结果

本工作区已使用 Python 3.14.4、NumPy 2.5.4 完成离线验证。最终运行结果记录在 `VERIFICATION.md`。
