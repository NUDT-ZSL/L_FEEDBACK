# 实际验证记录

## 环境

```text
Python 3.14.4
NumPy 2.5.4
平台：Linux 本地工作区（命令与实现不依赖 Linux 特性，CLI 可在 Windows 10+ 运行）
```

`audioop` 在当前 Python 3.14 中已移除；实现直接完成 PCM 解码、重采样、量化和 WAV 写入，因此不受影响。

## 命令

```bash
python generate_samples.py --output-dir samples
python audio_timeline_tool.py \
  --timeline samples/timeline.json \
  --audio-dir samples/audio \
  --output-dir demo_output
python -m unittest discover -s tests -v
```

## 端到端结果

命令行退出码为 `0`，报告统计为：

```text
Processed 3 audio file(s); skipped 3.
```

成功处理：

- `impact_48k.wav`：48000 Hz 单声道，1.2 s -> 1.2 s。
- `stretch_44k.wav`：44100 Hz 单声道，`time_scale=2.0`，0.6 s -> 1.2 s。
- `invalid_scale_48k.wav`：`time_scale=11.0` 被判为非法，按 1.0 处理，0.5 s -> 0.5 s，并产生 `INVALID_TIME_SCALE`。

逐音频跳过：

- `stereo_unsupported.wav`：`UNSUPPORTED_FORMAT`，原始时长 0.4 s。
- `float_unsupported.wav`：非 PCM，`UNSUPPORTED_FORMAT`，原始时长 0.3 s。
- `does_not_exist.wav`：`AUDIO_READ_ERROR`。

事件与循环结果示例：

- `impact` 第一个事件：`0.200000 s -> 0.204395833 s`，偏移 `+4.395833 ms`。
- `impact` 第二个事件：`0.700000 s -> 0.704291667 s`，偏移 `+4.291667 ms`。
- 1.5 s 事件超出 1.2 s 音频：`EVENT_OUT_OF_RANGE`，`aligned_time=1.5`。
- `impact` 循环 `[0.1, 0.35]`：生成 `impact_loop_0.wav`，12000 帧，首尾采样均为 0。
- `impact` 循环 `[0.5, 0.515]`：`LOOP_TOO_SHORT`，不生成文件。
- `stretch` 事件：`0.800000 s -> 0.807166667 s`，符合 2 倍伸缩后的峰值位置。

输出格式检查：

- 所有生成的 WAV 均为 48000 Hz、单声道、16 位 PCM。
- 最大输出峰值为 `29203` PCM 值，即 `-1.000449 dBFS`，不超过 `-1 dBFS`。
- `stretch_processed.wav` 输出 RMS 约为 `0.100000168`，对应简化目标 `-20 LUFS`。
- 生成的循环文件首采样和末采样均为 `0`。

## 自动化测试结果

```text
Ran 13 tests

OK
```

测试包含两次确定性生成的文件 SHA-256 对比，所有样本文件逐字节一致。
