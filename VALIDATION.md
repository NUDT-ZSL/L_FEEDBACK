# 实际验证记录

验证日期：2026-11-17（Asia/Shanghai）

## 环境

- Python `3.14.4`
- NumPy `2.5.5`
- FFmpeg `8.0.1-3ubuntu2`
- ffprobe `8.0.1-3ubuntu2`

## 自动化测试

执行：

```bash
python3 -m unittest discover -s tests -v
```

实际结果：

```text
Ran 8 tests in 2.941s

OK
```

八项测试全部通过，覆盖正/负音效偏移、音效逐样本比对、44.1 kHz 单声道对白
转 48 kHz 立体声、三轨输入、音效重混/重采样、MP4 输出、AAC 源音效 PCM
回退，以及 `FILE_NOT_FOUND`、`INVALID_ARGUMENT`、`UNSUPPORTED_FORMAT`、
`ENCODING_FAILED` 四类失败报告。

## 确定性样本端到端验证

执行：

```bash
ROOT=/tmp/uid155-validation
python3 synthesize_samples.py --output-dir "$ROOT/samples" --container mkv
python3 localize_dialogue.py \
  -i "$ROOT/samples/original.mkv" \
  -l "$ROOT/samples/localized_dialogue.wav" \
  -o "$ROOT/validated_output.mkv" \
  -r "$ROOT/validated_report.json" \
  --sound-offset-ms 250
```

关键结果：

- 退出码 `0`，标准输出与报告文件均为可解析 JSON 且内容一致。
- 输入为两路 48 kHz 立体声 PCM；本地化对白为 44.1 kHz 单声道。
- 输出轨 0/1 均为 48 kHz、双声道 PCM，各 `72000` 样本，时长 `1.5 s`。
- 对白转换前后：`44100 Hz / 1 ch / 1.5 s` ->
  `48000 Hz / 2 ch / 1.5 s`。
- 音效偏移 `250 ms`，前置静音 `12000` 样本，保留 `60000` 样本，
  尾部截断 `60000` 样本。
- 对齐误差 `0.0 ms`，在 `5 ms` 容差内。
- 工具内部解码复核：两路音轨均通过逐样本比较，`verification.passed=true`。
- 标准错误按 `0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100` 百分比输出进度。

## 容器/编码附加验证

- Matroska + 原始立体声 PCM：输出沿用 `pcm_s16le`，音效保留部分逐样本一致。
- Matroska + 24-bit PCM：输出沿用 `pcm_s24le`，音效保留部分逐样本一致。
- MP4 + 原始 PCM：输出 `pcm_s16le`，两路立体声轨和时长均正确。
- Matroska + AAC 源音效：输出回退为 `pcm_f32le`，报告记录
  `source_effects_codec=aac` 和 PCM 回退原因，输出验证通过。
- MP4 + 三路 AAC、偏移 `-250 ms`：报告 `ignored_audio_tracks=1`，
  输出两路 `pcm_f32le` 立体声，前部截断 `12000` 样本，对齐误差 `0 ms`。

实际使用的验证样本和输出位于执行时创建的 `/tmp/uid155-validation.*`
临时目录；仓库不提交生成的媒体文件。
