# 游戏过场动画对白本地化替换工具

本工具将双轨过场音频中的轨 0 对白替换为本地化单声道对白，将轨 1
音效按指定偏移重新对齐，并输出新的容器文件和可机读 JSON 报告。所有测试
素材均由本地确定性生成，不访问网络或外部数据。

## 交付内容

- `localize_dialogue.py`：命令行主工具。
- `synthesize_samples.py`：使用 FFmpeg `lavfi` 生成本地测试素材。
- `tests/test_localize_dialogue.py`：端到端自动化测试，共 8 项。
- `VALIDATION.md`：实际验证环境、命令和结果。
- `requirements.txt`：Python 依赖。

## 环境

- Python 3.9+
- NumPy 1.24+
- FFmpeg 6.0+，`ffmpeg` 和 `ffprobe` 位于 `PATH`

安装 Python 依赖：

```bash
python3 -m pip install -r requirements.txt
```

## 使用

参数可使用短选项、长选项或位置参数。位置参数顺序为：
`原始容器 本地化对白 输出容器 报告路径`。

```bash
python3 localize_dialogue.py \
  --input original.mkv \
  --localized-dialogue localized.wav \
  --output replaced.mkv \
  --report report.json \
  --sound-offset-ms 250
```

等价的短选项形式：

```bash
python3 localize_dialogue.py \
  -i original.mkv \
  -l localized.wav \
  -o replaced.mkv \
  -r report.json \
  --sound-offset-ms -300
```

`--sound-offset-ms` 可省略，默认值为 `0`；支持负数和小数。进度信息只写标准
错误。成功或失败时标准输出只包含一个 JSON 对象，内容与报告文件一致。

## 时间轴约定

- 本地化对白起始点固定为输出时间轴的 `0`。
- 输出容器时长等于转换后的本地化对白时长。
- 音效起始位置 = `sound_offset_ms`，定义在对白时间轴上。
- 正偏移：音效轨前部填充静音，然后复制音效；对白结束处之后的音效尾部被截断。
- 负偏移：丢弃音效开头会对应对白零点之前的部分，其余内容从输出零点开始；
  对白结束处之后的音效尾部同样被截断。
- 被保留、补静音或截断的音效样本均按原值复制，不改变其波形内容。

例如，对白为 1.5 s、音效为 2.0 s、偏移为 `-300 ms` 时，输出为 1.5 s：
原始音效前 300 ms 被截断，之后 1.5 s 内容从输出零点开始，音效尾部另有
200 ms 被截断。报告分别记录前部、尾部和总截断样本数。

## 处理规则

1. 用 `ffprobe` 枚举音频流，按音频轨顺序选择前两路。超过两路时忽略其余
   音轨，并记录 `ignored_audio_tracks`。
2. 校验轨 0 为立体声、本地化对白为单声道、输入至少有两路音频。
3. 以原始对白采样率为目标采样率，将本地化单声道对白重采样后原样复制到
   左右声道，形成立体声对白。
4. 音效若非立体声则重混为立体声；采样率不同则重采样到目标采样率。已满足
   立体声和采样率要求时，不做不必要的重混或重采样。
5. 按上述时间轴构造两路等长立体声 PCM 样本，并编码为最终容器。
6. 编码后重新探测并解码输出音轨，核对两轨数量、采样率、声道数、样本数和
   预期波形；验证失败按 `ENCODING_FAILED` 处理。
7. 临时原始样本、编码中间文件和临时目录在处理结束后自动清理；输出容器
   使用同目录临时文件原子替换。

### 编码策略

若原始音效本身是无损 PCM 编码，优先沿用该编码；否则回退到
`pcm_f32le -> pcm_s24le -> pcm_s16le` 中当前容器支持的第一种 PCM 编码。
这样可避免把已解码音效再次送入有损编码器，从而保持解码后的音效样本序列。
报告中的 `output_format` 记录容器、实际编码和回退原因。

若源音效为立体声且采样率已等于目标采样率，保留部分与源解码样本逐样本一致。
源音效为非立体声或不同采样率时，报告明确标记 `remixed`/`resampled`，因为
重混或重采样本身必然改变样本序列。

## 报告字段

成功报告包含：

- `status`: `"success"`
- `error_code`: `null`
- `output_file`: 输出容器绝对路径
- `dialogue_track`: 转换前后采样率、声道数、时长、样本数
- `sound_effect_track`: 采样率、声道数、输出时长、偏移量、填充/截断统计
- `ignored_audio_tracks`: 被忽略的第 3 路及后续音频轨数量
- `alignment`: 请求/实际偏移、对齐误差、5 ms 容差和判定结果
- `output_format`: 实际容器及音频编码、PCM 回退说明
- `verification`: 输出音轨数量和逐样本验证结果
- `processing_timestamp`: UTC ISO-8601 时间戳

失败报告保留相同顶层字段，`status` 为 `"failure"`，并写入字符串
`error_code` 与错误说明。

## 错误码

| 错误码 | 含义 |
| --- | --- |
| `FILE_NOT_FOUND` | 原始容器或本地化对白不存在 |
| `UNSUPPORTED_FORMAT` | 媒体无法解析、音频轨不足、对白轨非立体声、本地化对白非单声道等 |
| `INVALID_ARGUMENT` | 缺少/冲突参数、非法偏移值、路径类型错误或试图覆盖输入 |
| `ENCODING_FAILED` | 输出目录不可写、所有 PCM 编码失败或输出验证失败 |
| `TOOL_NOT_AVAILABLE` | `ffmpeg`/`ffprobe` 不在 `PATH` |
| `PROCESSING_FAILED` | 非预期处理错误或对齐误差超过 5 ms |

业务错误退出码为非零；参数解析错误为 `2`，其他处理错误为 `1`。

## 生成样本

```bash
python3 synthesize_samples.py --output-dir /tmp/uid155-samples
```

默认生成：

- `original.mkv`：两路 48 kHz 立体声 PCM，对白 2.0 s，音效 2.5 s。
- `localized_dialogue.wav`：44.1 kHz 单声道，1.5 s。
- `manifest.json`：样本结构说明。

可使用 `--container mp4`、`--audio-codec aac` 和 `--extra-track` 覆盖容器、
源编码及三轨场景。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖：

- 正偏移的前置静音、5 ms 内对齐及音效逐样本保持。
- 负偏移的前部截断、尾部截断和截断样本统计。
- 单声道音效重混、不同采样率重采样、三轨输入和忽略轨计数。
- MP4 输出、AAC 音效源及 PCM 回退。
- 文件缺失、参数无效、格式不支持、编码失败四类业务错误码。
- 标准输出 JSON 可解析、报告文件与标准输出一致、进度输出和临时文件清理。
