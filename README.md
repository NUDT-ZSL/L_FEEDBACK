# 离线音频时间轴混音器

读取工程 JSON 和 WAV 素材，在内存中按采样点对齐、混合时间轴，随后调用本地 FFmpeg 编码为 WAV PCM 16/24 位或 FLAC，并生成与输出音频同名的 `.report.json` 报告。

## 环境

- Python 3.10+
- `numpy`、`soundfile`
- FFmpeg 6.0+，位于 `PATH`，或通过 `--ffmpeg` / `FFMPEG_BINARY` 指定

```bash
python -m pip install -r requirements.txt
python mix_audio.py --help
```

Windows 可使用 `py -3` 代替 `python`。也可以执行 `python -m audio_mixer`，两种入口完全等价。

## 使用方法

```bash
python mix_audio.py path/to/project.json path/to/output_dir
```

可选参数：

```bash
python mix_audio.py project.json output --ffmpeg "C:\ffmpeg\bin\ffmpeg.exe"
```

输出目录不存在时会自动创建。默认输出文件名为工程 JSON 的文件名主干；也可在工程中通过 `output_name`（或 `output_filename`）指定。

## 工程格式

```json
{
  "sample_rate": 48000,
  "channels": 2,
  "output_format": "wav24",
  "output_name": "demo_mix",
  "total_duration_sec": 4.5,
  "clips": [
    {
      "source": "assets/voice.wav",
      "start_sec": 0.25,
      "duration_sec": 2.0,
      "gain_db": -3.0,
      "fade_in_sec": 0.1,
      "fade_out_sec": 0.2
    }
  ]
}
```

- `source` 为相对路径时，相对于工程 JSON 所在目录解析。
- `gain_db`、`fade_in_sec`、`fade_out_sec` 可省略，默认均为 `0`。
- 所有 `*_sec` 字段都由同一个 `seconds_to_samples()` 函数换算为采样点，采用四舍五入（半值远离零）。
- 片段占用区间为 `[start_sample, start_sample + duration_sample)`；超过 `total_duration_sec` 对应采样长度的部分被截断。
- `output_name` 中的目录部分会被忽略，文件始终写入命令行给出的输出目录。

支持的输出格式：

- WAV PCM 16 位：`wav`（默认 16 位）、`wav16`、`wav_pcm16`、`pcm_s16le`
- WAV PCM 24 位：`wav24`、`wav_pcm24`、`pcm_s24le`
- FLAC：`flac`、`flac16`、`flac24`
- 对象形式：`{"container": "wav", "bit_depth": 24}` 或 `{"format": "flac"}`

## 处理规则

- 素材支持单声道和立体声。不同采样率使用线性插值重采样；素材声道少时循环复制声道，多时只取前 N 个声道。
- 片段增益按 `10 ** (gain_db / 20)` 转为线性倍数。淡入淡出使用等功率正弦曲线，结果限制在 `[0, 1]`。
- 当淡入与淡出总长超过片段长度时，两者按比例缩短，并记录 `FADE_SCALED` 警告。
- 素材短于声明的片段长度时，缺失部分补静音并记录 `SOURCE_SHORTER_THAN_CLIP`。
- 含两个及以上活动片段的采样点使用 `0.95 * tanh(x / 0.95)` 软限幅，避免重叠相加削波。
- 软限幅后的最终峰值若超过 `0 dBFS`，整个时间轴归一化到 `-0.1 dBFS`（线性峰值 `0.988553`），并记录实际增益。
- 音频写为临时浮点 WAV，再由 FFmpeg 编码。只有编码成功后才原子替换为最终文件；编码失败会删除输出音频，但仍写入失败报告。
- 报告中的 `duration_sec` 和 `peak_dbfs` 来自最终编码文件的回读结果。静音峰值定义为 `-120.0 dBFS`。

## 报告

例如输出 `demo_mix.wav` 时，报告为 `demo_mix.report.json`。主要字段：

```json
{
  "schema_version": 1,
  "status": "success",
  "output_file": "demo_mix.wav",
  "output_format": "wav_pcm24",
  "sample_rate": 48000,
  "channels": 2,
  "sample_count": 216000,
  "duration_sec": 4.5,
  "peak_dbfs": -3.214,
  "normalization_gain_db": 0.0,
  "normalization_gain_linear": 1.0,
  "clips": [],
  "ffmpeg": {
    "executable": "ffmpeg",
    "command_summary": "ffmpeg ... -c:a pcm_s24le ...",
    "returncode": 0
  },
  "logs": [],
  "error": null
}
```

每个片段的 `status` 为 `success`、`truncated`、`warning`、`skipped` 之一；`statuses` 可同时包含 `success` 和 `warning`。片段报告还包含 `error_code`、`warning_codes`、实际采样区间和实际时长。

片段级错误码：

- `INVALID_CLIP`、`INVALID_SOURCE`、`INVALID_FIELD`
- `INVALID_START`、`INVALID_DURATION`、`DURATION_BELOW_ONE_SAMPLE`
- `INVALID_TIME_RANGE`、`INVALID_FADE_RANGE`、`INVALID_GAIN`
- `SOURCE_NOT_FOUND`、`SOURCE_DECODE_ERROR`

片段级警告码：

- `TIMELINE_TRUNCATED`、`FADE_SCALED`、`NEGATIVE_FADE_CLAMPED`
- `RESAMPLED`、`SOURCE_SHORTER_THAN_CLIP`

FFmpeg 编码失败时报告 `status` 为 `failed`，`error.code` 为 `FFMPEG_ENCODE_FAILED`，并在 `error.ffmpeg_stderr` 中保存 FFmpeg 的 stderr。

## 退出码

| 退出码 | 含义 |
| --- | --- |
| `0` | 编码成功；片段级错误不影响该退出码 |
| `2` | 工程文件不存在 |
| `3` | JSON 格式错误 |
| `4` | 工程结构或格式字段无效 |
| `5` | 找不到 FFmpeg |
| `6` | FFmpeg 编码失败 |
| `7` | 输出目录、输出文件检查或报告写入等 I/O 错误 |

## 测试

测试完全由代码生成确定性 WAV 和 JSON，不依赖外部素材：

```bash
python -m unittest discover -v
```

覆盖采样点半增舍入、时间轴对齐、线性重采样、声道填充/裁剪、淡入淡出边界、重叠软限幅、超峰值归一化、损坏素材、WAV24/FLAC 编码、报告回读一致性、片段错误成功退出和 FFmpeg 失败处理。本次实际执行结果见 `VERIFICATION.md`。

