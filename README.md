# audiopack — 离线音频时间线打包工具

把一组 WAV 片段与 `timeline.json` 事件描述渲染成一条 48 kHz / 24-bit PCM /
立体声 WAV，并输出切片报告 JSON。支持片段裁剪与循环、头尾淡化、事件增益、
优先级重叠覆盖，以及 ITU-R BS.1770-4 整体响度归一化（目标 -23 LUFS）。

## 环境要求

- Windows 10/11 64 位，Python 3.10 或 3.11（代码仅使用 3.10+ 语法，亦可在更新版本运行）
- FFmpeg 6.0+ 置于 PATH（运行期音频处理本身不调用 FFmpeg；FFmpeg 用于环境侧
  格式探测/转码等配套用途，核心重采样为内置 sinc 实现，无外部进程依赖）
- 依赖：`numpy`、`soundfile`、`pyloudnorm`（测试用 `pytest`）

```bash
python -m pip install -r requirements.txt
```

## 用法

```bash
python -m audiopack <input_dir> <output.wav> [--report report.json] [--memory-limit-mb 512]
```

输入目录必须包含 `timeline.json` 与至少一个 `.wav`。报告默认写到
`<output>_report.json`。

### timeline.json

```json
{
  "sample_rate": 48000,
  "events": [
    {
      "start_ms": 0,
      "end_ms": 1000,
      "clip": "tone.wav",
      "fade_in_ms": 20,
      "fade_out_ms": 20,
      "gain_db": -3.0,
      "priority": 1,
      "loop": false
    }
  ]
}
```

`priority`、`loop` 缺省分别为 `0`、`false`；`sample_rate` 缺省 `48000`。

## 规则与错误码

- 毫秒→采样点全程统一：`floor(round(ms * sr / 1000))`（Python 内建 `round`，
  跨平台确定）。
- 任意 `E_` 级错误退出码为 1，错误以单行 JSON 写到 stderr；仅 `W_` 警告时退出码 0。

| 代码 | 含义 |
| --- | --- |
| `E_TIMELINE_FORMAT` | JSON 结构/字段类型错误、缺 timeline、无 WAV |
| `E_MISSING_CLIP` | 事件引用的片段文件不存在 |
| `E_RANGE` | 字段越界（负时间、end<=start、淡化超时长、增益越界等） |
| `E_CORRUPT_CLIP` | WAV 无法解码/数据异常；跳过该事件并继续其余事件 |
| `E_MEMORY_LIMIT` | 音频缓冲估算超过 512 MiB（可用 `--memory-limit-mb` 调整），并清理部分输出 |
| `W_SHORT_CLIP` | 非循环事件素材不足，尾部补静音 |
| `W_LOW_LOUDNESS` | 原始响度 ≤ -70 LUFS（或不可测），增益封顶 +20 dB |

## 确定性

同一输入重复运行产生字节一致的 WAV 与报告：无时间戳字段、JSON 键排序输出、
固定噪声种子、单线程固定顺序浮点累加、24-bit 量化由 `soundfile/libsndfile`
确定性完成。

## 测试

```bash
python -m pytest
```

测试样本由 `tests/samples.py` 以确定性函数（正弦/静音/固定种子噪声）生成，
覆盖重叠覆盖、循环/补静音、响度达标、报告对齐、错误码/退出码、损坏续跑、
内存限额与字节级确定性，详见 `TECH_NOTES.md`。
