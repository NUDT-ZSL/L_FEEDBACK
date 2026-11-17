# BGM Loop Mixer

一个面向 Windows 的 Python 命令行 WAV 混音工具。它按样本截取循环区间，
进行 Kaiser 窗 sinc 抗混叠重采样，以有理数周期调度循环，支持最多 8 轨线性
混音、逐次淡入淡出、限幅统计和原子输出。

## 环境安装

目标环境为 Windows 10/11 64 位、Python 3.10-3.12：

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

`soundfile` 会安装自带 libsndfile 的 wheel，不需要另装系统音频库。

## 快速开始

生成内置确定性测试音频：

```bat
python mixer.py synth --output loop.wav --sample-rate 44100 --duration 3.0 --frequency 440 --amplitude 0.6
```

示例 `play.json`：

```json
{
  "duration_seconds": 12.0,
  "loop": {
    "start_seconds": 0.25,
    "end_seconds": 2.25
  },
  "fade": {
    "in_ms": 10,
    "out_ms": 10,
    "curve": "equal_power"
  },
  "tracks": [
    {
      "volume": 0.8,
      "offset_seconds": 0.0
    }
  ]
}
```

运行混音：

```bat
python mixer.py --input loop.wav --target-rate 48000 --config play.json --output out.wav
```

成功时 stdout 只有一行 JSON：

```json
{"status":"ok","error_code":null,"output":"...\\out.wav","output_sample_rate":48000,"output_frames":576000,"output_channels":1,"output_duration_seconds":12.0,"clip_count":0,"clipped_frames":0,"loop_count":6,"fade_curve":"equal_power","fade_in_ms":10.0,"fade_out_ms":10.0,"peak_before_limit":0.48,"peak_after_limit":0.48,"tracks":[{"path":"...\\loop.wav","source_sample_rate":44100,"volume":0.8,"loop_count":6}]}
```

输出 WAV 固定为 24-bit PCM。实际输出采样率、帧数、限幅次数、循环次数、
淡变曲线和错误码都在摘要中。

## 配置格式

顶层字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `duration_seconds` | 否 | 播放总时长。省略时为各轨 `offset + 单次循环长度` 的最大值。 |
| `duration_ms` | 否 | 总时长的毫秒表示，不能与 `duration_seconds` 同时出现。 |
| `loop.start_seconds` | 否 | 循环起点，省略为 `0`。 |
| `loop.end_seconds` | 否 | 循环终点，终点样本不包含；省略为源文件末尾。 |
| `fade.in_ms` / `fade.out_ms` | 否 | 每次循环的淡入/淡出毫秒数，默认各 `10`。两者之和不能超过重采样后的循环长度。 |
| `fade.curve` | 否 | `equal_power`（默认）或 `linear`。 |
| `tracks` | 否 | 1-8 条音轨；省略时把 `--input` 作为唯一音轨。 |

音轨字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `path` | 否 | 该轨 WAV；省略时使用命令行的 `--input`。相对路径以配置文件目录为准。 |
| `volume` | 否 | 线性音量，范围 `[0, 1]`，默认 `1.0`。 |
| `offset_seconds` | 否 | 相对主时间轴的起始偏移，不得为负，默认 `0.0`。 |
| `loop` / `fade` | 否 | 可按音轨覆盖顶层循环点和淡变设置。 |

兼容常见别名，例如 `file`、`gain`、`offset_ms`、`loop_start_seconds`、
`fade_in_ms`、`fade_curve` 等。未知字段会被忽略。

`--input` 仍按命令行实例提供；只要某条音轨没有 `path`，就会使用它。配置中
的所有音轨都指定了路径时，`--input` 只作为必填的默认来源参数存在，不额外
增加音轨。

## 行为定义

- 循环秒数先以 `round(seconds * source_rate)` 对齐到最近源样本，终点样本不包含。
- 每轨先将对齐后的循环段重采样到目标采样率，再按偏移叠加。
- 单次循环的理论周期可以是非整数目标帧。工具对第 `k` 次循环使用
  `round(k * source_frames * target_rate / source_rate)` 作为绝对起点，而不是
  反复累加取整后的周期，因此总相位误差始终不超过半个样本。
- 淡入/淡出在每次循环中重新应用，循环段首尾增益精确为 0。
- 输出声道数为各输入轨最大声道数；单声道轨混入立体声输出时会复制到左右声道。
- 超过 `±1.0` 的每个声道样本会先计数再裁剪。`clip_count` 是发生裁剪的声道样本
  数，`clipped_frames` 是至少一个声道发生裁剪的帧数。
- `loop_count` 是主音轨在播放区间内实际开始的循环次数（包含第一次）。

## 错误与退出码

失败时 stdout 同样只有一行 JSON，例如：

```json
{"status":"error","error_code":"E_IO","exit_code":5,"message":"output directory is not writable: [Errno 13] Permission denied: ..."}
```

| 错误码 | 退出码 | 含义 |
| --- | --- | --- |
| `E_INVALID_INPUT` | 2 | 参数、JSON、配置、WAV 格式、采样率范围、声道数、音量、偏移或循环范围非法 |
| `E_RESAMPLE` | 3 | 重采样无法完成或重采样后循环过短 |
| `E_MIX` | 4 | 混音内部数据错误 |
| `E_IO` | 5 | 读写、目录权限或最终文件替换失败，消息保留原始系统错误 |

输出先写入同目录隐藏临时文件，全部成功后才用 `os.replace` 替换目标文件。
任何失败都会删除临时文件，不留下部分 WAV。

## 输入限制

- WAV：采样率 `8000..192000 Hz`，单声道或双声道，至少 1 帧。
- 目标采样率：`8000..192000 Hz` 的整数。
- 最多 8 条音轨，音量 `[0,1]`，偏移不得为负。
- 循环起点不得为负，终点必须大于起点且不能超出源 WAV。
- 源样本中的 NaN/Inf 会被拒绝。

## 测试与验收

运行 19 项自动化测试：

```bat
python -m unittest discover -s tests -v
```

运行完整验收（包含 31 分钟循环连续性、10 分钟双轨性能和确定性检查）：

```bat
python tools/run_acceptance.py
```

报告写入 `artifacts/acceptance_report.json`。本机实际结果见
[VERIFICATION.md](VERIFICATION.md)，实现细节见 [TECHNICAL.md](TECHNICAL.md)。
