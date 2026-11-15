# 游戏音频事件离线处理器

本项目从零实现一个确定性的离线事件音频处理器：读取多格式 WAV 与事件
JSON，按采样级时间轴施加等功率声像 LFO、清理循环首尾爆音、加窗 sinc
重采样，最终输出立体声 WAV 混音和可机器读取的 JSON 报告。

实现仅依赖 Python 标准库与 NumPy，不需要网络、音频设备或外部数据集。

## 快速开始

要求 Python 3.9+、NumPy 1.22+。

```bash
# 1. 生成确定性多格式样本与 events.json
python main.py generate --out-dir demo_data

# 2. 处理为 48 kHz / 16-bit 立体声 WAV
python main.py process --events demo_data/events.json --output out/mix.wav --report out/report.json

# 3. 运行完整端到端自检（生成样本、处理、错误码、内存溢出路径）
python main.py selftest --out-dir validation

# 4. 运行标准库单元测试
python -m unittest discover -s tests -v
```

也支持直接传参数和模块入口：

```bash
python main.py --events demo_data/events.json --output out/mix.wav
python -m audio_event_mixer process --events demo_data/events.json --output out/mix.wav
python generate_samples.py --out-dir demo_data
```

`process` 可选参数：

- `--sample-rate` / `--target-sample-rate`：目标采样率，默认 `48000`。
- `--bit-depth`：输出整数 PCM 位深，支持 `16` 或 `24`。
- `--memory-limit-mb`：工作集预算，默认 `2048` MiB。
- `--chunk-frames`：流式处理块大小，默认 `65536` 帧。
- `--report`：报告路径；省略时为输出 WAV 同目录的 `*.report.json`。

## 项目结构

```text
audio_event_mixer/
  wavio.py       RIFF/WAV 检查、流式读取、16/24-bit 原子写入
  events.py      JSON 事件解析与字段/循环范围校验
  dsp.py         等功率声像、循环交叉淡化、加窗 sinc 重采样
  processor.py   采样级时间轴、流式混音、软限幅、JSON 报告
  generator.py   确定性多格式样本生成器
  selftest.py    端到端验收检查
  cli.py         命令行入口
main.py
generate_samples.py
tests/
```

## 输入事件格式

顶层可以是事件数组，或包含 `events` / `tracks` 数组的对象。相对音频路径
默认相对于事件 JSON 所在目录解析。

```json
{
  "schema_version": "1.0",
  "events": [
    {
      "id": "footstep_loop",
      "file": "samples/source_44k_s16_mono.wav",
      "time": 0.10025,
      "pan_frequency_hz": 0.5,
      "pan_depth": 0.8,
      "loop": {
        "start": 0.05,
        "end": 0.55
      }
    }
  ]
}
```

必填字段：

- 文件：`file`，兼容 `path`、`file_path`、`source`、`source_path`。
- 触发时间：`time`，单位秒，必须有限且 `>= 0`。
- 声像频率：`pan_frequency_hz`，单位 Hz，必须有限且 `>= 0`。
- 声像深度：`pan_depth`，范围 `[0.0, 1.0]`。

可选字段：

- `loop`：`{ "start": 秒, "end": 秒 }`、`[start, end]` 或
  `null`；也支持顶层 `loop_start` / `loop_end`。
- `pan_phase_degrees`：LFO 初相位，默认 `0`。
- `id`：报告中的事件标识。

缺失字段、非数值、非有限值、越界深度、负触发时间、起止顺序错误或循环
区域超出音频有效范围，均产生 `E_INVALID_EVENT` 并终止处理。循环终点
允许恰好等于音频时长。

## 处理算法

### 声像摆动

每条事件在“事件本地时间”上计算 LFO，起点为事件第一帧：

```text
lfo(t)   = depth * sin(2*pi*frequency*t + phase)
theta(t) = (lfo(t) + 1) * pi / 4
left(t)  = cos(theta(t))
right(t) = sin(theta(t))
```

因此 `left^2 + right^2 == 1`，深度为 0 时左右增益精确相等且均为
`sqrt(0.5)`。立体声源在声像处理前按声道等权下混为单声道；输出始终为
立体声，且同一采样率下事件输出帧数与源帧数一致。

### 循环首尾交叉淡化

循环区间映射到目标采样率后，淡化长度为：

```text
F = min(floor(0.10 * (loop_end_sample - loop_start_sample)),
        floor(0.005 * target_sample_rate))
```

循环尾部 `F` 帧与循环头部的循环移位副本做等功率交叉淡化：尾部使用
`cos` 淡出，头部使用 `sin` 淡入。交叉淡化在声像处理后对左右声道分别
执行，所以最终 WAV 中循环回绕点的最后一帧与循环起点帧相等。处理只修改
循环尾部窗口，不改变事件总帧数。

### 加窗 sinc 重采样

源采样率与目标采样率不同时，使用 Kaiser 窗 sinc 插值：

- 半宽 `24` 个采样，Kaiser `beta = 8.6`。
- 降采样时截止频率按 `target/source` 缩放，并留 3% 过渡保护。
- 权重逐输出样本归一化，边缘采用对称反射，常量信号和 DC 不被改变。
- 输出帧数为 `round(source_frames * target_rate / source_rate)`，时长误差
  最大半个目标采样。

### 时间轴与限幅

触发采样点为 `floor(time * target_rate + 0.5)`，即四舍五入到最近采样点。
所有事件按触发采样点写入同一个 `float64` 立体声缓冲区，重叠区域逐采样
相加，最后对整条时间轴执行 `tanh` 软限幅并量化为 16/24-bit PCM。输出
时间轴长度为所有事件 `trigger_sample + event_frames` 的最大值。

## WAV 支持

读取：

- 16-bit 整数 PCM
- 24-bit 整数 PCM
- 32-bit 整数 PCM
- 32-bit IEEE float PCM
- 普通 `WAVE_FORMAT_PCM`、`WAVE_FORMAT_IEEE_FLOAT` 与
  `WAVE_FORMAT_EXTENSIBLE`

整数 PCM 正常路径使用标准库 `wave` 分块读取；float 与 extensible 格式
使用同一套已校验 RIFF 元数据做 NumPy 小端解码。整数样本统一缩放为
`float64`，范围 `[-1, 1)`。

写出：

- 16-bit 整数 PCM
- 24-bit 整数 PCM
- 固定立体声、目标采样率

压缩编码（如 format tag 7）、非 16/24/32-bit PCM、非 32-bit float、
损坏块对齐等返回 `E_UNSUPPORTED_FORMAT`。

## JSON 报告

成功报告包含：

- 顶层：`status`、`target_sample_rate`、`output_path`、`report_path`、
  `timeline_frames`、`output`、`processing`、`events`。
- 每个事件：源路径与源格式、触发秒/触发采样点/采样误差、重采样后帧数、
  实际应用的声像频率和深度、循环起止采样点、淡化采样数、
  交叉淡化前后边界跳变、输出路径。

失败报告包含：

```json
{
  "status": "error",
  "error_code": "E_INVALID_EVENT",
  "failure_event_index": 0,
  "error": {
    "code": "E_INVALID_EVENT",
    "message": "...",
    "event_index": 0
  }
}
```

报告使用临时文件加 `os.replace` 原子写出，可被标准 JSON 解析器读取。

## 错误码与退出码

| 错误码 | 含义 | CLI 退出码 |
| --- | --- | --- |
| `E_INVALID_EVENT` | JSON、必填字段或数值/循环范围非法 | 2 |
| `E_FILE_NOT_FOUND` | 事件 JSON 或音频源文件不存在/不可读 | 3 |
| `E_UNSUPPORTED_FORMAT` | 压缩编码或非支持位深/格式 | 4 |
| `E_OUTPUT_DENIED` | 无法创建、写入或原子替换输出 | 5 |

业务错误同时写入失败报告并输出到标准错误。输出路径拒绝不属于某个具体
事件时，`failure_event_index` 为 `null`。

## 资源与原子性

- 输入 WAV 分块读取，混音与输出分块计算/写出。
- 单个工作数组默认最多驻留 256 MiB；超过阈值自动落到临时 `memmap`。
- 时间轴混音缓冲区也遵循同一阈值，处理结束后关闭映射并删除临时文件。
- 峰值工作集低于 2 GiB；`--memory-limit-mb` 可用于收紧预算。
- WAV 先写同目录临时文件，全部成功后才原子替换目标，失败不会破坏已有
  输出。
- 所有 WAV、JSON、memmap 和临时文件句柄均通过上下文或 `finally` 释放。

## Python API

```python
from audio_event_mixer import process_events, read_wav

report = process_events(
    "demo_data/events.json",
    "out/mix.wav",
    "out/report.json",
    target_sample_rate=48000,
    output_bit_depth=16,
    memory_limit_bytes=2 * 1024**3,
)
print(report["timeline_frames"])

audio = read_wav("out/mix.wav")
print(audio.samples.shape, audio.info.sample_rate)
```

其他公开函数包括：

- `load_events`
- `panning_gains` / `pan_mono`
- `apply_loop_cleanup`
- `resample_windowed_sinc`
- `read_wav` / `inspect_wav`

## 实际验证结果

验证环境：WSL2 Linux、Python 3.14.4、NumPy 2.5.4。实现只使用 Windows
Python 同样提供的标准库与 NumPy API，未使用 Docker 或 POSIX 专有功能。

```text
$ python -m unittest discover -s tests -v
...
Ran 32 tests
OK

$ python main.py selftest --out-dir validation
status=passed checks=16 report=.../validation/selftest_report.json
```

端到端主样本包含 4 个重叠事件，覆盖 44.1/48/96/32 kHz、
16/24/32-bit 整数、32-bit float、单/双声道：

| 指标 | 实测 |
| --- | --- |
| 输出 | 48 kHz、双声道、16-bit PCM、39360 帧、0.82 秒 |
| 最大触发对齐误差 | `1.82e-12` 采样（限制 1 采样） |
| 重采样时长误差 | `0.0` 秒（限制 0.001 秒） |
| 循环淡化长度 | 两个循环均为 `240` 采样，等于公式结果 |
| 循环边界最大跳变 | `6.94e-18`（处理前最大约 `0.0996`） |
| LFO 等功率最大误差 | `2.22e-16` |
| 混音峰值（tanh 前 / 后） | `0.8949 / 0.7138` |
| 输出 DC 左 / 右 | `3.61e-4 / 4.91e-4` |
| 24-bit 输出 | `pcm_s24le`、39360 帧 |
| 四类错误码 | 均在异常与失败报告中一致返回 |
| 原子保护 | 处理失败后既有输出字节完全未变 |
| memmap 溢出路径 | 8 MiB 预算下 240000 帧混音使用 `memmap` |
| 120 秒内存实测 | 5.76M 帧、8 MiB 数组预算、最大 RSS `217.7 MiB` |

完整机器可读结果见 `validation/selftest_report.json`，主处理报告见
`validation/output/mix.report.json`。
